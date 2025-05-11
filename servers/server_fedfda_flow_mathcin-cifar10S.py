import time
from copy import deepcopy
import torch
import numpy as np
from statsmodels.stats.correlation_tools import cov_nearest
from servers.server_base import Server # Assuming server_base exists
from clients.client_fedfda import ClientFedFDA
from torch.distributions.multivariate_normal import MultivariateNormal
import warnings
import logging # Added
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger("FedFDA_Flow")

class ServerFedFDA(Server):
    # Modified __init__ to accept logger
    def __init__(self, args):
        super().__init__(args)
        self.logger = logger # Store logger instance
        self.eps = 1e-4
        # Pass logger to clients during initialization
        self.clients = [
            ClientFedFDA(args, i, self.logger) for i in range(self.num_clients)
        ]

        # --- Initialization of global stats ---
        # It might be better to initialize based on average client stats if possible
        # Or use a small subset of global data if available
        self.global_means = torch.zeros([self.num_classes, self.D])
        self.global_covariance = torch.eye(self.D) * 0.1 # Small variance identity
        self.logger.info(f"Initialized global covariance with shape: {self.global_covariance.shape}")
        self.global_priors = torch.ones(self.num_classes) / self.num_classes
        self.r = 0

        # --- Optimization: FedAvgM Parameters ---
        self.server_momentum = getattr(args, 'server_momentum', 0.9)
        self.logger.info(f"Using server momentum (FedAvgM): {self.server_momentum}")
        # Initialize buffer on CPU, move to device if needed during aggregation
        self.server_momentum_buffer = {name: torch.zeros_like(param.data, device='cpu')
                                       for name, param in self.model.named_parameters()}

        # --- Optimization: Corruption Weighting Parameters ---
        self.corruption_weight = getattr(args, 'corruption_weight', 0.5)
        
        # --- Enhanced Corruption-Type Specific Weighting ---
        # Different corruptions affect features differently, so we weight them accordingly
        self.noise_corruption_weight = getattr(args, 'noise_corruption_weight', 0.4)
        self.blur_corruption_weight = getattr(args, 'blur_corruption_weight', 0.6)
        self.other_corruption_weight = getattr(args, 'other_corruption_weight', 0.5)
        
        # Mapping of corruption types to weight categories
        self.corruption_type_map = {
            'gaussian_noise': 'noise', 
            'shot_noise': 'noise', 
            'impulse_noise': 'noise',
            'defocus_blur': 'blur', 
            'motion_blur': 'blur',
            'frost': 'other', 
            'fog': 'other', 
            'brightness': 'other', 
            'contrast': 'other', 
            'jpeg_compression': 'other'
        }
        
        if min(self.noise_corruption_weight, self.blur_corruption_weight, self.other_corruption_weight) < 1.0:
            self.logger.info(f"Using corruption-type specific weighting: noise={self.noise_corruption_weight}, blur={self.blur_corruption_weight}, other={self.other_corruption_weight}")


    def train(self):
        self.logger.info("Starting FedFDA training...")
        for r in range(1, self.global_rounds+1):
            start_time = time.time()
            self.r = r

            self.sample_active_clients()
            if not self.active_clients:
                 self.logger.warning(f"Round [{r}/{self.global_rounds}]\t No clients selected. Skipping round.")
                 continue
            self.logger.info(f"Round [{r}/{self.global_rounds}]\t Selected {len(self.active_clients)} clients.")

            self.send_models()

            # Train clients
            try:
                 train_acc, train_loss = self.train_clients() # Assumes this aggregates results correctly
                 train_time = time.time() - start_time
                 self.logger.info(f"Round [{r}/{self.global_rounds}]\t Client training completed. Avg Acc: {train_acc:.2f}, Avg Loss: {train_loss:.4f}")
            except Exception as e:
                 self.logger.exception(f"Round [{r}/{self.global_rounds}]\t Error during client training phase: {e}. Skipping aggregation.")
                 continue

            # Aggregate models and statistics
            try:
                self.aggregate_models()
                self.logger.debug(f"Round [{r}/{self.global_rounds}]\t Aggregation complete.")
            except Exception as e:
                self.logger.exception(f"Round [{r}/{self.global_rounds}]\t Error during aggregation: {e}")
                continue # Skip evaluation if aggregation fails

            round_time = time.time() - start_time
            self.train_times.append(train_time) # Assuming these lists exist from base class
            self.round_times.append(round_time)

            # Logging
            log_msg_core = f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Round Time [{round_time:.2f}s]"
            if r % self.eval_gap == 0 or r == self.global_rounds:
                try:
                     # Run personalized evaluation
                     ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized()
                     log_msg = log_msg_core + f"\t P-Test Loss [{ptest_loss:.4f}]\t P-Test Acc [{ptest_acc:.2f} +/- {ptest_acc_std:.2f}]"
                     self.logger.info(log_msg)
                except Exception as e:
                     self.logger.exception(f"Round [{r}/{self.global_rounds}]\t Error during personalized evaluation: {e}")
                     self.logger.info(log_msg_core + "\t P-Test [Evaluation Failed]") # Log that eval failed
            else:
                 self.logger.info(log_msg_core)


    def aggregate_models(self):
        # --- Aggregate Base Model (FedAvgM) ---
        active_clients_with_data = [c for c in self.active_clients if c.num_train > 0]
        if not active_clients_with_data:
             self.logger.warning("No active clients with training data for model aggregation. Skipping model update.")
        else:
            total_samples_model = sum(c.num_train for c in active_clients_with_data)
            if total_samples_model == 0:
                self.logger.warning("Total samples for model aggregation is zero. Skipping model update.")
            else:
                avg_delta = {name: torch.zeros_like(param.data, device='cpu') # Aggregate on CPU
                             for name, param in self.model.named_parameters()}
                global_params_cpu = {name: param.data.cpu().clone() for name, param in self.model.named_parameters()}

                for c in active_clients_with_data:
                    # Apply advanced weighting based on corruption type
                    client_weight = c.num_train / total_samples_model
                    
                    # Apply corruption-type specific weighting
                    if hasattr(c, 'is_corrupted_client') and c.is_corrupted_client and hasattr(c, 'augmentation_type'):
                        corruption_type = c.augmentation_type
                        corruption_category = self.corruption_type_map.get(corruption_type, 'other')
                        
                        # Apply different weights based on corruption category and severity
                        if corruption_category == 'noise':
                            weight_factor = self.noise_corruption_weight
                            # For severe noise, reduce weight further
                            if hasattr(c, 'severity_level') and c.severity_level > 3:
                                weight_factor *= 0.8
                        elif corruption_category == 'blur':
                            weight_factor = self.blur_corruption_weight
                        else:  # 'other' corruptions
                            weight_factor = self.other_corruption_weight
                        
                        client_weight *= weight_factor
                        self.logger.debug(f"Client {c.client_idx}: Applied {corruption_category} corruption weight {weight_factor:.2f} (final weight: {client_weight:.4f})")
                    
                    client_params = c.model.state_dict() # Get params (likely already on CPU after client training)
                    for name, param_data_cpu in client_params.items():
                         if name in avg_delta:
                             avg_delta[name] += client_weight * (param_data_cpu.cpu() - global_params_cpu[name])

                # Update server momentum buffer and global model (on CPU, move to device if needed)
                self.model.cpu() # Ensure global model is on CPU for update
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                         if name in self.server_momentum_buffer:
                             self.server_momentum_buffer[name] = self.server_momentum * self.server_momentum_buffer[name] + avg_delta[name]
                             param.data += self.server_momentum_buffer[name] # Update global model param on CPU

        # --- Aggregate Gaussian Estimates (Corruption-Aware) ---
        total_samples_stats = 0.0 # Use float for potentially fractional weights
        new_global_means = torch.zeros_like(self.global_means) # Assume global stats are on CPU
        new_global_covariance = torch.zeros_like(self.global_covariance)

        # First compute effective total weights for normalization
        for c in active_clients_with_data:
            weight = float(c.num_train)
            if hasattr(c, 'is_corrupted_client') and c.is_corrupted_client:
                # Apply corruption-type specific weighting for statistics
                if hasattr(c, 'augmentation_type'):
                    corruption_type = c.augmentation_type
                    corruption_category = self.corruption_type_map.get(corruption_type, 'other')
                    
                    if corruption_category == 'noise':
                        weight *= self.noise_corruption_weight
                    elif corruption_category == 'blur':
                        weight *= self.blur_corruption_weight
                    else:  # 'other' corruptions
                        weight *= self.other_corruption_weight
                else:
                    # Fallback to general corruption weight if augmentation_type not available
                    weight *= self.corruption_weight
            total_samples_stats += weight

        if total_samples_stats > 1e-9: # Check for non-zero effective weight
            for c in active_clients_with_data:
                client_weight = float(c.num_train)
                # Apply adaptive corruption weighting based on client type and corruption type
                if hasattr(c, 'is_corrupted_client') and c.is_corrupted_client:
                    # Apply corruption-type specific weighting for statistics
                    if hasattr(c, 'augmentation_type'):
                        corruption_type = c.augmentation_type
                        corruption_category = self.corruption_type_map.get(corruption_type, 'other')
                        
                        if corruption_category == 'noise':
                            client_weight *= self.noise_corruption_weight
                        elif corruption_category == 'blur':
                            client_weight *= self.blur_corruption_weight
                        else:  # 'other' corruptions
                            client_weight *= self.other_corruption_weight
                        
                        # For higher severity, reduce weight further
                        if hasattr(c, 'severity_level') and c.severity_level > 3:
                            client_weight *= (1.0 - 0.1 * (c.severity_level - 3))
                    else:
                        # Fallback to general corruption weight if augmentation_type not available
                        client_weight *= self.corruption_weight
                
                normalized_weight = client_weight / total_samples_stats

                adaptive_means_cpu = c.adaptive_means.data.cpu()
                adaptive_covariance_cpu = c.adaptive_covariance.data.cpu()

                new_global_means += normalized_weight * adaptive_means_cpu
                new_global_covariance += normalized_weight * adaptive_covariance_cpu

            self.global_means = new_global_means
            try:
                 global_cov_np = self.global_covariance.numpy() + 1e-6 * np.eye(self.D) # Add small jitter before nearest
                 global_cov_psd_np = cov_nearest(global_cov_np, method="clipped", threshold=1e-4)
                 self.global_covariance = torch.Tensor(global_cov_psd_np)
            except Exception as e:
                 self.logger.warning(f"Global covariance nearest PSD failed during aggregation: {e}. Adding epsilon regularization.")
                 self.global_covariance += 1e-4 * torch.eye(self.D) # Fallback regularization
        else:
             self.logger.warning("Total effective samples for statistics aggregation is zero. Skipping statistics update.")


    def send_models(self):
        """Sends the current global model state and statistics to active clients."""
        self.model.cpu() # Ensure model is on CPU before getting state_dict
        global_model_state = self.model.state_dict()
        global_means_cpu = self.global_means.cpu()
        global_covariance_cpu = self.global_covariance.cpu()

        for c in self.active_clients:
            try:
                c.model.load_state_dict(global_model_state)
                c.global_means.data = global_means_cpu.clone()
                c.global_covariance.data = global_covariance_cpu.clone()

                if self.r == 1: # Initialize local/adaptive stats on first round
                    c.means.data = global_means_cpu.clone()
                    c.covariance.data = global_covariance_cpu.clone()
                    c.adaptive_means.data = global_means_cpu.clone()
                    c.adaptive_covariance.data = global_covariance_cpu.clone()
            except Exception as e:
                 self.logger.error(f"Error sending model/stats to client {c.client_idx}: {e}")


    def evaluate_personalized(self):
        self.logger.info("Starting personalized evaluation...")
        # Ensure the global model used for FE is on the correct device
        eval_device = self.device # Use the server's device
        self.model.to(eval_device)

        total_samples = sum(c.num_test for c in self.clients if c.num_test > 0)
        if total_samples == 0:
            self.logger.warning("No test samples available across clients for evaluation.")
            return 0.0, 0.0, 0.0

        weighted_loss = 0
        weighted_acc = 0
        accs = []
        kl_divs = []
        
        # Track metrics by corruption category
        noise_accs = []
        blur_accs = []
        other_accs = []
        clean_accs = []
        
        # Store client-specific metrics for analysis
        client_metrics = {}

        current_global_model_state = deepcopy(self.model.state_dict()) # State on eval_device
        current_global_means_dev = self.global_means.to(eval_device)
        current_global_covariance_dev = self.global_covariance.to(eval_device)
        # Ensure global cov is PSD for evaluation KL calc
        try:
             current_global_covariance_np = current_global_covariance_dev.cpu().numpy() + 1e-6 * np.eye(self.D)
             global_cov_eval_psd_np = cov_nearest(current_global_covariance_np, method="clipped", threshold=1e-4)
             global_cov_eval_psd_dev = torch.Tensor(global_cov_eval_psd_np).to(eval_device) + self.eps * torch.eye(self.D, device=eval_device) # Use server eps
        except Exception as e:
             self.logger.warning(f"Evaluation: Global covariance nearest PSD failed: {e}. Using regularization.")
             global_cov_eval_psd_dev = current_global_covariance_dev + self.eps * torch.eye(self.D, device=eval_device)


        for c in self.clients:
            if c.num_train == 0 or c.num_test == 0: # Skip clients with no train or test data
                continue

            # Temporarily load global FE state
            c.model.load_state_dict(current_global_model_state)
            c.model.eval()
            c.model.to(eval_device)
            # Pass global stats (already on eval_device)
            c.global_means = current_global_means_dev
            c.global_covariance = current_global_covariance_dev # The potentially non-psd one for beta solving

            try:
                # 1. Compute features on train data
                c_feats, c_labels = c.compute_feats(split="train") # Returns features on eval_device
                if c_feats is None or len(c_labels) == 0:
                     self.logger.warning(f"Evaluation: Client {c.client_idx} has no training features/labels. Skipping evaluation.")
                     continue

                # 2. Solve beta with early stopping
                if hasattr(c, 'solve_beta_with_early_stopping'):
                    c.solve_beta_with_early_stopping(feats=c_feats, labels=c_labels, patience=3)
                else:
                    c.solve_beta(feats=c_feats, labels=c_labels) # Stores beta on CPU

                # 3. Estimate local MLE statistics
                means_mle, scatter_mle, _, counts = c.compute_mle_statistics(feats=c_feats, labels=c_labels) # Stats on eval_device

                # 4. Update adaptive statistics for evaluation
                cov_denom = max(1.0, np.sum(counts) - c.num_classes)
                cov_mle = (scatter_mle / cov_denom) + c.eps * torch.eye(c.D, device=eval_device)
                try:
                    local_cov_psd_np = cov_nearest(cov_mle.cpu().numpy() + 1e-6 * np.eye(self.D), method="clipped", threshold=1e-4)
                    local_cov_psd_dev = torch.Tensor(local_cov_psd_np).to(eval_device) + c.eps * torch.eye(c.D, device=eval_device)
                except Exception:
                    self.logger.warning(f"Evaluation: Client {c.client_idx} local cov nearest PSD failed. Using regularized MLE.")
                    local_cov_psd_dev = cov_mle + c.eps * torch.eye(c.D, device=eval_device)

                means_mle_full = current_global_means_dev.clone()
                valid_indices = [idx for idx, m in enumerate(means_mle) if m is not None and counts[idx] >= c.min_samples]
                if valid_indices:
                     means_mle_full[valid_indices] = torch.stack([means_mle[i] for i in valid_indices])

                means_beta_dev = c.means_beta.to(eval_device)
                cov_beta_dev = c.cov_beta.to(eval_device)
                adaptive_means_eval = means_beta_dev.unsqueeze(1) * means_mle_full + (1 - means_beta_dev.unsqueeze(1)) * current_global_means_dev
                adaptive_covariance_eval = cov_beta_dev * local_cov_psd_dev + (1 - cov_beta_dev) * global_cov_eval_psd_dev # Use PSD global cov here

                # Collect client's beta value and corruption info
                client_info = {
                    'means_beta': means_beta_dev.mean().item(),
                    'cov_beta': cov_beta_dev.item(),
                    'is_corrupted': hasattr(c, 'is_corrupted_client') and c.is_corrupted_client,
                    'corruption_type': getattr(c, 'augmentation_type', None),
                    'severity': getattr(c, 'severity_level', None)
                }

                # 5. Set LDA weights
                c.set_lda_weights(adaptive_means_eval, adaptive_covariance_eval, c.priors)

                # 6. Evaluate
                with torch.no_grad():
                    acc, loss = c.evaluate() # Uses test set loader
                    accs.append(acc.cpu())
                    weighted_loss += (c.num_test / total_samples) * loss.cpu()
                    weighted_acc += (c.num_test / total_samples) * acc.cpu()
                    
                    # Store results by corruption category
                    if hasattr(c, 'is_corrupted_client') and c.is_corrupted_client and hasattr(c, 'augmentation_type'):
                        corruption_type = c.augmentation_type
                        if corruption_type in ['gaussian_noise', 'shot_noise', 'impulse_noise']:
                            noise_accs.append(acc.cpu())
                        elif corruption_type in ['defocus_blur', 'motion_blur']:
                            blur_accs.append(acc.cpu())
                        else:
                            other_accs.append(acc.cpu())
                    elif not (hasattr(c, 'is_corrupted_client') and c.is_corrupted_client):
                        clean_accs.append(acc.cpu())
                    
                    # Save metrics for this client
                    client_info['accuracy'] = acc.cpu().item()
                    client_info['loss'] = loss.cpu().item()
                    client_metrics[c.client_idx] = client_info

                    # KL Divergence
                    kl_div_per_class = []
                    for k in range(c.num_classes):
                         if counts[k] > c.min_samples:
                             try:
                                 # Use stats already confirmed/made PSD
                                 local_dist = MultivariateNormal(means_mle_full[k], local_cov_psd_dev)
                                 global_dist = MultivariateNormal(current_global_means_dev[k], global_cov_eval_psd_dev)
                                 kl_div_per_class.append(torch.distributions.kl.kl_divergence(local_dist, global_dist))
                             except ValueError as mvn_error:
                                 # self.logger.debug(f"Skipping KL for client {c.client_idx}, class {k} due to MVTNormal error: {mvn_error}")
                                 pass # Skip class if MVTNormal fails (e.g. cov not PSD despite efforts)
                             except Exception as kl_error:
                                 # self.logger.warning(f"KL calculation error for client {c.client_idx}, class {k}: {kl_error}")
                                 pass
                    if kl_div_per_class:
                        mean_kl_tensor = torch.mean(torch.stack(kl_div_per_class)).cpu()
                        kl_divs.append(mean_kl_tensor)
                        client_info['kl_div'] = mean_kl_tensor.item()

            except Exception as eval_client_e:
                self.logger.exception(f"Error evaluating client {c.client_idx}: {eval_client_e}")
            finally:
                # Ensure client model is moved back to CPU after evaluation
                c.model.to('cpu')
                c.global_means = c.global_means.cpu() # Restore CPU copy
                c.global_covariance = c.global_covariance.cpu()


        # Move global model back to CPU after evaluation round finishes
        self.model.cpu()

        final_acc = weighted_acc.item()
        final_loss = weighted_loss.item()
        final_std = torch.std(torch.stack(accs)).item() if accs else 0.0
        avg_kl = torch.stack(kl_divs).mean().item() if kl_divs else 0.0

        # Calculate category-specific metrics
        noise_avg = torch.mean(torch.stack(noise_accs)).item() if noise_accs else 0.0
        blur_avg = torch.mean(torch.stack(blur_accs)).item() if blur_accs else 0.0 
        other_avg = torch.mean(torch.stack(other_accs)).item() if other_accs else 0.0
        clean_avg = torch.mean(torch.stack(clean_accs)).item() if clean_accs else 0.0
        
        self.logger.info(f"Personalized Eval Completed. KL Div: {avg_kl:.4f}")
        self.logger.info(f"Corruption Category Results: Noise: {noise_avg:.2f}, Blur: {blur_avg:.2f}, Other: {other_avg:.2f}, Clean: {clean_avg:.2f}")
        
        # Find clients with extreme beta values that might need attention
        extreme_beta_clients = [(cid, info) for cid, info in client_metrics.items() 
                                if (info['is_corrupted'] and info['means_beta'] > 0.8) or 
                                   (not info['is_corrupted'] and info['means_beta'] < 0.3)]
        if extreme_beta_clients:
            self.logger.info(f"Clients with potentially suboptimal beta values: {len(extreme_beta_clients)}")
            for cid, info in extreme_beta_clients[:5]:  # Log just a few examples
                self.logger.debug(f"Client {cid}: beta={info['means_beta']:.2f}, corrupted={info['is_corrupted']}, "
                               f"type={info['corruption_type']}, acc={info['accuracy']:.2f}")

        return final_acc, final_loss, final_std
        
    def solve_beta_with_early_stopping(self, feats, labels, seed=0, patience=3):
        """
        Enhanced beta solving with early stopping to prevent overfitting
        """
        if self.local_beta:
            self.means_beta = torch.ones(1).to(self.device)
            self.cov_beta = torch.ones(1).to(self.device)
            return

        feats = feats.to(self.device)

        vals, counts = np.unique(labels, return_counts=True)
        valid_classes = [v for v, c in zip(vals, counts) if c >= self.num_cv_folds]

        if len(valid_classes) < 2:
             self.logger.warning(f"Client {self.client_idx}: Not enough classes ({len(valid_classes)}) with >= {self.num_cv_folds} samples for beta CV. Using beta=1 (local stats).")
             self.means_beta = torch.ones(1).cpu()
             self.cov_beta = torch.ones(1).cpu()
             return

        mask = np.isin(labels, valid_classes)
        pruned_feats = feats[mask]
        pruned_labels = labels[mask]

        try:
            skf = StratifiedKFold(n_splits=self.num_cv_folds, random_state=seed, shuffle=True)
            
            # For early stopping
            best_loss = float('inf')
            best_beta = None
            patience_counter = 0
            
            # Simple grid search with early stopping
            beta_values = torch.linspace(0.1, 0.9, 9)  # Test 9 values from 0.1 to 0.9
            
            for beta in beta_values:
                total_fold_loss = 0
                
                # CPU arrays for skf split
                pruned_feats_np = pruned_feats.cpu().numpy()
                
                for i, (train_idx, test_idx) in enumerate(skf.split(pruned_feats_np, pruned_labels)):
                    feats_tr = pruned_feats[train_idx]
                    labels_tr = pruned_labels[train_idx]
                    feats_te = pruned_feats[test_idx]
                    labels_te = pruned_labels[test_idx]
                    
                    means_tr, scatter_tr, _, counts_tr = self.compute_mle_statistics(feats=feats_tr, labels=labels_tr)
                    
                    means_tr_full = self.global_means.to(self.device).clone()
                    valid_indices_tr = [idx for idx, m in enumerate(means_tr) if m is not None]
                    if valid_indices_tr:
                        means_tr_full[valid_indices_tr] = torch.stack([means_tr[i] for i in valid_indices_tr])
                    
                    cov_denom_tr = max(1.0, np.sum(counts_tr) - len(np.unique(labels_tr)))
                    cov_tr = (scatter_tr / cov_denom_tr) + self.eps * torch.eye(self.D, device=self.device)
                    
                    fold_loss = self.beta_classifier(beta.reshape(1), means_tr_full, cov_tr, feats_te, labels_te)
                    if torch.isinf(fold_loss):
                        self.logger.warning(f"Client {self.client_idx}: Infinite loss for beta={beta.item():.1f}")
                        total_fold_loss = torch.tensor(1e9, device=self.device)
                        break
                    
                    total_fold_loss += fold_loss
                
                # Average loss across folds
                avg_loss = total_fold_loss / self.num_cv_folds
                
                # Check if this is the best beta so far
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_beta = beta
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= patience:
                    self.logger.debug(f"Client {self.client_idx}: Early stopping beta search at beta={beta.item():.1f} (best={best_beta.item():.1f})")
                    break
            
            # Use best beta found
            if best_beta is not None:
                optimal_beta = best_beta.detach().cpu()
                
                if self.single_beta:
                    self.means_beta = torch.ones_like(self.means_beta) * optimal_beta
                    self.cov_beta = optimal_beta
                else:
                    self.means_beta = torch.ones_like(self.means_beta) * optimal_beta
                    self.cov_beta = optimal_beta
            else:
                # Fallback
                self.means_beta = torch.ones(1).cpu() * 0.5
                self.cov_beta = torch.tensor(0.5)
                
        except Exception as e:
            self.logger.exception(f"Client {self.client_idx}: Error in beta search with early stopping: {e}")
            # Fallback to safe mid-range value
            self.means_beta = torch.ones(1).cpu() * 0.5
            self.cov_beta = torch.tensor(0.5)