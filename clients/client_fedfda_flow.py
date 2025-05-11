import torch
import numpy as np
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.correlation_tools import cov_nearest
import torchmin
from clients.client_base import Client
import torchvision.transforms as t
import logging # Added
import math
from data.utils.loader import get_augmented_partition, get_partition, get_digit5, get_office10

# Assume logger is passed during initialization or obtained globally
logger = logging.getLogger("FedFDA_Flow") # Alternative if not passed

class ClientFedFDA(Client):
    # Modified __init__ to accept a logger instance
    def __init__(self, args, client_idx, logger):
        super().__init__(args, client_idx)
        self.logger = logger # Store logger instance
        self.eps = 1e-4
        self.means_beta = torch.ones(size=(self.num_classes,)) * 0.5
        self.cov_beta = torch.Tensor([0.5])
        # local statistics
        self.means = torch.Tensor(torch.rand([self.num_classes, self.D]))
        self.covariance = torch.Tensor(torch.eye(self.D))

        try:
            counts, priors = self.get_label_distribution("train")
            self.priors = priors.cpu()
            self.priors = self.priors + self.eps
            self.priors = self.priors / self.priors.sum()
            self.class_counts = counts.cpu()
        except Exception as e:
            self.logger.error(f"Client {self.client_idx}: Failed to get label distribution: {e}")
            # Initialize with uniform priors if loading fails
            self.priors = torch.ones(self.num_classes) / self.num_classes
            self.class_counts = torch.zeros(self.num_classes, dtype=torch.long)


        # global statistics
        self.global_means = deepcopy(self.means)
        self.global_covariance = deepcopy(self.covariance)
        # interpolated statistics
        self.adaptive_means = deepcopy(self.means)
        self.adaptive_covariance = deepcopy(self.covariance)
        # interpolation term solver
        self.single_beta = args.single_beta
        self.local_beta = args.local_beta
        self.num_cv_folds = 2
        self.min_samples = self.num_cv_folds
        if self.local_beta:
            self.single_beta = True

        # --- Corruption Information ---
        self.is_corrupted_client = self.client_idx < 50 # Assumes 100 clients total
        self.augmentation_type = None  # Will be set during dataset setup
        self.severity_level = None     # Will be set during dataset setup
        
        # --- Adaptive Beta Initialization ---
        # Initialize beta differently based on corruption type if known
        if hasattr(self, 'augmentation') and self.is_corrupted_client:
            # Noise-based corruptions benefit from lower beta (more global influence)
            if self.augmentation in ['gaussian_noise', 'shot_noise', 'impulse_noise']:
                self.means_beta = torch.ones(size=(self.num_classes,)) * 0.3
                self.cov_beta = torch.Tensor([0.3])
            # Blur-based corruptions can use moderate beta
            elif self.augmentation in ['defocus_blur', 'motion_blur']:
                self.means_beta = torch.ones(size=(self.num_classes,)) * 0.4
                self.cov_beta = torch.Tensor([0.4])
            # Weather/condition-based corruptions need higher beta for more local adaptation
            elif self.augmentation in ['frost', 'fog', 'brightness', 'contrast', 'jpeg_compression']:
                self.means_beta = torch.ones(size=(self.num_classes,)) * 0.6
                self.cov_beta = torch.Tensor([0.6])

        # --- Enhanced Augmentation for Clean Clients ---
        # Define different augmentation methods for clean clients
        self.clean_client_augmentations = [
            # Horizontal flip is already implemented in train()
            # Color jitter
            lambda x: x + 0.05 * torch.randn_like(x) if torch.rand(1) < 0.3 else x,
            # Random crop-like effect (center crop with padding)
            lambda x: self._random_resized_crop(x) if torch.rand(1) < 0.3 else x,
            # Random rotation (via affine grid)
            lambda x: self._random_rotation(x, max_degrees=10) if torch.rand(1) < 0.3 else x,
        ]
        
    def _random_resized_crop(self, x):
        """Simple implementation of random resized crop for tensor data"""
        batch_size, channels, height, width = x.shape
        scale = 0.8 + 0.2 * torch.rand(1)  # Scale between 0.8 and 1.0
        
        # Apply scale by taking center crop and resizing back (simulated crop)
        crop_size = int(scale.item() * min(height, width))
        start_h = (height - crop_size) // 2
        start_w = (width - crop_size) // 2
        
        cropped = x[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
        try:
            # Use interpolate to resize back to original (works in-place with tensors)
            return torch.nn.functional.interpolate(cropped, size=(height, width), mode='bilinear', align_corners=False)
        except Exception:
            # Fallback if interpolation fails
            return x
            
    def _random_rotation(self, x, max_degrees=10):
        """Apply random rotation to batch of images using affine grid"""
        batch_size, channels, height, width = x.shape
        
        angle = torch.rand(1) * 2 * max_degrees - max_degrees  # Between -max_degrees and max_degrees
        radian = angle * math.pi / 180
        
        # Create rotation matrix 
        cos_val = torch.cos(radian)
        sin_val = torch.sin(radian)
        
        try:
            # Create affine transformation matrix
            affine_matrix = torch.tensor([
                [cos_val, -sin_val, 0],
                [sin_val, cos_val, 0]
            ], device=x.device).float()
            
            # Repeat for batch
            affine_matrix = affine_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # Create sampling grid
            grid = torch.nn.functional.affine_grid(
                affine_matrix, x.size(), align_corners=False
            )
            
            # Apply grid sample
            return torch.nn.functional.grid_sample(
                x, grid, mode='bilinear', align_corners=False
            )
        except Exception:
            # Return original if rotation fails
            return x

    def train(self):
        # Safely load data
        try:
             trainloader = self.load_train_data()
        except Exception as e:
             self.logger.error(f"Client {self.client_idx}: Failed to load training data: {e}")
             return 0.0, float('inf') # Return dummy values indicating failure

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        self.model = self.model.to(self.device)
        self.means = self.means.to(self.device)
        self.covariance = self.covariance.to(self.device)
        self.adaptive_means = self.adaptive_means.to(self.device)
        self.adaptive_covariance = self.adaptive_covariance.to(self.device)
        self.global_means = self.global_means.to(self.device)
        self.global_covariance = self.global_covariance.to(self.device)
        self.model.train()

        # Store the augmentation type and severity level (if available from dataset setup)
        if hasattr(self, 'augmentation') and hasattr(self, 'severity'):
            self.augmentation_type = self.augmentation
            self.severity_level = self.severity
            # Log the client's corruption configuration
            self.logger.debug(f"Client {self.client_idx}: Augmentation={self.augmentation_type}, Severity={self.severity_level}")

        self.set_lda_weights(self.global_means, self.global_covariance, self.priors)

        # Freeze LDA layer
        for param in self.model.fc.parameters():
            param.requires_grad_(False)

        # Meter alternative (using simple lists for aggregation)
        epoch_losses = []
        epoch_accs = []
        total_samples = 0

        feats_ep = []
        labels_ep = []
        for e in range(1, self.local_epochs+1):
            for i, (x, y) in enumerate(trainloader):
                # Data loading check
                if x is None or y is None:
                    self.logger.warning(f"Client {self.client_idx}: Received None data batch, skipping.")
                    continue
                    
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size = x.size(0)
                total_samples += batch_size

                # --- Apply Advanced Augmentation to Clean Clients ---
                if not self.is_corrupted_client:
                    # Apply horizontal flip (common augmentation)
                    if torch.rand(1) < 0.5:
                        x = torch.flip(x, dims=[3])
                        
                    # Apply additional augmentations with some probability
                    for aug_fn in self.clean_client_augmentations:
                        x = aug_fn(x)

                # forward pass
                try:
                     feats, output = self.model(x, return_feat=True)
                     loss = self.loss(output, y)
                except Exception as model_e:
                     self.logger.error(f"Client {self.client_idx}: Error during model forward/loss calculation: {model_e}")
                     # Skip batch or handle error appropriately
                     continue

                # Calculate metrics for logging (using frozen classifier)
                with torch.no_grad():
                     temp_acc = (output.argmax(1) == y).float().mean() * 100.0
                     epoch_accs.append(temp_acc.item() * batch_size) # Store weighted acc
                     epoch_losses.append(loss.item() * batch_size)   # Store weighted loss

                # accumulate features and labels
                feats_ep.append(feats.detach())
                labels_ep.append(y.cpu().numpy())

                # backward pass
                try:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except Exception as backward_e:
                    self.logger.error(f"Client {self.client_idx}: Error during backward pass/optimizer step: {backward_e}")
                    # May need to skip rest of epoch or handle more gracefully
                    continue
                    
        # --- Post-Training Processing ---
        if not feats_ep:
             self.logger.warning(f"Client {self.client_idx}: No features collected during training.")
             # Handle case where training produced no usable features
             # Move tensors back to CPU before returning
             self.model = self.model.to("cpu")
             # ... (move other tensors to cpu) ...
             return 0.0, float('inf')

        # Solve beta and update statistics
        try:
            feats_ep = torch.cat(feats_ep, dim=0)
            labels_ep = np.concatenate(labels_ep, axis=0)
            
            # --- Apply Adaptive Beta Constraint Based on Corruption ---
            # Store original beta values
            orig_means_beta = self.means_beta.clone() if hasattr(self, 'means_beta') else None
            orig_cov_beta = self.cov_beta.clone() if hasattr(self, 'cov_beta') else None
                
            # Solve for beta
            self.solve_beta_with_early_stopping(feats_ep, labels_ep)
            
            # Apply constraints for corrupted clients if needed
            if self.is_corrupted_client and self.augmentation_type:
                # Get corruption category
                noise_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise']
                blur_corruptions = ['defocus_blur', 'motion_blur']
                other_corruptions = ['frost', 'fog', 'brightness', 'contrast', 'jpeg_compression']
                
                # Apply min/max constraints based on corruption type
                if self.augmentation_type in noise_corruptions:
                    # Noise benefits from more global influence at high severity
                    max_beta = max(0.7, 1.0 - (0.1 * self.severity_level))
                    self.means_beta = torch.clamp(self.means_beta, max=max_beta)
                    self.cov_beta = torch.clamp(self.cov_beta, max=max_beta)
                elif self.augmentation_type in blur_corruptions:
                    # Blur needs balanced approach
                    min_beta = 0.3
                    max_beta = 0.8
                    self.means_beta = torch.clamp(self.means_beta, min=min_beta, max=max_beta)
                    self.cov_beta = torch.clamp(self.cov_beta, min=min_beta, max=max_beta)
                else:
                    # Other corruptions need more local adaptation
                    min_beta = 0.4
                    self.means_beta = torch.clamp(self.means_beta, min=min_beta)
                    self.cov_beta = torch.clamp(self.cov_beta, min=min_beta)
                
                # Log if beta was constrained
                if orig_means_beta is not None and not torch.allclose(orig_means_beta, self.means_beta):
                    self.logger.debug(f"Client {self.client_idx}: Beta constrained from {orig_means_beta.mean().item():.2f} to {self.means_beta.mean().item():.2f} based on corruption type")

            means_mle, scatter_mle, _, counts = self.compute_mle_statistics(feats=feats_ep, labels=labels_ep)
            
            # Ensure covariance is PSD
            cov_denom = max(1.0, np.sum(counts) - self.num_classes) # Avoid division by zero or negative
            cov_mle = (scatter_mle / cov_denom) + self.eps * torch.eye(self.D, device=self.device)
            try:
                 cov_psd_np = cov_nearest(cov_mle.cpu().numpy(), method="clipped", threshold=1e-4)
                 cov_psd = torch.Tensor(cov_psd_np).to(self.device)
            except Exception as cov_e:
                 self.logger.warning(f"Client {self.client_idx}: Covariance nearest PSD failed: {cov_e}. Using regularized MLE cov.")
                 cov_psd = cov_mle # Fallback

            means_mle_full = self.global_means.to(self.device).clone()
            valid_indices = [idx for idx, m in enumerate(means_mle) if m is not None]
            if valid_indices:
                 means_mle_full[valid_indices] = torch.stack([means_mle[i] for i in valid_indices])

            self.update(means_mle_full, cov_psd)
        except Exception as post_train_e:
            self.logger.error(f"Client {self.client_idx}: Error during post-training processing (beta/stats): {post_train_e}")
            # Return previous metrics if possible, or indicate failure
            avg_acc = sum(epoch_accs) / total_samples if total_samples > 0 else 0.0
            avg_loss = sum(epoch_losses) / total_samples if total_samples > 0 else float('inf')
            # Move tensors back to CPU
            self.model = self.model.to("cpu")
            # ... (move other tensors to cpu) ...
            return avg_acc, avg_loss

        # Move tensors back to CPU
        self.model = self.model.to("cpu")
        self.means = self.means.cpu()
        self.covariance = self.covariance.cpu()
        self.adaptive_means = self.adaptive_means.cpu()
        self.adaptive_covariance = self.adaptive_covariance.cpu()
        self.global_means = self.global_means.cpu()
        self.global_covariance = self.global_covariance.cpu()

        # Calculate final average accuracy and loss for the training phase
        avg_acc = sum(epoch_accs) / total_samples if total_samples > 0 else 0.0
        avg_loss = sum(epoch_losses) / total_samples if total_samples > 0 else float('inf')

        return avg_acc, avg_loss

    def beta_classifier(self, beta, means_local, cov_local, feats, labels):
        # Ensure tensors are on the correct device
        beta = beta.to(self.device)
        means_local = means_local.to(self.device)
        cov_local = cov_local.to(self.device)
        feats = feats.to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        priors = self.priors.to(self.device)
        global_means_dev = self.global_means.to(self.device)
        global_covariance_dev = self.global_covariance.to(self.device)

        if self.single_beta:
            beta_val = beta.clip(0,1)
            means = beta_val * means_local + (1-beta_val) * global_means_dev
            cov = beta_val * cov_local + (1-beta_val) * global_covariance_dev
        else:
            beta_mean = beta[0].clip(0,1)
            beta_cov = beta[-1].clip(0,1)
            means = beta_mean * means_local + (1-beta_mean) * global_means_dev
            cov = beta_cov * cov_local + (1-beta_cov) * global_covariance_dev

        cov = (1-self.eps)*cov + self.eps * torch.eye(self.D, device=self.device)

        try:
             y_pred_logits = self.lda_classify(feats, means=means, covariance=cov, priors=priors, use_lstsq=True)
             loss = torch.nn.functional.cross_entropy(y_pred_logits, labels)
        except Exception as e:
             self.logger.error(f"Client {self.client_idx}: Error in beta_classifier LDA/Loss step: {e}")
             loss = torch.tensor(float('inf')) # Return infinite loss on error
        
        return loss

    def solve_beta(self, feats, labels, seed=0):
        """Original beta solving method - kept for compatibility"""
        if self.local_beta:
            self.means_beta = torch.ones(1).cpu()
            self.cov_beta = torch.ones(1).cpu()
            return

        feats = feats.to(self.device) # Keep feats on GPU

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

            def loss_fn(beta_param):
                total_fold_loss = 0
                # Use CPU numpy arrays for skf.split
                pruned_feats_np = pruned_feats.cpu().numpy()
                
                for i, (train_idx, test_idx) in enumerate(skf.split(pruned_feats_np, pruned_labels)):
                    # Get fold data (keep on GPU)
                    feats_tr, labels_tr = pruned_feats[train_idx], pruned_labels[train_idx]
                    feats_te, labels_te = pruned_feats[test_idx], pruned_labels[test_idx]

                    means_tr, scatter_tr, _, counts_tr = self.compute_mle_statistics(feats=feats_tr, labels=labels_tr)

                    means_tr_full = self.global_means.to(self.device).clone()
                    valid_indices_tr = [idx for idx, m in enumerate(means_tr) if m is not None]
                    if valid_indices_tr:
                         means_tr_full[valid_indices_tr] = torch.stack([means_tr[i] for i in valid_indices_tr])
                    
                    cov_denom_tr = max(1.0, np.sum(counts_tr) - len(np.unique(labels_tr)))
                    if cov_denom_tr > 0 :
                         cov_tr = (scatter_tr / cov_denom_tr) + self.eps * torch.eye(self.D, device=self.device)
                         try:
                             cov_psd_tr = torch.Tensor(cov_nearest(cov_tr.cpu().numpy(), method="clipped", threshold=1e-4)).to(self.device)
                         except Exception:
                             cov_psd_tr = cov_tr
                    else:
                         cov_psd_tr = self.global_covariance.to(self.device) # Use global if cannot estimate

                    fold_loss = self.beta_classifier(beta_param, means_tr_full, cov_psd_tr, feats_te, labels_te)
                    # Handle potential inf loss from beta_classifier
                    if torch.isinf(fold_loss):
                         self.logger.warning(f"Client {self.client_idx}: Infinite loss encountered in beta CV fold {i}. Returning large finite value.")
                         return torch.tensor(1e9, device=self.device) # Return large value instead of inf
                    total_fold_loss += fold_loss
                
                return total_fold_loss / self.num_cv_folds

            # Optimization
            if self.single_beta:
                x0 = 0.5 * torch.ones(size=(1,), device=self.device, requires_grad=True)
            else:
                x0 = 0.5 * torch.ones(size=(2,), device=self.device, requires_grad=True)

            result = torchmin.minimize(loss_fn, x0=x0, method='l-bfgs', max_iter=20, options={"gtol": 1e-4})

            # Check optimization result status if available in torchmin v0.0.5+
            # if not result.success:
            #    self.logger.warning(f"Client {self.client_idx}: Beta optimization did not converge. Message: {result.message}")

            optimal_beta = result.x.detach().cpu().clip(0, 1)

            if self.single_beta:
                self.means_beta = torch.ones_like(self.means_beta) * optimal_beta[0]
                self.cov_beta = optimal_beta[0]
            else:
                self.means_beta = torch.ones_like(self.means_beta) * optimal_beta[0]
                self.cov_beta = optimal_beta[1]

        except Exception as e:
            self.logger.exception(f"Client {self.client_idx}: Unhandled Error during beta optimization: {e}. Using beta=1.")
            self.means_beta = torch.ones(1).cpu()
            self.cov_beta = torch.ones(1).cpu()
            
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

    # --- update, compute_mle_statistics, lda_classify, set_lda_weights remain largely the same ---
    # --- but ensure device consistency and add specific warnings/errors ---

    def update(self, means_mle, cov_mle):
        means_mle_dev = means_mle.to(self.device)
        cov_mle_dev = cov_mle.to(self.device)
        means_beta_dev = self.means_beta.to(self.device)
        cov_beta_dev = self.cov_beta.to(self.device)
        global_means_dev = self.global_means.to(self.device)
        global_covariance_dev = self.global_covariance.to(self.device)

        self.means.data = means_mle_dev.data
        self.covariance.data = cov_mle_dev.data

        self.adaptive_means.data = means_beta_dev.unsqueeze(1) * means_mle_dev.data + (1-means_beta_dev.unsqueeze(1)) * global_means_dev.data
        self.adaptive_covariance.data = cov_beta_dev * cov_mle_dev.data + (1-cov_beta_dev) * global_covariance_dev.data

        self.means = self.means.cpu()
        self.covariance = self.covariance.cpu()
        self.adaptive_means = self.adaptive_means.cpu()
        self.adaptive_covariance = self.adaptive_covariance.cpu()

    def compute_mle_statistics(self, split="train", feats=None, labels=None):
        if feats is None or labels is None:
             self.model = self.model.to(self.device)
             try:
                 feats, labels = self.compute_feats(split=split) # Assuming returns features on device
             except Exception as e:
                 self.logger.error(f"Client {self.client_idx}: Failed computing features for split '{split}': {e}")
                 return [None]*self.num_classes, torch.zeros((self.D, self.D), device=self.device), torch.zeros(self.num_classes, device=self.device), np.zeros(self.num_classes, dtype=int)
             finally:
                self.model = self.model.to("cpu") # Ensure model moved back
        else:
             feats = feats.to(self.device)

        means = [None] * self.num_classes
        scatter = torch.zeros((self.D, self.D), device=self.device)

        if len(labels) > 0:
             unique_labels, counts = np.unique(labels, return_counts=True)
             bincounts = np.bincount(labels, minlength=self.num_classes)
             priors = torch.Tensor(bincounts / float(len(labels))).float().to(self.device)
             class_counts = torch.Tensor(bincounts).long().cpu()

             for i, y in enumerate(unique_labels):
                 class_mask = (labels == y)
                 y_count = counts[i]
                 if y_count > 0:
                     class_feats = feats[class_mask]
                     means[y] = torch.mean(class_feats, dim=0)
                     feats_centered = class_feats - means[y]
                     scatter += torch.mm(feats_centered.t(), feats_centered)
        else: # Handle empty labels case
             priors = torch.zeros(self.num_classes, device=self.device)
             class_counts = torch.zeros(self.num_classes, dtype=torch.long).cpu()

        return means, scatter, priors, class_counts.numpy()

    def lda_classify(self, Z, means=None, covariance=None, priors=None, use_lstsq=True):
        Z = Z.to(self.device)
        means = means.to(self.device)
        covariance = covariance.to(self.device)
        if priors is None:
            priors = self.priors.to(self.device)
        else:
            priors = priors.to(self.device)

        cov_reg = (1-self.eps)*covariance + self.eps * torch.eye(self.D, device=self.device)

        try:
            if use_lstsq:
                coefs = torch.linalg.lstsq(cov_reg, means.T)[0].T
            else:
                cov_inv = torch.linalg.inv(cov_reg)
                coefs = torch.matmul(means, cov_inv)
        except torch.linalg.LinAlgError:
             self.logger.warning(f"Client {self.client_idx}: Covariance matrix singular during classify. Using pseudo-inverse.")
             cov_pinv = torch.linalg.pinv(cov_reg)
             coefs = torch.matmul(means, cov_pinv)
        except Exception as e:
             self.logger.error(f"Client {self.client_idx}: Unhandled error during lda_classify coefficient calculation: {e}")
             # Return zeros or handle otherwise? Maybe raise?
             raise e # Re-raise the exception

        safe_priors = torch.clamp(priors, min=1e-9)
        intercepts = -0.5 * torch.diag(torch.matmul(means, coefs.T)) + torch.log(safe_priors)
        return Z @ coefs.T + intercepts

    def set_lda_weights(self, means=None, covariance=None, priors=None, use_lstsq=True):
        if means is None: means = self.means
        if covariance is None: covariance = self.covariance
        if priors is None: priors = self.priors

        with torch.no_grad():
            means_dev = means.to(self.device)
            covariance_dev = covariance.to(self.device)
            priors_dev = priors.to(self.device)
            cov_reg = (1-self.eps)*covariance_dev + self.eps * torch.eye(self.D, device=self.device)

            try:
                 if use_lstsq:
                     coefs = torch.linalg.lstsq(cov_reg, means_dev.T)[0].T
                 else:
                     cov_inv = torch.linalg.inv(cov_reg)
                     coefs = torch.matmul(means_dev, cov_inv)
            except torch.linalg.LinAlgError:
                  self.logger.warning(f"Client {self.client_idx}: Covariance matrix singular during weight setting. Using pseudo-inverse.")
                  cov_pinv = torch.linalg.pinv(cov_reg)
                  coefs = torch.matmul(means_dev, cov_pinv)
            except Exception as e:
                 self.logger.error(f"Client {self.client_idx}: Unhandled error during set_lda_weights coefficient calculation: {e}")
                 raise e

            safe_priors = torch.clamp(priors_dev, min=1e-9)
            intercepts = -0.5 * torch.diag(torch.matmul(means_dev, coefs.T)) + torch.log(safe_priors)

            self.model.fc = self.model.fc.to(self.device)
            self.model.fc.weight.data = coefs.detach()
            self.model.fc.bias.data = intercepts.detach()

    def setup_augmented_dataset(self, args, augmentation=None, severity=None):
        self.logger = logger # Store logger instance
        indices = np.load(f"data/partition/{self.partition_path}/client_{self.client_idx}.npz")
        
        augmentations = [
            "motion_blur",
            "defocus_blur",
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "frost",
            "fog",
            "jpeg_compression",
            "brightness",
            "contrast"
        ]
        
        # by default, select one augmentation and severity level
        # assumes 100 total clients and 50 clients for augmentation
        if augmentation is None:
            self.augmentation = augmentations[self.client_idx % 10]
            self.severity = (self.client_idx % 5)+1
            # Now apply the adaptive beta initialization based on corruption type
            self.is_corrupted_client = self.client_idx < 50
            if self.is_corrupted_client:
                if self.augmentation in ['gaussian_noise', 'shot_noise', 'impulse_noise']:
                    self.means_beta = torch.ones(size=(self.num_classes,)) * 0.3
                    self.cov_beta = torch.Tensor([0.3])
                elif self.augmentation in ['defocus_blur', 'motion_blur']:
                    self.means_beta = torch.ones(size=(self.num_classes,)) * 0.4
                    self.cov_beta = torch.Tensor([0.4])
                elif self.augmentation in ['frost', 'fog', 'brightness', 'contrast', 'jpeg_compression']:
                    self.means_beta = torch.ones(size=(self.num_classes,)) * 0.6
                    self.cov_beta = torch.Tensor([0.6])
                self.logger.debug(f"Client {self.client_idx}: Initialized beta={self.means_beta[0]} for corruption {self.augmentation} (severity {self.severity})")
        # otherwise, use the given augmentation/severity pair
        else:
            self.augmentation = augmentation
            self.severity = severity
            
        train_dataset, test_dataset = get_augmented_partition(args, indices, self.augmentation, self.severity)
        self.trainset, self.valset = self.partial_dataset(train_dataset)
        self.testset = test_dataset
        self.num_train = len(self.trainset)
        self.num_val = len(self.valset)
        self.num_test = len(self.testset)