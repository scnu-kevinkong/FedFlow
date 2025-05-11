import time
from copy import deepcopy
import torch
from servers.server_base import Server
from clients.client_fedavg import ClientFedAvg
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, TensorDataset
from utils.util import AverageMeter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import numpy as np
from collections import deque
import traceback
import random

class FlowMatchingModel(nn.Module):
    # Using dropout_rate=0.15 as seen in your log file initialization message
    def __init__(self, param_dim, hidden_dim=2048, rank=256, dropout_rate=0.15):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256))
        self.input_norm = nn.LayerNorm(param_dim)
        self.low_rank_proj = nn.Linear(param_dim, rank)
        self.main_net = nn.Sequential(
            nn.Linear(rank + 256, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, rank)
        )
        self.inv_proj = nn.Linear(rank, param_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Small gain for stability with GELU/SiLU might be helpful
            nn.init.xavier_normal_(m.weight, gain=0.02) # Slightly increased gain
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def compute_flow_loss(self, real_params, noise_params, t, delta=1.0):
        target_flow = real_params - noise_params
        # Clamp input to prevent extreme values before forward pass
        noise_params = torch.clamp(noise_params, -15.0, 15.0)
        pred_flow = self(noise_params, t)
        diff = pred_flow - target_flow
        # Huber loss
        loss = torch.where(torch.abs(diff) < delta, 0.5 * diff ** 2, delta * (torch.abs(diff) - 0.5 * delta))
        return torch.mean(loss)

    def forward(self, z, t):
        if z.dim() == 1: z = z.unsqueeze(0)
        # Ensure t has the correct dimensions for broadcasting
        if t.dim() == 0: t = t.view(1, 1).expand(z.size(0), 1)
        elif t.dim() == 1 and t.size(0) == z.size(0) and t.size(-1) != 1 : t = t.unsqueeze(-1)
        elif t.dim() == 1 and t.size(0) == 1: t = t.expand(z.size(0), 1) # Expand if single time value given
        elif t.dim() == 2 and t.size(0) == 1 and t.size(1) == 1: t = t.expand(z.size(0), 1) # Expand if single time value given
        elif t.dim() != 2 or t.size(0) != z.size(0) or t.size(1) != 1:
             raise ValueError(f"Unexpected shape for t: {t.shape}, expected ({z.size(0)}, 1)")


        z_norm = self.input_norm(z)
        z_proj = self.low_rank_proj(z_norm)
        t_emb = self.time_embed(t.float())
        x = torch.cat([z_proj, t_emb], dim=-1)
        output = self.inv_proj(self.main_net(x))
        return output

    @torch.no_grad()
    def generate(self, init_params, num_steps=100):
        z = init_params.clone()
        dt = 1.0 / num_steps
        self.eval()
        # Slightly relaxed clamp value during generation might help exploration
        clamp_val = 20.0

        # Pre-allocate time tensor on CPU
        t_cpu = torch.ones(z.size(0), 1, device='cpu')

        for step in range(num_steps):
            t_val = 1.0 - step / num_steps
            # Update CPU tensor and move to device only when needed
            t = t_cpu * t_val
            t_dev = t.to(z.device, non_blocking=True)
            z_dev = z # Already on device

            try:
                # Ensure t_dev has the correct shape before forward pass
                if t_dev.dim() != 2 or t_dev.size(1) != 1:
                     t_dev = t_dev.view(z_dev.size(0), 1)

                pred = self(z_dev, t_dev)

                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    print(f"Warning: NaN/Inf detected in FM prediction at step {step}. Clamping and continuing.")
                    # Attempt to recover by clamping the prediction itself
                    pred = torch.nan_to_num(pred, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
                    # If still problematic after clamp, might need to stop
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                         print(f"Stopping generation at step {step} due to persistent NaN/Inf.")
                         return z # Return current state

                z = z + pred * dt
                # Clamp results after update
                z = torch.clamp(z, -clamp_val, clamp_val)
                # Ensure no NaNs propagate (though clamping should handle inf)
                z = torch.nan_to_num(z, nan=0.0)


            except Exception as e:
                 print(f"Error during FM generation step {step}: {e}")
                 traceback.print_exc()
                 # Return current state instead of potentially corrupted z
                 return z

        # Final check
        if torch.isnan(z).any() or torch.isinf(z).any():
             print("Warning: Final generated parameters contain NaN/Inf. Clamping final result.")
             z = torch.nan_to_num(z, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
             # Consider returning init_params if final state is unreliable
             # return init_params

        return z


class ServerFedAvg(Server):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.augmented_scenario = getattr(args, 'augmented', False)
        self.use_robust_similarity = getattr(args, 'use_robust_similarity', True) # Keep True for robustness
        self.adaptive_cosine = getattr(args, 'adaptive_cosine', True) # Keep True for adaptivity

        # --- Tuned Hyperparameters ---
        print(f"Optimized Robust FlowMatch Server Initializing...")

        # 1. Relax Cosine Filtering
        self.cosine_base_threshold = getattr(args, 'cosine_base_threshold', -0.15) # Relaxed from -0.1
        self.cosine_stricter_offset = getattr(args, 'cosine_stricter_offset', 0.1) # Offset for potentially corrupted clients (remains 0.1)
        self.cosine_low_agreement_penalty = getattr(args, 'cosine_low_agreement_penalty', 0.0) # DISABLED penalty (was 0.1)
        self.low_agreement_threshold = getattr(args, 'low_agreement_threshold', 0.3) # Keep definition of low agreement

        # 2. Relax Norm Filtering
        self.norm_percentile_threshold = getattr(args, 'norm_percentile_threshold', 95) # Relaxed from 90

        # 3. Flow Model Trust & Reliability
        self.fm_trust_loss_threshold = getattr(self.args, 'fm_trust_loss_threshold', 0.25) # Keep threshold for trusting FM
        self.fm_reliability_threshold = getattr(args, 'fm_reliability_threshold', 0.92) # Increased from 0.9 to refine fewer clients

        # 4. Reduce FM Training Overhead
        self.fm_epochs = getattr(self.args, 'fm_epochs', 15) # Reduced from 30

        # Other FM parameters (mostly kept from original tuned version)
        self.fm_lr = getattr(self.args, 'fm_lr', 3e-5)
        self.fm_wd = getattr(self.args, 'fm_wd', 1e-4)
        # Dropout as seen in logs
        self.fm_dropout_rate = getattr(self.args, 'fm_dropout_rate', 0.15)
        self.fm_min_buffer = getattr(self.args, 'fm_min_buffer', 500)
        self.fm_batch_size = getattr(self.args, 'fm_batch_size', 128)
        self.fm_train_freq = getattr(self.args, 'fm_train_freq', 1)
        self.fm_grad_norm = getattr(self.args, 'fm_grad_norm', 1.0)
        self.fm_gen_steps = getattr(self.args, 'fm_gen_steps', 100)
        self.param_buffer = deque(maxlen=getattr(self.args, 'fm_buffer_size', 2000)) # Buffer size seems reasonable

        # Mixing parameters (kept from original tuned version)
        self.mix_alpha_min = getattr(args, 'mix_alpha_min', 0.8)
        self.mix_alpha_max = getattr(args, 'mix_alpha_max', 0.95)
        self.fm_loss_trust_scale = getattr(args, 'fm_loss_trust_scale', 0.05)

        # Reliability update parameter
        self.reliability_beta = getattr(args, 'reliability_beta', 0.9)
        # Rounds before cosine filter activates
        self.relax_cosine_rounds = getattr(args, 'relax_cosine_rounds', 5)


        print(f"Augmented (Corruption) Scenario: {self.augmented_scenario}")
        print(f"Using Robust Similarity Metric: {self.use_robust_similarity}")
        print(f"Using Adaptive Cosine Thresholds: {self.adaptive_cosine}")
        print(f"  - Base Cosine Threshold: {self.cosine_base_threshold}")
        print(f"  - Stricter Offset (Corrupted Clients): {self.cosine_stricter_offset}")
        print(f"  - Low Agreement Penalty: {self.cosine_low_agreement_penalty} (0 means disabled)")
        print(f"Norm Percentile Threshold: {self.norm_percentile_threshold}")
        print(f"FM Trust Loss Threshold: {self.fm_trust_loss_threshold}")
        print(f"FM Reliability Threshold (for Refinement): {self.fm_reliability_threshold}")
        print(f"FM Training Epochs: {self.fm_epochs}")


        self.clients = []
        for client_idx in range(self.num_clients):
            # Ensure ClientFedAvg can be instantiated correctly
            try:
                 c = ClientFedAvg(args, client_idx)
                 self.clients.append(c)
            except Exception as e:
                 print(f"Error creating Client {client_idx}: {e}")
                 print("Ensure ClientFedAvg class is defined and args are appropriate.")
                 raise e


        self.flow_model = None
        self.last_fm_loss = float('inf')
        self.client_reliability_scores = {}

        # Determine parameter dimension and initialize FM
        self.model.cpu() # Ensure model is on CPU for consistent flattening
        # Filter only float params requiring grad
        self.param_dim = sum(p.numel() for p in self.model.parameters() if p.requires_grad and p.is_floating_point())

        if self.param_dim > 0:
            print(f"Determined trainable float param_dim: {self.param_dim}")
            self.model.to(self.device) # Move back to target device
            self.init_flow_model()
        else:
            print("Warning: Model has no trainable float parameters. FlowMatch part disabled.")
            self.flow_model = None
            self.optimizer = None
            self.scheduler_epochs = None
            self.fm_lr_scheduler_rounds = None

        self.param_mean = None
        self.param_std = None
        self.r = 0
        self.previous_global_params_flat = None


    def init_flow_model(self):
        """Initializes the FlowMatchingModel, optimizer, and schedulers."""
        if self.param_dim == 0 or self.flow_model is not None: # Avoid re-initialization if already done
             if self.param_dim == 0: print("FM not initialized (param_dim is 0).")
             return

        print(f"Initializing FlowMatchingModel with param_dim={self.param_dim}, dropout={self.fm_dropout_rate}")
        self.flow_model = FlowMatchingModel(self.param_dim, dropout_rate=self.fm_dropout_rate).to(self.device)
        self.optimizer = torch.optim.AdamW(self.flow_model.parameters(), lr=self.fm_lr, weight_decay=self.fm_wd)

        # Calculate total FM training steps based on rounds and frequency
        total_fm_train_events = self.global_rounds // self.fm_train_freq
        if total_fm_train_events == 0: total_fm_train_events = 1 # Avoid division by zero

        # Linear decay across rounds for the base LR used in AdamW
        # End factor 0.1 means LR decays to 10% of initial fm_lr over the global rounds
        self.fm_lr_scheduler_rounds = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_fm_train_events)
        print(f"Initialized FM LR scheduler across rounds (Linear decay to 0.1 over ~{total_fm_train_events} FM training events).")

        # Epoch scheduler will be re-initialized before each FM training session
        self.scheduler_epochs = None


    def send_models(self):
        """Sends the current global model to active clients."""
        if not self.active_clients: return
        # Ensure model is on the correct device before deepcopying
        self.model.to(self.device)
        for c in self.active_clients:
            try:
                # Send a copy to the client's specific device
                c.set_model(deepcopy(self.model).to(c.device))
            except Exception as e:
                print(f"Error sending model to Client {c.client_idx}: {e}")
                traceback.print_exc()

    def _flatten_params(self, model):
        """Flattens trainable float parameters of a model."""
        # Move model to CPU temporarily for consistent parameter ordering
        original_device = next(model.parameters()).device
        model.cpu()
        # Explicitly filter for floating-point tensors requiring gradients
        trainable_params = [p.detach().clone().float() for p in model.parameters() if p.requires_grad and p.is_floating_point()]
        model.to(original_device) # Move back to original device

        if not trainable_params:
            # Return an empty tensor on CPU if no such parameters exist
            return torch.tensor([], dtype=torch.float32, device='cpu')
        # Concatenate flattened parameters
        flat_params = torch.cat([p.flatten() for p in trainable_params])
        return flat_params # Remains on CPU

    def _unflatten_params(self, flat_params_cpu, model):
        """Unflattens parameters back into the model structure."""
        if self.param_dim == 0:
            print("Warning: Cannot unflatten, param_dim is 0.")
            return
        if flat_params_cpu.numel() != self.param_dim:
             raise ValueError(f"Parameter dimension mismatch: flat_params_cpu has {flat_params_cpu.numel()} elements, expected {self.param_dim}")

        # Ensure model is on the correct device
        model.to(self.device)
        # Move flattened params to the model's device
        flat_params = flat_params_cpu.to(self.device)

        pointer = 0
        for param in model.parameters():
            # Only update parameters that were included in flattening
            if param.requires_grad and param.is_floating_point():
                numel = param.numel()
                if pointer + numel > flat_params.numel():
                    # This check should theoretically not fail if param_dim was calculated correctly
                    raise ValueError(f"Flattened params tensor too short. Pointer {pointer}, numel {numel}, total {flat_params.numel()}")
                try:
                    # Copy data, ensuring view matches parameter shape
                    param.data.copy_(flat_params[pointer : pointer + numel].view_as(param.data))
                except Exception as e:
                    print(f"Error copying data for param with shape {param.shape}: {e}")
                    print(f"Slice shape: {flat_params[pointer : pointer + numel].shape}")
                    raise e
                pointer += numel

        # Sanity check: did we use all the parameters?
        if pointer != self.param_dim:
            print(f"Warning: Unflattening finished, but pointer ({pointer}) does not match expected param_dim ({self.param_dim}). This might indicate non-float or non-trainable params were present.")


    def train_flow_model(self):
        """Trains the Flow Matching model on the parameter buffer."""
        if not self.flow_model or not self.optimizer:
            print("FM model or optimizer not initialized, skipping training.")
            return
        if len(self.param_buffer) < self.fm_min_buffer:
            print(f"Buffer size {len(self.param_buffer)} < {self.fm_min_buffer}. Skipping FM training.")
            return

        print(f"Training Flow Model with {len(self.param_buffer)} samples...")
        # Ensure all params in buffer have the correct dimension
        valid_params_list = [p.cpu().float() for p in self.param_buffer if p.numel() == self.param_dim] # Move to CPU, ensure float

        if len(valid_params_list) < self.fm_min_buffer:
             print(f"Not enough valid params ({len(valid_params_list)}) in buffer matching param_dim {self.param_dim}. Skipping FM training.")
             self.last_fm_loss = float('inf') # Indicate FM is not ready
             return

        # Stack tensors on CPU
        params_tensor_cpu = torch.stack(valid_params_list).float()

        # Normalize parameters (using robust mean/std estimation if needed)
        # Simple mean/std for now, consider quantiles if extreme outliers persist
        self.param_mean = params_tensor_cpu.mean(dim=0, keepdim=True)
        self.param_std = params_tensor_cpu.std(dim=0, keepdim=True) + 1e-8 # Add epsilon for stability

        # Normalize on CPU, handle potential NaNs arising from zero std dev dimensions
        params_tensor_normalized_cpu = (params_tensor_cpu - self.param_mean) / self.param_std
        params_tensor_normalized_cpu = torch.nan_to_num(params_tensor_normalized_cpu, nan=0.0)

        # Create Dataset and DataLoader
        dataset = TensorDataset(params_tensor_normalized_cpu)
        # Avoid dropping last batch if dataset is small
        drop_last = len(dataset) > self.fm_batch_size
        # Use num_workers=0 for simplicity and compatibility
        loader = DataLoader(dataset, batch_size=self.fm_batch_size, shuffle=True, drop_last=drop_last, num_workers=0, pin_memory=True if self.device != 'cpu' else False)

        # --- FM Training Loop ---
        self.flow_model.train()
        # Re-initialize epoch scheduler for cosine annealing within this training session
        # Use a small eta_min to allow learning rate to decay close to zero
        self.scheduler_epochs = CosineAnnealingLR(self.optimizer, T_max=self.fm_epochs, eta_min=1e-7)

        initial_lr = self.optimizer.param_groups[0]['lr'] # Get LR after round scheduler step
        print(f"Starting FM Train. Initial LR for this session: {initial_lr:.2e}")

        losses = AverageMeter()
        for epoch in range(self.fm_epochs):
            epoch_losses = AverageMeter()
            for batch_idx, batch in enumerate(loader):
                if not batch: continue # Skip empty batches if any
                # Move normalized batch to device
                real_params_normalized = batch[0].to(self.device, non_blocking=True)

                # Sample time t and noise
                t = torch.rand(real_params_normalized.size(0), 1, device=self.device) * 0.999 + 0.001 # Avoid t=0 or t=1
                noise = torch.randn_like(real_params_normalized)

                # Create noisy parameters (interpolation)
                noise_params_normalized = real_params_normalized * t + noise * (1 - t)

                # Compute flow loss (using Huber loss from FlowMatchingModel)
                try:
                    flow_loss = self.flow_model.compute_flow_loss(real_params_normalized, noise_params_normalized, t)
                except Exception as e:
                    print(f"Error during compute_flow_loss: {e}")
                    print(f" Shapes - Real Params: {real_params_normalized.shape}, Noisy Params: {noise_params_normalized.shape}, t: {t.shape}")
                    traceback.print_exc()
                    continue # Skip batch on error


                if not torch.isfinite(flow_loss):
                    print(f"Warning: Non-finite FM loss: {flow_loss.item()} in epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                    # Consider zeroing gradients if optimizer state might be corrupted
                    self.optimizer.zero_grad()
                    continue

                epoch_losses.update(flow_loss.item(), real_params_normalized.size(0))

                # Backpropagation and optimization step
                self.optimizer.zero_grad()
                flow_loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), self.fm_grad_norm)
                self.optimizer.step()

            # Step the within-session epoch scheduler
            self.scheduler_epochs.step()
            current_epoch_lr = self.scheduler_epochs.get_last_lr()[0]

            # Log progress periodically
            if (epoch + 1) % 5 == 0 or epoch == self.fm_epochs - 1:
                 print(f"  FM Epoch {epoch+1}/{self.fm_epochs} Avg Loss: {epoch_losses.avg:.4f}, Current LR: {current_epoch_lr:.2e}")
            losses.update(epoch_losses.avg, 1) # Track average loss across epochs

        self.last_fm_loss = losses.avg
        print(f"Round {self.r} Flow Model Training Final Avg Loss: {self.last_fm_loss:.4f}")
        self.flow_model.eval() # Set model to evaluation mode after training


    def aggregate_parameters(self):
        """Aggregates parameters using optimized robust FlowMatch logic."""
        if not self.active_clients:
            print(f"Round {self.r}: No active clients selected.")
            return

        # Get current global parameters (flattened on CPU)
        current_global_params_flat_cpu = self._flatten_params(self.model)
        if self.param_dim == 0 or current_global_params_flat_cpu.numel() == 0:
            print("Warning: Global model has no trainable float parameters. Skipping aggregation.")
            return
        # Move to device for calculations
        current_global_params_flat = current_global_params_flat_cpu.to(self.device)

        # --- Stage 1: Collect Client Updates ---
        collected_client_data = []
        update_norms = []
        param_updates_list = [] # Store updates for agreement calculation

        print(f"Round {self.r}: Collecting updates from {len(self.active_clients)} clients.")
        for c in self.active_clients:
            try:
                # Flatten client model parameters on CPU
                client_params_flat_cpu = self._flatten_params(c.model)

                # Validate parameter dimension
                if client_params_flat_cpu.numel() != self.param_dim:
                    print(f"Warning: Client {c.client_idx} param dim mismatch ({client_params_flat_cpu.numel()} vs {self.param_dim}). Skipping.")
                    continue

                # Move to device for calculations
                client_params_flat = client_params_flat_cpu.to(self.device)

                # Calculate parameter update and its norm
                param_update = client_params_flat - current_global_params_flat
                update_norm = torch.norm(param_update).item()

                # Skip clients with non-finite or near-zero norms
                if not np.isfinite(update_norm) or update_norm < 1e-9:
                    print(f"Warning: Client {c.client_idx} has non-finite or near-zero norm ({update_norm:.2e}). Skipping.")
                    continue

                # Store relevant client data
                collected_client_data.append({
                    'idx': c.client_idx,
                    'params_cpu': client_params_flat_cpu.clone(), # Store CPU version for buffer
                    'params_dev': client_params_flat, # Store device version for calculations
                    'param_update': param_update,
                    'update_norm': update_norm,
                    'weight': c.num_train, # Use num_train for weighting
                })
                update_norms.append(update_norm)
                param_updates_list.append(param_update) # Add to list for agreement calc

            except Exception as e:
                print(f"Error collecting parameters from Client {c.client_idx}: {e}")
                traceback.print_exc()
                continue # Skip problematic client

        if not collected_client_data:
            print("No valid client parameters collected this round. Global model unchanged.")
            # Ensure the model state is consistent (unflatten original params)
            # self._unflatten_params(current_global_params_flat_cpu, self.model) # Not strictly needed if model wasn't modified
            self.previous_global_params_flat = current_global_params_flat_cpu.clone() # Store for next round reference
            return

        # --- Stage 2: Calculate Reference Direction and Agreement Score ---
        mean_update_direction = None
        agreement_score = 1.0 # Default to high agreement if calculation fails or not enough clients

        if len(param_updates_list) > 1:
            # Stack updates and calculate mean update
            stacked_updates = torch.stack(param_updates_list)
            mean_update = torch.mean(stacked_updates, dim=0)
            mean_update_norm = torch.norm(mean_update)

            # Calculate mean update direction if norm is sufficient
            if mean_update_norm > 1e-6:
                mean_update_direction = mean_update / mean_update_norm
                print(f"Using current round mean update direction (Norm: {mean_update_norm:.4f}).")

                # Calculate agreement score (average cosine similarity to the mean direction)
                similarities = []
                for update in param_updates_list:
                    update_norm_tensor = torch.norm(update)
                    if update_norm_tensor > 1e-6:
                        # Ensure correct shapes for cosine_similarity: (N, D), (N, D) -> (N,) or (D,), (D,) -> scalar
                        sim = F.cosine_similarity((update / update_norm_tensor).unsqueeze(0), mean_update_direction.unsqueeze(0)).item()
                        similarities.append(sim)

                if similarities:
                     # Clip average similarity to be non-negative
                     agreement_score = max(0.0, sum(similarities) / len(similarities))
                     print(f"Update agreement score (avg sim to mean): {agreement_score:.3f}")
                else:
                     print("Could not calculate similarities.")
                     # agreement_score remains 1.0
            else:
                print("Mean update norm too small, cannot calculate reliable direction.")
                mean_update_direction = None # Cannot use cosine filter
                # agreement_score remains 1.0
        else:
            print("Not enough updates (<2) for agreement calculation.")
            # agreement_score remains 1.0

        # --- Stage 3: Adaptive Filtering ---
        accepted_client_data = []
        total_samples_accepted = 0

        # Calculate norm threshold (using the OPTIMIZED percentile)
        norm_thresh_val = np.percentile(update_norms, self.norm_percentile_threshold) if update_norms else float('inf')
        print(f"Adaptive norm threshold ({self.norm_percentile_threshold}th percentile): {norm_thresh_val:.4f}")

        # Determine if cosine similarity filter should be used
        use_cosine_filter = mean_update_direction is not None and self.r >= self.relax_cosine_rounds

        if not use_cosine_filter:
             print(f"Cosine similarity filter disabled (Round {self.r} < {self.relax_cosine_rounds} or no reference direction).")

        # Iterate through collected data and apply filters
        for data in collected_client_data:
            client_idx = data['idx']
            norm = data['update_norm']
            param_update = data['param_update']
            cosine_sim = -1.0 # Default value if not calculated

            # Calculate cosine similarity if filter is active
            if use_cosine_filter:
                 update_norm_tensor = torch.norm(param_update)
                 if update_norm_tensor > 1e-6:
                     normalized_update = param_update / update_norm_tensor
                     # Ensure shapes are compatible for cosine_similarity
                     cosine_sim = F.cosine_similarity(normalized_update.unsqueeze(0), mean_update_direction.unsqueeze(0)).item()
                 else:
                     # Handle zero-norm updates (should have been filtered, but as safety)
                     cosine_sim = 1.0 # Treat as perfectly aligned if norm is zero
            data['cosine_sim'] = cosine_sim

            # Determine the effective cosine threshold for this client
            current_cosine_threshold = self.cosine_base_threshold # Start with OPTIMIZED base
            is_potentially_harmful = self.augmented_scenario and client_idx < 50 # Check group
            if is_potentially_harmful:
                current_cosine_threshold += self.cosine_stricter_offset # Add offset for this group

            # --- Incorporate OPTIMIZED Low Agreement Logic ---
            # If penalty is 0, this block does nothing. If negative, it relaxes the threshold.
            if self.adaptive_cosine and agreement_score < self.low_agreement_threshold and self.cosine_low_agreement_penalty != 0.0:
                print(f"  Low agreement detected ({agreement_score:.3f} < {self.low_agreement_threshold:.3f}), adjusting cosine threshold by {self.cosine_low_agreement_penalty:.3f}.")
                current_cosine_threshold += self.cosine_low_agreement_penalty # Add penalty (which is 0 or negative)

            # Apply filters
            norm_passed = norm <= norm_thresh_val
            sim_passed = (not use_cosine_filter) or (cosine_sim >= current_cosine_threshold)

            # Log filtering decision
            log_msg = f"Client {client_idx}: Norm={norm:.3f} (Thresh={norm_thresh_val:.3f}, Passed={norm_passed})"
            if use_cosine_filter:
                log_msg += f", Sim={cosine_sim:.3f} (Thresh={current_cosine_threshold:.3f}, Passed={sim_passed})"
            else:
                log_msg += ", Sim=N/A"

            if norm_passed and sim_passed:
                accepted_client_data.append(data)
                total_samples_accepted += data['weight']
                # Add accepted client parameters (CPU version) to the buffer
                self.param_buffer.append(data['params_cpu'])
                # print(f"{log_msg} -> ACCEPTED") # Optional: Log accepted clients
            else:
                print(f"{log_msg} -> REJECTED")

        # Handle case where no clients pass filters
        if not accepted_client_data:
            print("No clients accepted after filtering. Global model remains unchanged.")
            # self._unflatten_params(current_global_params_flat_cpu, self.model) # Model already holds these params
            self.previous_global_params_flat = current_global_params_flat_cpu.clone()
            return

        print(f"Accepted {len(accepted_client_data)}/{len(collected_client_data)} clients after filtering.")

        # --- Stage 4: Update Reliability Scores (for accepted clients) ---
        num_accepted = len(accepted_client_data)
        if num_accepted > 0:
            # Calculate mean and std dev of norms among ACCEPTED clients for reliability scoring
            accepted_norms = [d['update_norm'] for d in accepted_client_data]
            mean_norm_accepted = np.mean(accepted_norms) if accepted_norms else 0.0
            std_norm_accepted = np.std(accepted_norms) + 1e-8 if accepted_norms else 1e-8

            for data in accepted_client_data:
                client_idx = data['idx']
                norm = data['update_norm']
                # Normalize norm difference based on accepted clients' stats
                normalized_diff_norm = (norm - mean_norm_accepted) / std_norm_accepted
                # Calculate reliability score based on normalized difference (closer to mean is better)
                current_round_reliability = torch.exp(-0.5 * torch.abs(torch.tensor(normalized_diff_norm))).clamp(0.0, 1.0).item()

                # Smooth reliability score using EMA
                previous_score = self.client_reliability_scores.get(client_idx, current_round_reliability) # Default to current if no history
                new_score = self.reliability_beta * previous_score + (1 - self.reliability_beta) * current_round_reliability
                self.client_reliability_scores[client_idx] = new_score
                data['reliability_score'] = new_score # Store for potential use in refinement


        # --- Stage 5: Train Flow Model & Step Round Scheduler ---
        fm_trained_this_round = False
        # Check if conditions are met for FM training
        if self.flow_model and len(self.param_buffer) >= self.fm_min_buffer and self.r % self.fm_train_freq == 0:
            print(f"Round {self.r}: Training FM (Freq={self.fm_train_freq}, Buffer={len(self.param_buffer)} >= {self.fm_min_buffer}).")
            train_start_time = time.time()
            self.train_flow_model() # Train FM (uses OPTIMIZED fm_epochs)
            fm_train_time = time.time() - train_start_time
            print(f"FM Training took {fm_train_time:.2f}s.")
            fm_trained_this_round = True
            # Step the round-based LR scheduler ONLY after successful FM training
            if self.fm_lr_scheduler_rounds:
                self.fm_lr_scheduler_rounds.step()
                current_fm_lr = self.fm_lr_scheduler_rounds.get_last_lr()[0]
                print(f"Stepped round FM LR scheduler. Current base FM LR: {current_fm_lr:.2e}")
        elif self.flow_model:
            print(f"Round {self.r}: Skipping FM training (Condition not met). Last FM Loss: {self.last_fm_loss:.4f}")
            if self.fm_lr_scheduler_rounds:
                 current_fm_lr = self.fm_lr_scheduler_rounds.get_last_lr()[0] # Log current LR even if not stepped
                 print(f"FM LR remains: {current_fm_lr:.2e} (FM not trained this round)")
        else:
             print(f"Round {self.r}: Skipping FM training (FM not initialized).")


        # --- Stage 6: Parameter Generation / Refinement ---
        processed_params_list = []
        processed_weights = [] # Weights for final aggregation

        # Check if FM is ready and trustworthy for generation/refinement
        fm_ready_for_gen = (self.flow_model is not None and
                            self.param_mean is not None and # Needs mean/std from training
                            self.param_std is not None and
                            self.last_fm_loss != float('inf')) # Needs a valid loss value

        fm_is_trustworthy = fm_ready_for_gen and (self.last_fm_loss < self.fm_trust_loss_threshold)

        if fm_ready_for_gen:
             status_msg = "TRUSTWORTHY" if fm_is_trustworthy else "NOT TRUSTWORTHY"
             print(f"Flow Model is ready and {status_msg} (Loss: {self.last_fm_loss:.4f} vs Thresh: {self.fm_trust_loss_threshold:.4f}).")
             if fm_is_trustworthy: print("FM Refinement enabled for low reliability clients.")
             else: print("FM Refinement disabled.")
        else:
             print("Flow Model not ready/trained. FM Refinement disabled.")

        refinement_count = 0
        if fm_is_trustworthy:
             # Move mean/std to device once, ensure they are 1D
             mean_1d_dev = self.param_mean.squeeze().to(self.device)
             std_1d_dev = self.param_std.squeeze().to(self.device)
             self.flow_model.eval() # Ensure FM is in eval mode

        # Process each accepted client
        for data in accepted_client_data:
            client_idx = data['idx']
            params_flat_dev = data['params_dev'] # Use parameters already on device
            reliability_score = data.get('reliability_score', self.mix_alpha_min) # Get smoothed reliability
            final_flat_params = params_flat_dev # Default to original filtered params

            try:
                # Apply refinement only if FM is trustworthy AND client reliability is below the OPTIMIZED threshold
                if fm_is_trustworthy and reliability_score < self.fm_reliability_threshold:
                     # Normalize parameters for FM input
                     params_norm = (params_flat_dev - mean_1d_dev) / std_1d_dev
                     params_norm = torch.nan_to_num(params_norm, nan=0.0) # Handle potential division by zero

                     # Generate parameters using FM (expects batched input)
                     generated_norm_batched = self.flow_model.generate(params_norm.unsqueeze(0), num_steps=self.fm_gen_steps)
                     generated_norm = generated_norm_batched.squeeze(0) # Remove batch dim

                     # Check for issues during generation
                     if torch.isnan(generated_norm).any() or torch.isinf(generated_norm).any():
                         print(f"  Client {client_idx}: Generation resulted in NaN/Inf. Skipping refinement.")
                         # Keep final_flat_params as the original filtered params
                     else:
                         # De-normalize generated parameters
                         generated_flat = generated_norm * std_1d_dev + mean_1d_dev
                         generated_flat = torch.nan_to_num(generated_flat, nan=0.0) # Ensure no NaNs

                         # --- Dynamic Alpha Calculation (from original code) ---
                         # Base alpha depends on reliability score relative to threshold
                         base_alpha = self.mix_alpha_min + (self.mix_alpha_max - self.mix_alpha_min) * (reliability_score / self.fm_reliability_threshold)
                         # Adjust based on FM loss (better FM -> more trust -> smaller min alpha allowed)
                         loss_factor = 1.0 - max(0, min(1, self.last_fm_loss / self.fm_trust_loss_threshold)) # 0 (bad loss) to 1 (good loss)
                         dynamic_min_alpha = self.mix_alpha_min * (1 - loss_factor * (1 - self.fm_loss_trust_scale)) # Lower bound adjusts based on loss
                         # Interpolate alpha between dynamic min and max based on reliability
                         mix_alpha = dynamic_min_alpha + (self.mix_alpha_max - dynamic_min_alpha) * (reliability_score / self.fm_reliability_threshold)
                         # Clamp alpha within bounds
                         mix_alpha = max(self.mix_alpha_min, min(self.mix_alpha_max, mix_alpha))
                         # --- End Dynamic Alpha ---

                         # Mix original and generated parameters
                         final_flat_params = mix_alpha * params_flat_dev + (1 - mix_alpha) * generated_flat
                         refinement_count += 1
                         print(f"  Client {client_idx}: Applied FM refinement (Rel={reliability_score:.3f} < {self.fm_reliability_threshold:.3f}, FM Loss={self.last_fm_loss:.3f}, Alpha={mix_alpha:.3f}).")
                # else: # Optional logging for skipped refinement
                #      if fm_is_trustworthy and reliability_score >= self.fm_reliability_threshold:
                #            # print(f"  Client {client_idx}: Skipped FM (Reliability High).")
                #            pass
                #      elif not fm_is_trustworthy:
                #            # print(f"  Client {client_idx}: Skipped FM (FM Untrustworthy).")
                #            pass


                # Final clamping and NaN handling for safety
                final_clamp_val = 20.0 # Clamp range for aggregated/refined params
                final_flat_params = torch.clamp(final_flat_params, -final_clamp_val, final_clamp_val)
                final_flat_params = torch.nan_to_num(final_flat_params, nan=0.0, posinf=final_clamp_val, neginf=-final_clamp_val)

                # Add processed parameters (on device) and weights to lists for aggregation
                processed_params_list.append(final_flat_params.detach()) # Detach to prevent gradient flow
                processed_weights.append(data['weight'])

            except Exception as e:
                print(f"Error during parameter refinement for client {client_idx}: {e}. Using original filtered params.")
                traceback.print_exc()
                # Fallback: use the original filtered params if refinement fails
                final_clamp_val = 20.0
                final_flat_params = torch.clamp(params_flat_dev, -final_clamp_val, final_clamp_val)
                final_flat_params = torch.nan_to_num(final_flat_params, nan=0.0, posinf=final_clamp_val, neginf=-final_clamp_val)
                processed_params_list.append(final_flat_params.detach())
                processed_weights.append(data['weight'])

        # Log refinement summary
        if fm_is_trustworthy:
             print(f"Applied FM refinement to {refinement_count}/{len(accepted_client_data)} accepted clients (Rel < {self.fm_reliability_threshold:.2f}).")
        # else: # Already logged that refinement was disabled

        # --- Stage 7: Weighted Aggregation ---
        if not processed_params_list:
            print("No parameters processed after refinement stage. Global model remains unchanged.")
            # self._unflatten_params(current_global_params_flat_cpu, self.model)
            self.previous_global_params_flat = current_global_params_flat_cpu.clone()
            return

        aggregated_flat_params = None
        # Ensure total weight is positive before normalizing
        if total_samples_accepted <= 1e-9: # Use a small epsilon for float comparison
            print("Warning: Total weight zero or negligible. Using simple average.")
            if len(processed_params_list) > 0:
                # Stack tensors on the correct device for mean calculation
                aggregated_flat_params = torch.mean(torch.stack(processed_params_list), dim=0)
            else:
                 # Fallback to current global params if list is somehow empty
                 aggregated_flat_params = current_global_params_flat
        else:
            # Perform weighted average
            client_weights_tensor = torch.tensor(processed_weights, dtype=torch.float32, device=self.device)
            normalized_weights = client_weights_tensor / total_samples_accepted
            # Initialize aggregated params tensor on the correct device
            aggregated_flat_params = torch.zeros(self.param_dim, dtype=torch.float32, device=self.device)
            for i, final_flat in enumerate(processed_params_list):
                # Sanity check dimension before adding
                if final_flat.numel() == self.param_dim:
                    aggregated_flat_params += normalized_weights[i] * final_flat # Use item() if weights are single-element tensors
                else:
                    print(f"Error: Param size mismatch during weighted average for client {accepted_client_data[i]['idx']}. Skipping its contribution.")

        # --- Stage 8: Update Global Model ---
        if aggregated_flat_params is None or aggregated_flat_params.numel() != self.param_dim:
            print("Aggregation failed (result is None or has wrong dim). Global model remains unchanged.")
            # self._unflatten_params(current_global_params_flat_cpu, self.model)
            self.previous_global_params_flat = current_global_params_flat_cpu.clone()
            return

        try:
            # Final safety clamp and NaN check before unflattening
            final_clamp_val = 20.0
            aggregated_flat_params = torch.clamp(aggregated_flat_params, -final_clamp_val, final_clamp_val)
            aggregated_flat_params = torch.nan_to_num(aggregated_flat_params, nan=0.0, posinf=final_clamp_val, neginf=-final_clamp_val)

            # Unflatten into the global model (expects CPU tensor)
            self._unflatten_params(aggregated_flat_params.cpu(), self.model)

            # Store the newly aggregated parameters (CPU version) for the next round
            self.previous_global_params_flat = aggregated_flat_params.cpu().clone()
            print(f"Global model updated using {len(processed_params_list)} clients. Aggregation: Optimized Filtered, Adaptive FM Mix (Trustworthy={fm_is_trustworthy}).")

        except Exception as e:
            print(f"CRITICAL Error unflattening aggregated parameters: {e}. Restoring previous model state.")
            traceback.print_exc()
            # Restore previous state if unflattening fails
            if self.previous_global_params_flat is not None:
                 self._unflatten_params(self.previous_global_params_flat, self.model)
            else: # Fallback if even previous state is unavailable (shouldn't happen often)
                 self._unflatten_params(current_global_params_flat_cpu, self.model)


    # --- train method (Copied from your original code, ensure it calls the modified aggregate_parameters) ---
    def train(self):
        print("\n>>> Optimized Robust FlowMatch Training Starting <<<")
        print(">>> Ensure client-side LR scheduling is active! <<<")

        for r_loop in range(1, self.global_rounds + 1):
            start_time = time.time()
            self.r = r_loop

            self.sample_active_clients()
            if not self.active_clients:
                print(f"Round {self.r}: No clients selected. Skipping round.")
                # Optional: Add a small delay if skipping rounds frequently
                # time.sleep(1)
                continue

            # Send the current global model to the selected clients
            self.send_models()

            # Train clients locally
            train_acc, train_loss = float('nan'), float('nan') # Initialize with NaN
            try:
                # train_clients should return average accuracy and loss
                train_acc, train_loss = self.train_clients()
                # Ensure results are finite floats
                if isinstance(train_loss, torch.Tensor): train_loss = train_loss.item() if torch.isfinite(train_loss).all() else float('nan')
                elif not isinstance(train_loss, (float, int)) or not np.isfinite(train_loss): train_loss = float('nan')
                if isinstance(train_acc, torch.Tensor): train_acc = train_acc.item() if torch.isfinite(train_acc).all() else float('nan')
                elif not isinstance(train_acc, (float, int)) or not np.isfinite(train_acc): train_acc = float('nan')
            except Exception as e:
                print(f"Error during client training in round {self.r}: {e}")
                traceback.print_exc()
                # Keep train_acc, train_loss as NaN

            train_time = time.time() - start_time

            # Aggregate parameters using the optimized method
            aggregation_start_time = time.time()
            self.aggregate_parameters() # Calls the optimized robust aggregation
            aggregation_time = time.time() - aggregation_start_time

            round_time = time.time() - start_time
            self.train_times.append(train_time) # Assuming self.train_times exists from ServerBase
            self.round_times.append(round_time) # Assuming self.round_times exists from ServerBase

            # Evaluate periodically
            if self.r % self.eval_gap == 0 or self.r == self.global_rounds:
                print(f"\n--- Evaluating Round {self.r}/{self.global_rounds} ---")
                test_acc, test_loss, test_acc_std = self.evaluate()
                ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized()
                print(f"Round [{self.r}/{self.global_rounds}] Results:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%") # Added % sign
                print(f"  Test Loss (Global): {test_loss:.4f} | Test Acc (Global): {test_acc:.2f}% (Std: {test_acc_std:.2f}%)")
                print(f"  Test Loss (Pers): {ptest_loss:.4f} | Test Acc (Pers): {ptest_acc:.2f}% (Std: {ptest_acc_std:.2f}%)")
                print(f"  Timings: Train={train_time:.2f}s | Aggr={aggregation_time:.2f}s | Round={round_time:.2f}s")
                # Log current base FM LR if applicable
                if self.fm_lr_scheduler_rounds:
                     current_fm_base_lr = self.fm_lr_scheduler_rounds.get_last_lr()[0]
                     print(f"  Current Base FM LR: {current_fm_base_lr:.2e}")
                print(f"--- End Evaluation ---")
            else:
                # More concise logging for non-evaluation rounds
                print(f"Round [{self.r}/{self.global_rounds}] Completed. Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Time: {round_time:.2f}s")

        print("\n>>> Optimized Robust FlowMatch Training Finished <<<")


    # --- evaluate method (Copied from your original code) ---
    def evaluate(self):
        total_samples = 0
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        self.model.eval() # Ensure model is in eval mode
        self.model.to(self.device) # Ensure model is on the correct device

        for c in self.clients:
             if c.num_test == 0: continue # Skip clients with no test data
             try:
                 # Client's evaluate method should use the current global model temporarily
                 # Assuming client.evaluate() handles setting the model internally or takes it as arg
                 # A safer way is to temporarily set the client model:
                 original_client_model_state = deepcopy(c.model.state_dict()) # Save client state if it's personalized
                 c.set_model(self.model) # Set global model for evaluation
                 acc, loss = c.evaluate()
                 c.model.load_state_dict(original_client_model_state) # Restore client state

                 weight = c.num_test
                 acc_val = acc.item() if isinstance(acc, torch.Tensor) else acc
                 loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

                 if np.isfinite(acc_val) and np.isfinite(loss_val):
                     accs.append(acc_val * 100.0) # Store as percentage
                     weighted_loss += weight * loss_val
                     weighted_acc += weight * acc_val # Accumulate weighted accuracy
                     total_samples += weight
                 else:
                     print(f"Warning: Non-finite acc/loss during global eval for client {c.client_idx}. Skipping.")
             except Exception as e:
                 print(f"Error evaluating global model on client {c.client_idx}: {e}")
                 traceback.print_exc()

        if total_samples == 0: return 0.0, 0.0, 0.0
        avg_loss = weighted_loss / total_samples
        avg_acc = (weighted_acc / total_samples) # Convert final avg acc to percentage
        std_acc = np.std(accs) if accs else 0.0 # Calculate std dev on percentages
        return avg_acc, avg_loss, std_acc


    # --- evaluate_personalized method (Copied from your original code) ---
    def evaluate_personalized(self):
        total_samples = 0
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        # No need to move self.model here, it's handled per client

        for c in self.clients:
            if c.num_test == 0: continue
            try:
                # Load global model state into client model for fine-tuning
                global_model_state = deepcopy(self.model.state_dict())
                c.model.load_state_dict(global_model_state)
                c.model.to(c.device) # Ensure client model is on its device

                # Fine-tune (assuming client.train() performs fine-tuning steps)
                c.train() # This should ideally be a separate fine-tuning method if different from regular training

                # Evaluate the fine-tuned model
                acc, loss = c.evaluate()

                weight = c.num_test
                acc_val = acc.item() if isinstance(acc, torch.Tensor) else acc
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

                if np.isfinite(acc_val) and np.isfinite(loss_val):
                    accs.append(acc_val * 100.0) # Store as percentage
                    weighted_loss += weight * loss_val
                    weighted_acc += weight * acc_val # Accumulate weighted accuracy
                    total_samples += weight
                else:
                    print(f"Warning: Non-finite acc/loss during personalized eval for client {c.client_idx}. Skipping.")
            except Exception as e:
                 print(f"Error during personalized evaluation for client {c.client_idx}: {e}")
                 traceback.print_exc()

        if total_samples == 0: return 0.0, 0.0, 0.0
        avg_loss = weighted_loss / total_samples
        avg_acc = (weighted_acc / total_samples) # Convert final avg acc to percentage
        std_acc = np.std(accs) if accs else 0.0 # Calculate std dev on percentages
        return avg_acc, avg_loss, std_acc