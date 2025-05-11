import time
from copy import deepcopy
import torch
from servers.server_base import Server # Assuming server_base defines Server class
from clients.client_fedavg import ClientFedAvg # Assuming client_fedavg defines ClientFedAvg
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, TensorDataset
from utils.util import AverageMeter # Assuming utils.util defines AverageMeter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import numpy as np
from collections import deque
import traceback
import random

# --- FlowMatchingModel remains the same ---
class FlowMatchingModel(nn.Module):
    # Using dropout_rate=0.15 initially, consider tuning (e.g., 0.1)
    def __init__(self, param_dim, hidden_dim=2048, rank=256, dropout_rate=0.15):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256))
        # Increased capacity slightly (more robust to noise)
        self.input_norm = nn.LayerNorm(param_dim)
        self.low_rank_proj = nn.Linear(param_dim, rank)
        self.main_net = nn.Sequential(
            nn.Linear(rank + 256, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
            # Added one more layer for potentially higher complexity
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
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
    def generate(self, init_params, num_steps=150): # Increased num_steps slightly
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
        self.augmented_scenario = getattr(args, 'augmented', False) # Keep True if using corrupted data
        self.adaptive_cosine = getattr(args, 'adaptive_cosine', True) # Keep True for adaptivity

        # --- Hyperparameters Tuned for CIFAR-100 ---
        print(f"Optimized Robust FlowMatch Server Initializing for CIFAR-100...")

        # 1. Stricter Filtering (Higher thresholds = Stricter)
        self.cosine_base_threshold = getattr(args, 'cosine_base_threshold', 0.0) # CIFAR100: Stricter (was -0.15)
        self.cosine_stricter_offset = getattr(args, 'cosine_stricter_offset', 0.1) # Keep offset for corrupted clients
        # Important Change: Make penalty *stricter* during low agreement
        self.cosine_low_agreement_penalty = getattr(args, 'cosine_low_agreement_penalty', 0.1) # CIFAR100: POSITIVE penalty (adds to threshold, was 0.0)
        self.low_agreement_threshold = getattr(args, 'low_agreement_threshold', 0.2) # CIFAR100: Stricter low agreement definition (was 0.3)

        # 2. Stricter Norm Filtering
        self.norm_percentile_threshold = getattr(args, 'norm_percentile_threshold', 90) # CIFAR100: Stricter (was 95)

        # 3. More Robust Flow Model Training
        self.fm_epochs = getattr(self.args, 'fm_epochs', 30) # CIFAR100: Increased (was 15)
        self.fm_lr = getattr(self.args, 'fm_lr', 2e-5) # CIFAR100: Reduced slightly (was 3e-5)
        self.fm_wd = getattr(self.args, 'fm_wd', 1e-4) # Keep same
        self.fm_dropout_rate = getattr(self.args, 'fm_dropout_rate', 0.1) # CIFAR100: Reduced dropout slightly (was 0.15)
        self.fm_min_buffer = getattr(self.args, 'fm_min_buffer', 1000) # CIFAR100: Increased significantly (was 500)
        self.fm_buffer_size = getattr(self.args, 'fm_buffer_size', 5000) # CIFAR100: Increased significantly (was 2000)
        self.fm_batch_size = getattr(self.args, 'fm_batch_size', 128) # Keep same, maybe decrease if OOM
        self.fm_train_freq = getattr(self.args, 'fm_train_freq', 1) # Train FM every round if buffer is sufficient
        self.fm_grad_norm = getattr(self.args, 'fm_grad_norm', 1.0) # Keep same

        # 4. More Cautious & Wider Refinement
        self.fm_trust_loss_threshold = getattr(self.args, 'fm_trust_loss_threshold', 0.15) # CIFAR100: Stricter (was 0.25)
        self.fm_reliability_threshold = getattr(args, 'fm_reliability_threshold', 0.65) # CIFAR100: Lower significantly to refine more clients (was 0.92)
        self.fm_gen_steps = getattr(self.args, 'fm_gen_steps', 150) # CIFAR100: Increased slightly (was 100)

        # Mixing parameters (kept from original tuned version, but effect depends on other params)
        self.mix_alpha_min = getattr(args, 'mix_alpha_min', 0.8)
        self.mix_alpha_max = getattr(args, 'mix_alpha_max', 0.95)
        self.fm_loss_trust_scale = getattr(args, 'fm_loss_trust_scale', 0.05)

        # Reliability update parameter
        self.reliability_beta = getattr(args, 'reliability_beta', 0.9)
        # Rounds before cosine filter activates
        self.relax_cosine_rounds = getattr(args, 'relax_cosine_rounds', 5) # Keep same

        print(f"Augmented (Corruption) Scenario: {self.augmented_scenario}")
        print(f"Using Adaptive Cosine Thresholds: {self.adaptive_cosine}")
        print(f"  - Base Cosine Threshold: {self.cosine_base_threshold}")
        print(f"  - Stricter Offset (Corrupted Clients): {self.cosine_stricter_offset}")
        print(f"  - Low Agreement Penalty (Stricter): +{self.cosine_low_agreement_penalty}")
        print(f"  - Low Agreement Definition: < {self.low_agreement_threshold}")
        print(f"Norm Percentile Threshold: {self.norm_percentile_threshold}")
        print(f"FM Min Buffer Size: {self.fm_min_buffer}")
        print(f"FM Buffer Size: {self.fm_buffer_size}")
        print(f"FM Training Epochs: {self.fm_epochs}")
        print(f"FM Learning Rate: {self.fm_lr}")
        print(f"FM Trust Loss Threshold: {self.fm_trust_loss_threshold}")
        print(f"FM Reliability Threshold (for Refinement): {self.fm_reliability_threshold}")
        print(f"FM Generation Steps: {self.fm_gen_steps}")
        print(f"FM Dropout Rate: {self.fm_dropout_rate}")


        self.clients = []
        for client_idx in range(self.num_clients):
            try:
                 c = ClientFedAvg(args, client_idx)
                 self.clients.append(c)
            except Exception as e:
                 print(f"Error creating Client {client_idx}: {e}")
                 print("Ensure ClientFedAvg class is defined and args are appropriate.")
                 raise e


        self.flow_model = None
        self.last_fm_loss = float('inf')
        self.client_reliability_scores = {} # Store reliability scores {client_idx: score}
        self.param_buffer = deque(maxlen=self.fm_buffer_size) # Use updated buffer size

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
        self.previous_global_params_flat = None # Store the previous round's aggregated params


    def init_flow_model(self):
        """Initializes the FlowMatchingModel, optimizer, and schedulers."""
        if self.param_dim == 0 or self.flow_model is not None: # Avoid re-initialization if already done
             if self.param_dim == 0: print("FM not initialized (param_dim is 0).")
             return

        print(f"Initializing FlowMatchingModel with param_dim={self.param_dim}, dropout={self.fm_dropout_rate}")
        # Consider increasing FM capacity if needed (e.g., hidden_dim or rank), but start with tuned hyperparameters
        self.flow_model = FlowMatchingModel(self.param_dim, dropout_rate=self.fm_dropout_rate).to(self.device)
        self.optimizer = torch.optim.AdamW(self.flow_model.parameters(), lr=self.fm_lr, weight_decay=self.fm_wd)

        # Calculate total FM training steps based on rounds and frequency
        total_fm_train_events = self.global_rounds // self.fm_train_freq
        if total_fm_train_events == 0: total_fm_train_events = 1 # Avoid division by zero

        # Linear decay across rounds for the base LR used in AdamW
        self.fm_lr_scheduler_rounds = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_fm_train_events)
        print(f"Initialized FM LR scheduler across rounds (Linear decay to 0.1 over ~{total_fm_train_events} FM training events).")

        # Epoch scheduler will be re-initialized before each FM training session
        self.scheduler_epochs = None


    def send_models(self):
        """Sends the current global model to active clients."""
        if not self.active_clients: return
        self.model.to(self.device)
        for c in self.active_clients:
            try:
                c.set_model(deepcopy(self.model).to(c.device))
            except Exception as e:
                print(f"Error sending model to Client {c.client_idx}: {e}")
                traceback.print_exc()

    def _flatten_params(self, model):
        """Flattens trainable float parameters of a model."""
        original_device = next(model.parameters()).device
        model.cpu()
        trainable_params = [p.detach().clone().float() for p in model.parameters() if p.requires_grad and p.is_floating_point()]
        model.to(original_device)

        if not trainable_params:
            return torch.tensor([], dtype=torch.float32, device='cpu')
        flat_params = torch.cat([p.flatten() for p in trainable_params])
        return flat_params # Remains on CPU

    def _unflatten_params(self, flat_params_cpu, model):
        """Unflattens parameters back into the model structure."""
        if self.param_dim == 0:
            print("Warning: Cannot unflatten, param_dim is 0.")
            return
        if flat_params_cpu.numel() != self.param_dim:
             raise ValueError(f"Parameter dimension mismatch: flat_params_cpu has {flat_params_cpu.numel()} elements, expected {self.param_dim}")

        model.to(self.device)
        flat_params = flat_params_cpu.to(self.device)

        pointer = 0
        for param in model.parameters():
            if param.requires_grad and param.is_floating_point():
                numel = param.numel()
                if pointer + numel > flat_params.numel():
                     raise ValueError(f"Flattened params tensor too short. Pointer {pointer}, numel {numel}, total {flat_params.numel()}")
                try:
                    param.data.copy_(flat_params[pointer : pointer + numel].view_as(param.data))
                except Exception as e:
                    print(f"Error copying data for param with shape {param.shape}: {e}")
                    print(f"Slice shape: {flat_params[pointer : pointer + numel].shape}")
                    raise e
                pointer += numel
        # This warning might appear if model has non-float/non-trainable params, generally okay.
        # if pointer != self.param_dim:
        #    print(f"Warning: Unflattening finished pointer ({pointer}) != param_dim ({self.param_dim}).")


    def train_flow_model(self):
        """Trains the Flow Matching model on the parameter buffer."""
        if not self.flow_model or not self.optimizer:
            print("FM model or optimizer not initialized, skipping training.")
            return
        # Use updated minimum buffer requirement
        if len(self.param_buffer) < self.fm_min_buffer:
            print(f"Buffer size {len(self.param_buffer)} < {self.fm_min_buffer}. Skipping FM training.")
            return

        print(f"Training Flow Model with {len(self.param_buffer)} samples...")
        valid_params_list = [p.cpu().float() for p in self.param_buffer if p.numel() == self.param_dim]

        # Check against updated minimum buffer requirement again after validation
        if len(valid_params_list) < self.fm_min_buffer:
             print(f"Not enough valid params ({len(valid_params_list)}) in buffer matching param_dim {self.param_dim} after validation. Skipping FM training.")
             self.last_fm_loss = float('inf')
             return

        params_tensor_cpu = torch.stack(valid_params_list).float()

        # Simple mean/std normalization (robust alternatives like median/IQR could be considered)
        self.param_mean = params_tensor_cpu.mean(dim=0, keepdim=True)
        self.param_std = params_tensor_cpu.std(dim=0, keepdim=True) + 1e-8 # Epsilon for stability

        params_tensor_normalized_cpu = (params_tensor_cpu - self.param_mean) / self.param_std
        params_tensor_normalized_cpu = torch.nan_to_num(params_tensor_normalized_cpu, nan=0.0) # Handle potential NaNs

        dataset = TensorDataset(params_tensor_normalized_cpu)
        drop_last = len(dataset) > self.fm_batch_size
        loader = DataLoader(dataset, batch_size=self.fm_batch_size, shuffle=True, drop_last=drop_last, num_workers=0, pin_memory=True if self.device != 'cpu' else False)

        # --- FM Training Loop ---
        self.flow_model.train()
        # Use updated number of epochs
        self.scheduler_epochs = CosineAnnealingLR(self.optimizer, T_max=self.fm_epochs, eta_min=1e-7)

        initial_lr = self.optimizer.param_groups[0]['lr']
        print(f"Starting FM Train. Epochs: {self.fm_epochs}. Initial LR for this session: {initial_lr:.2e}")

        losses = AverageMeter()
        for epoch in range(self.fm_epochs): # Use updated fm_epochs
            epoch_losses = AverageMeter()
            for batch_idx, batch in enumerate(loader):
                if not batch: continue
                real_params_normalized = batch[0].to(self.device, non_blocking=True)

                t = torch.rand(real_params_normalized.size(0), 1, device=self.device) * 0.999 + 0.001
                noise = torch.randn_like(real_params_normalized)
                noise_params_normalized = real_params_normalized * t + noise * (1 - t)

                try:
                    flow_loss = self.flow_model.compute_flow_loss(real_params_normalized, noise_params_normalized, t)
                except Exception as e:
                    print(f"Error during compute_flow_loss: {e}")
                    traceback.print_exc()
                    continue

                if not torch.isfinite(flow_loss):
                    print(f"Warning: Non-finite FM loss: {flow_loss.item()} in epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                    self.optimizer.zero_grad()
                    continue

                epoch_losses.update(flow_loss.item(), real_params_normalized.size(0))

                self.optimizer.zero_grad()
                flow_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), self.fm_grad_norm)
                self.optimizer.step()

            self.scheduler_epochs.step()
            current_epoch_lr = self.scheduler_epochs.get_last_lr()[0]

            # Log progress less frequently for longer training
            if (epoch + 1) % 10 == 0 or epoch == self.fm_epochs - 1:
                 print(f"  FM Epoch {epoch+1}/{self.fm_epochs} Avg Loss: {epoch_losses.avg:.4f}, Current LR: {current_epoch_lr:.2e}")
            losses.update(epoch_losses.avg, 1)

        self.last_fm_loss = losses.avg
        print(f"Round {self.r} Flow Model Training Final Avg Loss: {self.last_fm_loss:.4f}")
        self.flow_model.eval()


    def aggregate_parameters(self):
        """Aggregates parameters using optimized robust FlowMatch logic for CIFAR-100."""
        if not self.active_clients:
            print(f"Round {self.r}: No active clients selected.")
            return

        current_global_params_flat_cpu = self._flatten_params(self.model)
        if self.param_dim == 0 or current_global_params_flat_cpu.numel() == 0:
            print("Warning: Global model has no trainable float parameters. Skipping aggregation.")
            return
        current_global_params_flat = current_global_params_flat_cpu.to(self.device)

        # --- Stage 1: Collect Client Updates ---
        collected_client_data = []
        update_norms = []
        param_updates_list = [] # Store updates for agreement calculation

        print(f"Round {self.r}: Collecting updates from {len(self.active_clients)} clients.")
        for c in self.active_clients:
            try:
                client_params_flat_cpu = self._flatten_params(c.model)
                if client_params_flat_cpu.numel() != self.param_dim:
                    print(f"Warning: Client {c.client_idx} param dim mismatch ({client_params_flat_cpu.numel()} vs {self.param_dim}). Skipping.")
                    continue

                client_params_flat = client_params_flat_cpu.to(self.device)
                param_update = client_params_flat - current_global_params_flat
                update_norm = torch.norm(param_update).item()

                if not np.isfinite(update_norm) or update_norm < 1e-9:
                    print(f"Warning: Client {c.client_idx} has non-finite or near-zero norm ({update_norm:.2e}). Skipping.")
                    continue

                collected_client_data.append({
                    'idx': c.client_idx,
                    'params_cpu': client_params_flat_cpu.clone(),
                    'params_dev': client_params_flat,
                    'param_update': param_update,
                    'update_norm': update_norm,
                    'weight': c.num_train,
                })
                update_norms.append(update_norm)
                param_updates_list.append(param_update)

            except Exception as e:
                print(f"Error collecting parameters from Client {c.client_idx}: {e}")
                traceback.print_exc()
                continue

        if not collected_client_data:
            print("No valid client parameters collected this round. Global model unchanged.")
            self.previous_global_params_flat = current_global_params_flat_cpu.clone()
            return

        # --- Stage 2: Calculate Reference Direction and Agreement Score ---
        mean_update_direction = None
        agreement_score = 1.0

        if len(param_updates_list) > 1:
            stacked_updates = torch.stack(param_updates_list)
            mean_update = torch.mean(stacked_updates, dim=0)
            mean_update_norm = torch.norm(mean_update)

            if mean_update_norm > 1e-6:
                mean_update_direction = mean_update / mean_update_norm
                print(f"Using current round mean update direction (Norm: {mean_update_norm:.4f}).")
                similarities = []
                for update in param_updates_list:
                    update_norm_tensor = torch.norm(update)
                    if update_norm_tensor > 1e-6:
                        sim = F.cosine_similarity((update / update_norm_tensor).unsqueeze(0), mean_update_direction.unsqueeze(0)).item()
                        similarities.append(sim)
                if similarities:
                     agreement_score = max(0.0, sum(similarities) / len(similarities)) # Ensure non-negative
                     print(f"Update agreement score (avg sim to mean): {agreement_score:.3f}")
                else: print("Could not calculate similarities.")
            else: print("Mean update norm too small, cannot calculate reliable direction.")
        else: print("Not enough updates (<2) for agreement calculation.")

        # --- Stage 3: Adaptive Filtering (Using CIFAR-100 tuned thresholds) ---
        accepted_client_data = []
        total_samples_accepted = 0

        # Use updated norm threshold percentile
        norm_thresh_val = np.percentile(update_norms, self.norm_percentile_threshold) if update_norms else float('inf')
        print(f"Adaptive norm threshold ({self.norm_percentile_threshold}th percentile): {norm_thresh_val:.4f}")

        use_cosine_filter = mean_update_direction is not None and self.r >= self.relax_cosine_rounds
        if not use_cosine_filter:
             print(f"Cosine similarity filter disabled (Round {self.r} < {self.relax_cosine_rounds} or no reference direction).")

        for data in collected_client_data:
            client_idx = data['idx']
            norm = data['update_norm']
            param_update = data['param_update']
            cosine_sim = -1.0 # Default

            if use_cosine_filter:
                 update_norm_tensor = torch.norm(param_update)
                 if update_norm_tensor > 1e-6:
                     cosine_sim = F.cosine_similarity((param_update / update_norm_tensor).unsqueeze(0), mean_update_direction.unsqueeze(0)).item()
                 else: cosine_sim = 1.0
            data['cosine_sim'] = cosine_sim

            # Determine the effective cosine threshold
            current_cosine_threshold = self.cosine_base_threshold # Start with stricter base for C100
            is_potentially_harmful = self.augmented_scenario and client_idx < 50
            if is_potentially_harmful:
                current_cosine_threshold += self.cosine_stricter_offset

            # --- Incorporate Stricter Low Agreement Logic ---
            # If agreement is low, make the threshold *stricter* by adding the penalty
            if self.adaptive_cosine and agreement_score < self.low_agreement_threshold:
                print(f"  Low agreement detected ({agreement_score:.3f} < {self.low_agreement_threshold:.3f}), adjusting cosine threshold by +{self.cosine_low_agreement_penalty:.3f}.")
                current_cosine_threshold += self.cosine_low_agreement_penalty # Make stricter

            norm_passed = norm <= norm_thresh_val
            sim_passed = (not use_cosine_filter) or (cosine_sim >= current_cosine_threshold)

            log_msg = f"Client {client_idx}: Norm={norm:.3f} (Thresh={norm_thresh_val:.3f}, Passed={norm_passed})"
            if use_cosine_filter:
                log_msg += f", Sim={cosine_sim:.3f} (Thresh={current_cosine_threshold:.3f}, Passed={sim_passed})"
            else:
                log_msg += ", Sim=N/A"

            if norm_passed and sim_passed:
                accepted_client_data.append(data)
                total_samples_accepted += data['weight']
                self.param_buffer.append(data['params_cpu']) # Add accepted client parameters (CPU version) to the buffer
                # print(f"{log_msg} -> ACCEPTED")
            else:
                print(f"{log_msg} -> REJECTED")

        if not accepted_client_data:
            print("No clients accepted after filtering. Global model remains unchanged.")
            self.previous_global_params_flat = current_global_params_flat_cpu.clone()
            return

        print(f"Accepted {len(accepted_client_data)}/{len(collected_client_data)} clients after filtering.")

        # --- Stage 4: Update Reliability Scores ---
        num_accepted = len(accepted_client_data)
        if num_accepted > 0:
            accepted_norms = [d['update_norm'] for d in accepted_client_data]
            mean_norm_accepted = np.mean(accepted_norms) if accepted_norms else 0.0
            std_norm_accepted = np.std(accepted_norms) + 1e-8 if accepted_norms else 1e-8

            for data in accepted_client_data:
                client_idx = data['idx']
                norm = data['update_norm']
                normalized_diff_norm = (norm - mean_norm_accepted) / std_norm_accepted
                current_round_reliability = torch.exp(-0.5 * torch.abs(torch.tensor(normalized_diff_norm))).clamp(0.0, 1.0).item()
                previous_score = self.client_reliability_scores.get(client_idx, current_round_reliability)
                new_score = self.reliability_beta * previous_score + (1 - self.reliability_beta) * current_round_reliability
                self.client_reliability_scores[client_idx] = new_score
                data['reliability_score'] = new_score


        # --- Stage 5: Train Flow Model & Step Round Scheduler ---
        fm_trained_this_round = False
        if self.flow_model and len(self.param_buffer) >= self.fm_min_buffer and self.r % self.fm_train_freq == 0:
            print(f"Round {self.r}: Training FM (Freq={self.fm_train_freq}, Buffer={len(self.param_buffer)} >= {self.fm_min_buffer}).")
            train_start_time = time.time()
            self.train_flow_model() # Uses updated fm_epochs
            fm_train_time = time.time() - train_start_time
            print(f"FM Training took {fm_train_time:.2f}s.")
            fm_trained_this_round = True
            if self.fm_lr_scheduler_rounds:
                self.fm_lr_scheduler_rounds.step()
                current_fm_lr = self.fm_lr_scheduler_rounds.get_last_lr()[0]
                print(f"Stepped round FM LR scheduler. Current base FM LR: {current_fm_lr:.2e}")
        elif self.flow_model:
            print(f"Round {self.r}: Skipping FM training (Condition not met). Last FM Loss: {self.last_fm_loss:.4f}")
            if self.fm_lr_scheduler_rounds:
                 current_fm_lr = self.fm_lr_scheduler_rounds.get_last_lr()[0]
                 print(f"FM LR remains: {current_fm_lr:.2e} (FM not trained this round)")
        else:
             print(f"Round {self.r}: Skipping FM training (FM not initialized).")


        # --- Stage 6: Parameter Generation / Refinement (Using CIFAR-100 tuned thresholds) ---
        processed_params_list = []
        processed_weights = []

        fm_ready_for_gen = (self.flow_model is not None and
                            self.param_mean is not None and
                            self.param_std is not None and
                            self.last_fm_loss != float('inf'))

        # Use stricter trust threshold
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
             mean_1d_dev = self.param_mean.squeeze().to(self.device)
             std_1d_dev = self.param_std.squeeze().to(self.device)
             self.flow_model.eval()

        for data in accepted_client_data:
            client_idx = data['idx']
            params_flat_dev = data['params_dev']
            reliability_score = data.get('reliability_score', self.mix_alpha_min)
            final_flat_params = params_flat_dev

            try:
                # Apply refinement if FM trustworthy AND client reliability is below the *updated lower* threshold
                if fm_is_trustworthy and reliability_score < self.fm_reliability_threshold:
                     params_norm = (params_flat_dev - mean_1d_dev) / std_1d_dev
                     params_norm = torch.nan_to_num(params_norm, nan=0.0)

                     # Use updated generation steps
                     generated_norm_batched = self.flow_model.generate(params_norm.unsqueeze(0), num_steps=self.fm_gen_steps)
                     generated_norm = generated_norm_batched.squeeze(0)

                     if torch.isnan(generated_norm).any() or torch.isinf(generated_norm).any():
                         print(f"  Client {client_idx}: Generation resulted in NaN/Inf. Skipping refinement.")
                     else:
                         generated_flat = generated_norm * std_1d_dev + mean_1d_dev
                         generated_flat = torch.nan_to_num(generated_flat, nan=0.0)

                         # --- Dynamic Alpha Calculation (logic unchanged, but inputs differ) ---
                         base_alpha = self.mix_alpha_min + (self.mix_alpha_max - self.mix_alpha_min) * (reliability_score / self.fm_reliability_threshold)
                         loss_factor = 1.0 - max(0, min(1, self.last_fm_loss / self.fm_trust_loss_threshold))
                         dynamic_min_alpha = self.mix_alpha_min * (1 - loss_factor * (1 - self.fm_loss_trust_scale))
                         mix_alpha = dynamic_min_alpha + (self.mix_alpha_max - dynamic_min_alpha) * (reliability_score / self.fm_reliability_threshold)
                         mix_alpha = max(self.mix_alpha_min, min(self.mix_alpha_max, mix_alpha))
                         # --- End Dynamic Alpha ---

                         final_flat_params = mix_alpha * params_flat_dev + (1 - mix_alpha) * generated_flat
                         refinement_count += 1
                         print(f"  Client {client_idx}: Applied FM refinement (Rel={reliability_score:.3f} < {self.fm_reliability_threshold:.3f}, FM Loss={self.last_fm_loss:.3f}, Alpha={mix_alpha:.3f}).")

                # Final clamping and NaN handling
                final_clamp_val = 20.0
                final_flat_params = torch.clamp(final_flat_params, -final_clamp_val, final_clamp_val)
                final_flat_params = torch.nan_to_num(final_flat_params, nan=0.0, posinf=final_clamp_val, neginf=-final_clamp_val)

                processed_params_list.append(final_flat_params.detach())
                processed_weights.append(data['weight'])

            except Exception as e:
                print(f"Error during parameter refinement for client {client_idx}: {e}. Using original filtered params.")
                traceback.print_exc()
                final_clamp_val = 20.0
                final_flat_params = torch.clamp(params_flat_dev, -final_clamp_val, final_clamp_val)
                final_flat_params = torch.nan_to_num(final_flat_params, nan=0.0, posinf=final_clamp_val, neginf=-final_clamp_val)
                processed_params_list.append(final_flat_params.detach())
                processed_weights.append(data['weight'])

        if fm_is_trustworthy:
             print(f"Applied FM refinement to {refinement_count}/{len(accepted_client_data)} accepted clients (Rel < {self.fm_reliability_threshold:.2f}).")

        # --- Stage 7: Weighted Aggregation ---
        if not processed_params_list:
            print("No parameters processed after refinement stage. Global model remains unchanged.")
            self.previous_global_params_flat = current_global_params_flat_cpu.clone()
            return

        aggregated_flat_params = None
        if total_samples_accepted <= 1e-9:
            print("Warning: Total weight zero or negligible. Using simple average.")
            if len(processed_params_list) > 0:
                aggregated_flat_params = torch.mean(torch.stack(processed_params_list), dim=0)
            else: aggregated_flat_params = current_global_params_flat
        else:
            client_weights_tensor = torch.tensor(processed_weights, dtype=torch.float32, device=self.device)
            normalized_weights = client_weights_tensor / total_samples_accepted
            aggregated_flat_params = torch.zeros(self.param_dim, dtype=torch.float32, device=self.device)
            for i, final_flat in enumerate(processed_params_list):
                if final_flat.numel() == self.param_dim:
                    aggregated_flat_params += normalized_weights[i] * final_flat
                else:
                    print(f"Error: Param size mismatch during weighted average for client {accepted_client_data[i]['idx']}. Skipping.")

        # --- Stage 8: Update Global Model ---
        if aggregated_flat_params is None or aggregated_flat_params.numel() != self.param_dim:
            print("Aggregation failed. Global model remains unchanged.")
            self.previous_global_params_flat = current_global_params_flat_cpu.clone()
            return

        try:
            final_clamp_val = 20.0
            aggregated_flat_params = torch.clamp(aggregated_flat_params, -final_clamp_val, final_clamp_val)
            aggregated_flat_params = torch.nan_to_num(aggregated_flat_params, nan=0.0, posinf=final_clamp_val, neginf=-final_clamp_val)

            self._unflatten_params(aggregated_flat_params.cpu(), self.model)
            self.previous_global_params_flat = aggregated_flat_params.cpu().clone()
            print(f"Global model updated using {len(processed_params_list)} clients. Aggregation: Tuned Filtered, Adaptive FM Mix (Trustworthy={fm_is_trustworthy}).")

        except Exception as e:
            print(f"CRITICAL Error unflattening aggregated parameters: {e}. Restoring previous model state.")
            traceback.print_exc()
            if self.previous_global_params_flat is not None:
                 self._unflatten_params(self.previous_global_params_flat, self.model)
            else:
                 self._unflatten_params(current_global_params_flat_cpu, self.model)


    # --- train method (Mostly unchanged structure, calls optimized aggregate_parameters) ---
    def train(self):
        print("\n>>> Optimized Robust FlowMatch Training Starting (CIFAR-100 Tuned) <<<")
        print(">>> Ensure client-side LR scheduling is active! <<<")

        for r_loop in range(1, self.global_rounds + 1):
            start_time = time.time()
            self.r = r_loop

            self.sample_active_clients()
            if not self.active_clients:
                print(f"Round {self.r}: No clients selected. Skipping round.")
                continue

            self.send_models()

            train_acc, train_loss = float('nan'), float('nan')
            try:
                train_acc, train_loss = self.train_clients()
                if isinstance(train_loss, torch.Tensor): train_loss = train_loss.item() if torch.isfinite(train_loss).all() else float('nan')
                elif not isinstance(train_loss, (float, int)) or not np.isfinite(train_loss): train_loss = float('nan')
                if isinstance(train_acc, torch.Tensor): train_acc = train_acc.item() if torch.isfinite(train_acc).all() else float('nan')
                elif not isinstance(train_acc, (float, int)) or not np.isfinite(train_acc): train_acc = float('nan')
            except Exception as e:
                print(f"Error during client training in round {self.r}: {e}")
                traceback.print_exc()

            train_time = time.time() - start_time

            aggregation_start_time = time.time()
            self.aggregate_parameters() # Calls the CIFAR-100 optimized aggregation
            aggregation_time = time.time() - aggregation_start_time

            round_time = time.time() - start_time
            self.train_times.append(train_time)
            self.round_times.append(round_time)

            if self.r % self.eval_gap == 0 or self.r == self.global_rounds:
                print(f"\n--- Evaluating Round {self.r}/{self.global_rounds} ---")
                test_acc, test_loss, test_acc_std = self.evaluate()
                # IMPORTANT: Ensure evaluate_personalized performs fine-tuning (c.train()) before evaluation
                ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized()
                print(f"Round [{self.r}/{self.global_rounds}] Results:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Test Loss (Global): {test_loss:.4f} | Test Acc (Global): {test_acc:.2f}% (Std: {test_acc_std:.2f}%)")
                print(f"  Test Loss (Pers): {ptest_loss:.4f} | Test Acc (Pers): {ptest_acc:.2f}% (Std: {ptest_acc_std:.2f}%)") # This is often the key metric vs FedAvgFT
                print(f"  Timings: Train={train_time:.2f}s | Aggr={aggregation_time:.2f}s | Round={round_time:.2f}s")
                if self.fm_lr_scheduler_rounds:
                     current_fm_base_lr = self.fm_lr_scheduler_rounds.get_last_lr()[0]
                     print(f"  Current Base FM LR: {current_fm_base_lr:.2e}")
                print(f"--- End Evaluation ---")
            else:
                print(f"Round [{self.r}/{self.global_rounds}] Completed. Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Time: {round_time:.2f}s")

        print("\n>>> Optimized Robust FlowMatch Training Finished (CIFAR-100 Tuned) <<<")


    # --- evaluate method (Unchanged from your provided code) ---
    def evaluate(self):
        total_samples = 0
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad(): # Ensure no gradients are computed during evaluation
            for c in self.clients:
                 if c.num_test == 0: continue
                 try:
                     # Temporarily set client model to global model for evaluation
                     # It's safer to pass the global model state_dict if client evaluate supports it
                     # Or deepcopy the model like this
                     original_client_model_state = deepcopy(c.model.state_dict())
                     c.set_model(self.model) # Assumes set_model loads state dict or replaces model
                     acc, loss = c.evaluate() # Client evaluate uses the temporary global model
                     c.model.load_state_dict(original_client_model_state) # Restore original client state
                     c.model.to(c.device) # Ensure restored model is on correct device

                     weight = c.num_test
                     # Ensure acc/loss are scalar floats
                     acc_val = acc.item() if isinstance(acc, torch.Tensor) else float(acc)
                     loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)


                     if np.isfinite(acc_val) and np.isfinite(loss_val):
                         accs.append(acc_val) # Store raw accuracy (0-1)
                         weighted_loss += weight * loss_val
                         weighted_acc += weight * acc_val
                         total_samples += weight
                     else:
                         print(f"Warning: Non-finite acc/loss during global eval for client {c.client_idx}. Skipping.")
                 except Exception as e:
                     print(f"Error evaluating global model on client {c.client_idx}: {e}")
                     traceback.print_exc()

        if total_samples == 0: return 0.0, 0.0, 0.0
        avg_loss = weighted_loss / total_samples
        avg_acc = (weighted_acc / total_samples) * 100.0 # Convert final avg acc to percentage
        std_acc = np.std([a * 100.0 for a in accs]) if accs else 0.0 # Calculate std dev on percentages
        return avg_acc / 100, avg_loss, std_acc / 100


    # --- evaluate_personalized method (Crucial for comparison with FedAvgFT) ---
    # --- Ensure this method correctly performs fine-tuning before evaluation ---
    def evaluate_personalized(self):
        total_samples = 0
        weighted_loss = 0
        weighted_acc = 0
        accs = []

        print("--- Starting Personalized Evaluation (includes client fine-tuning) ---")
        for c in self.clients:
            if c.num_test == 0: continue
            try:
                # 1. Save original client state (if any personalization exists)
                original_client_model_state = deepcopy(c.model.state_dict())

                # 2. Load current global model state into the client model
                global_model_state = deepcopy(self.model.state_dict())
                c.model.load_state_dict(global_model_state)
                c.model.to(c.device)

                # 3. **Fine-tune the client model** using its local training data
                #    Assuming c.train() performs the desired fine-tuning steps.
                #    If c.train() is too long, you might need a separate c.finetune() method.
                # print(f"  Fine-tuning client {c.client_idx}...")
                # You might want to control the number of fine-tuning steps/epochs here
                # e.g., c.finetune(epochs=1) or c.train(local_epochs=1)
                c.train() # Call the client's training method for fine-tuning

                # 4. Evaluate the fine-tuned model on the client's test set
                # print(f"  Evaluating fine-tuned client {c.client_idx}...")
                # Ensure evaluation is done without gradients
                with torch.no_grad():
                    c.model.eval() # Set to eval mode for evaluation
                    acc, loss = c.evaluate()

                # 5. Restore original client state (optional, depends if you reuse client state)
                # c.model.load_state_dict(original_client_model_state)
                # c.model.to(c.device)

                weight = c.num_test
                acc_val = acc.item() if isinstance(acc, torch.Tensor) else float(acc)
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)


                if np.isfinite(acc_val) and np.isfinite(loss_val):
                    accs.append(acc_val) # Store raw accuracy
                    weighted_loss += weight * loss_val
                    weighted_acc += weight * acc_val
                    total_samples += weight
                else:
                    print(f"Warning: Non-finite acc/loss during personalized eval for client {c.client_idx}. Skipping.")
            except Exception as e:
                 print(f"Error during personalized evaluation for client {c.client_idx}: {e}")
                 traceback.print_exc()
                 # Attempt to restore original state even on error
                 try:
                     c.model.load_state_dict(original_client_model_state)
                     c.model.to(c.device)
                 except:
                     pass # Ignore restore errors if initial copy failed


        print("--- Finished Personalized Evaluation ---")
        if total_samples == 0: return 0.0, 0.0, 0.0
        avg_loss = weighted_loss / total_samples
        avg_acc = (weighted_acc / total_samples) * 100.0 # Convert final avg acc to percentage
        std_acc = np.std([a * 100.0 for a in accs]) if accs else 0.0 # Calculate std dev on percentages
        return avg_acc / 100, avg_loss, std_acc / 100