# Updated sections in server_fedavg.py

import time
from copy import deepcopy
import torch
from servers.server_base import Server
# Assuming ClientFedAvg is modified as described above
from clients.client_fedavg import ClientFedAvg
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, TensorDataset
from utils.util import AverageMeter # Assuming this exists
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR # Assuming needed by FM model
import numpy as np
from collections import deque
import traceback
import random
import logging
from tqdm import tqdm # For progress bars, optional

# Configure basic logging (can be overridden in Server init if needed)
# Set default level to INFO. You can change this or configure handlers more elaborately.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Robust Aggregation Helper ---
@torch.no_grad()
def coordinate_wise_trimmed_mean(updates: list[torch.Tensor], trim_percentage: float) -> torch.Tensor:
    """
    Computes the coordinate-wise trimmed mean of parameter updates.
    (Implementation unchanged from previous version)
    """
    if not updates:
        # Return zero tensor with same dtype and device as first update if possible, else default
        return torch.zeros_like(updates[0]) if updates and isinstance(updates[0], torch.Tensor) else torch.tensor(0.0)


    stacked_updates = torch.stack(updates, dim=0) # Shape: (num_clients, param_dim)
    num_clients = stacked_updates.shape[0]
    # param_dim = stacked_updates.shape[1] # Not needed directly

    if num_clients <= 0: # Handle empty list explicitly
        return torch.zeros_like(updates[0]) if updates and isinstance(updates[0], torch.Tensor) else torch.tensor(0.0)
    elif num_clients <= 2: # Not enough clients to trim meaningfully
        return torch.mean(stacked_updates, dim=0)

    k = int(num_clients * trim_percentage)
    # Ensure k is valid for trimming
    if k * 2 >= num_clients:
        # Avoid trimming everything, maybe return median or simple mean
        k = max(0, (num_clients - 1) // 2) # Trim all but the median(s)
        # Re-check k based on potentially reduced num_clients for trimming
        if num_clients - (2*k) <= 0 : # Still trimming everything or invalid k
             if num_clients > 0:
                 # Fallback to mean if trimming isn't possible
                 return torch.mean(stacked_updates, dim=0)
             else:
                 # Should be caught by initial check, but safety fallback
                 return torch.zeros_like(updates[0]) if updates and isinstance(updates[0], torch.Tensor) else torch.tensor(0.0)


    # Sort along the client dimension for each parameter coordinate
    try:
        sorted_updates, _ = torch.sort(stacked_updates, dim=0) # Shape: (num_clients, param_dim)
    except Exception as e:
        logging.error(f"Error during torch.sort in CWTM: {e}")
        # Fallback to simple mean if sorting fails
        return torch.mean(stacked_updates, dim=0)


    # Trim k elements from each end if k > 0
    if k > 0:
        trimmed_updates = sorted_updates[k:-k, :] # Shape: (num_clients - 2*k, param_dim)
    else:
        trimmed_updates = sorted_updates # No trimming if k=0

    # Compute the mean of the remaining elements for each coordinate
    if trimmed_updates.shape[0] > 0: # Ensure there are elements left after trimming
        robust_mean_update = torch.mean(trimmed_updates, dim=0) # Shape: (param_dim,)
    elif stacked_updates.shape[0] > 0: # If trimming removed everything, fallback to mean of original
         robust_mean_update = torch.mean(stacked_updates, dim=0)
    else: # Should not happen if initial checks pass
        robust_mean_update = torch.zeros_like(updates[0]) if updates and isinstance(updates[0], torch.Tensor) else torch.tensor(0.0)


    return robust_mean_update

# --- Flow Matching Model (Conditional) ---
class FlowMatchingModel(nn.Module):
    def __init__(self, param_dim, hidden_dim=2048, rank=256, dropout_rate=0.1, condition_dim=128, fm_corruption_embed_dim=64, fm_reliability_embed_dim=64):
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # --- OPTIMIZATION: Increased condition_dim and corruption embedding size ---
        condition_dim = condition_dim
        corruption_embed_dim = fm_corruption_embed_dim # Increased default
        reliability_embed_dim = fm_reliability_embed_dim # Default
        time_embed_dim = condition_dim - corruption_embed_dim - reliability_embed_dim
        if time_embed_dim <= 0:
            self.logger.warning(f"Condition dim {condition_dim} too small for specified corruption/reliability dims. Adjusting.")
            # Adjust, e.g., prioritize corruption and reliability
            corruption_embed_dim = condition_dim // 3
            reliability_embed_dim = condition_dim // 3
            time_embed_dim = condition_dim - corruption_embed_dim - reliability_embed_dim


        # Embeddings for time and conditions
        self.time_embed = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, time_embed_dim))
        self.reliability_embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, reliability_embed_dim))
        self.corruption_embed = nn.Embedding(2, corruption_embed_dim) # 0: clean, 1: potentially corrupted

        self.input_norm = nn.LayerNorm(param_dim)
        self.proj = nn.Linear(param_dim, rank)
        self.dropout = nn.Dropout(dropout_rate)

        # MLP layers incorporating conditioning
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(rank + condition_dim, hidden_dim), # Combine param features and condition features
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, param_dim) # Output dimension matches parameter dimension
        )
        self.logger.info(f"Initialized Conditional FlowMatchingModel with param_dim={param_dim}, hidden_dim={hidden_dim}, rank={rank}, condition_dim={condition_dim} (T:{time_embed_dim}, R:{reliability_embed_dim}, C:{corruption_embed_dim})")

    def forward(self, x, t, reliability, is_corrupted_flag):
        """
        Args:
            x: Input parameters (client's original upload), shape (batch_size, param_dim)
            t: Time step (e.g., normalized round), shape (batch_size, 1)
            reliability: Reliability score, shape (batch_size, 1)
            is_corrupted_flag: 0 or 1, shape (batch_size,)
        """
        x_norm = self.input_norm(x)
        x_proj = self.proj(x_norm)

        # Generate embeddings
        time_embedding = self.time_embed(t)
        reliability_embedding = self.reliability_embed(reliability)
        corruption_embedding = self.corruption_embed(is_corrupted_flag)

        # Combine embeddings
        condition_embedding = torch.cat([time_embedding, reliability_embedding, corruption_embedding], dim=1)

        # Combine projected features and conditional embeddings
        if x_proj.shape[0] != condition_embedding.shape[0]:
             raise ValueError(f"Batch size mismatch: x_proj ({x_proj.shape[0]}) vs condition_embedding ({condition_embedding.shape[0]})")

        combined_features = torch.cat([x_proj, condition_embedding], dim=1)


        # Pass through MLP
        output = self.mlp(combined_features)
        return output

# --- RobustFlow Server ---
class ServerFedAvg(Server): # Rename the class if you prefer, e.g., ServerRobustFlow
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        # Basic setup
        self.clients = []
        for client_idx in range(self.num_clients):
            try:
                 c = ClientFedAvg(args, client_idx)
                 if not hasattr(c, 'reliability_score'): c.reliability_score = 1.0
                 if not hasattr(c, 'latest_params'): c.latest_params = None
                 # --- OPTIMIZATION: Store if client is potentially corrupted ---
                 c.is_corrupted = args.augmented and (0 <= client_idx < 50) # Assumes first 50 are augmented
                 self.clients.append(c)
            except Exception as e:
                 logging.exception(f"Failed to initialize client {client_idx}: {e}")
                 raise


        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("--- RobustFlow Server Initializing ---")

        # --- RobustFlow Hyperparameters ---
        self.augmented_scenario = args.augmented
        # 暂时不用 _25: 0.15, _50: 0.15, _75: 0.30, _100: 0.30
        self.robust_trim_percentage = getattr(args, 'robust_trim_percentage', 0.15) # Example: Increased trim
        self.norm_percentile_threshold = getattr(args, 'norm_percentile_threshold', 90.0) # Example: Stricter filtering
        self.reliability_ema_alpha = getattr(args, 'reliability_ema_alpha', 0.5)
        # --- OPTIMIZATION: Add separate threshold for corrupted clients ---
        self.corrupted_norm_percentile_threshold = getattr(args, 'corrupted_norm_percentile_threshold', 80.0) # Example: Even stricter
        # --- OPTIMIZATION: Add weight factor for corrupted clients in final avg ---
        # _25: 0.75, _50: 0.75, _75: 0.25, _100: 0.25
        self.corrupted_client_weight_factor = getattr(args, 'corrupted_client_weight_factor', 0.25) # Example: Down-weight corrupted

        # --- FlowMatch Hyperparameters ---
        self.use_flow_match = getattr(args, 'use_flow_match', True)
        self.fm_param_dim = sum(p.numel() for p in self.model.parameters())
        # --- OPTIMIZATION: Pass args to FlowMatchingModel ---
        self.fm_model = FlowMatchingModel(
            self.fm_param_dim,
            hidden_dim=getattr(args, 'fm_hidden_dim', 2048),
            rank=getattr(args, 'fm_rank', 256),
            dropout_rate=getattr(args, 'fm_dropout', 0.1),
            condition_dim=getattr(args, 'fm_condition_dim', 192), # Match FM model init
            fm_corruption_embed_dim=getattr(args, 'fm_corruption_embed_dim', 64),
            fm_reliability_embed_dim=getattr(args, 'fm_reliability_embed_dim', 64)
        ).to(self.device)
        self.fm_optimizer = torch.optim.AdamW(self.fm_model.parameters(), lr=getattr(args, 'fm_lr', 1e-4), weight_decay=1e-6) # Example: Reduced LR
        self.fm_epochs = getattr(args, 'fm_epochs', 10) # Example: Reduced epochs
        self.fm_trust_loss_threshold = getattr(args, 'fm_trust_loss_threshold', 0.1) # Example: Lower trust threshold
        # --- OPTIMIZATION: Use adaptive refinement instead of fixed threshold ---
        self.fm_refine_bottom_percentile = getattr(args, 'fm_refine_bottom_percentile', 20.0) # Example: Refine bottom 20% reliability
        # self.fm_reliability_threshold = getattr(args, 'fm_reliability_threshold', 0.3) # Keep for reference or fallback
        self.mix_alpha_min = getattr(args, 'mix_alpha_min', 0.5)
        self.mix_alpha_max = getattr(args, 'mix_alpha_max', 0.9)
        self.fm_batch_size = getattr(args, 'fm_batch_size', 64)
        fm_buffer_size_default = getattr(args, 'fm_buffer_size', 2000)
        self.param_buffer = deque(maxlen=fm_buffer_size_default)
        self.fm_warmup_rounds = getattr(args, 'fm_warmup_rounds', 30) # Example: Increased warmup
        self.fm_current_loss = float('inf')


        self.logger.info(f"Robust Aggregation: Coordinate-wise Trimmed Mean (Trim: {self.robust_trim_percentage*100}%)")
        self.logger.info(f"Initial Filtering: Norm Percentile Threshold = {self.norm_percentile_threshold} (Clean), {self.corrupted_norm_percentile_threshold} (Corrupted)")
        self.logger.info(f"Final Aggregation Weight Factor (Corrupted Clients): {self.corrupted_client_weight_factor}")
        self.logger.info(f"FlowMatch Enabled: {self.use_flow_match}")
        if self.use_flow_match:
             self.logger.info(f"  - Refinement: Bottom {self.fm_refine_bottom_percentile}% Reliability")
             self.logger.info(f"  - Trust Loss Threshold: {self.fm_trust_loss_threshold}")
             self.logger.info(f"  - Buffer Size: {fm_buffer_size_default}")
             self.logger.info(f"  - Warmup Rounds: {self.fm_warmup_rounds}")


    @torch.no_grad()
    def aggregate_models(self, client_results: list, initial_global_flat_params: torch.Tensor, current_round: int):
        """
        Performs robust aggregation (CWTM), reliability calculation, optional FlowMatch refinement,
        and final weighted averaging. Incorporates differential filtering and weighting.
        """
        if not client_results:
            self.logger.warning("No client results received for aggregation.")
            return

        self.model.eval() # Ensure model is in eval mode
        num_clients_received = len(client_results)
        self.logger.info(f"Starting aggregation for round {current_round} with {num_clients_received} client results.")

        # --- 1. Calculate Parameter Updates and Initial Norm Filtering (Differential) ---
        client_updates_clean = []
        client_updates_corrupted = []
        client_norms_clean = []
        client_norms_corrupted = []
        original_params_map = {} # Store original params for FM buffer/refinement

        if initial_global_flat_params is None or initial_global_flat_params.numel() == 0:
            self.logger.error("Initial global flat parameters are invalid. Cannot calculate updates.")
            return

        for res in client_results:
            client_idx = res['idx']
            if 'params' not in res or not isinstance(res['params'], dict):
                 self.logger.warning(f"Client {client_idx}: Invalid 'params' data received. Skipping.")
                 continue

            # --- OPTIMIZATION: Get corruption status ---
            is_corrupted = self.clients[client_idx].is_corrupted if client_idx < len(self.clients) else False

            client_flat_params = self.get_flat_params(res['params'])
            if client_flat_params is None or client_flat_params.numel() == 0:
                 self.logger.warning(f"Client {client_idx}: Could not flatten parameters. Skipping.")
                 continue

            if client_flat_params.shape != initial_global_flat_params.shape:
                self.logger.warning(f"Client {client_idx}: Parameter dimension mismatch ({client_flat_params.shape} vs {initial_global_flat_params.shape}). Skipping.")
                continue

            client_flat_params = client_flat_params.to(self.device)
            initial_global_flat_params = initial_global_flat_params.to(self.device)
            update = client_flat_params - initial_global_flat_params
            norm = torch.norm(update).item()

            if not np.isfinite(norm):
                 self.logger.warning(f"Client {client_idx}: Infinite norm detected ({norm}). Skipping.")
                 continue

            update_info = {'idx': client_idx, 'update': update, 'norm': norm, 'original_params': client_flat_params, 'is_corrupted': is_corrupted}
            original_params_map[client_idx] = client_flat_params # Store original params

            # Separate updates based on corruption status for differential filtering
            if is_corrupted:
                client_updates_corrupted.append(update_info)
                client_norms_corrupted.append(norm)
            else:
                client_updates_clean.append(update_info)
                client_norms_clean.append(norm)

        # Calculate separate norm thresholds
        norm_thresh_clean = np.percentile(client_norms_clean, self.norm_percentile_threshold) if client_norms_clean else float('inf')
        norm_thresh_corrupted = np.percentile(client_norms_corrupted, self.corrupted_norm_percentile_threshold) if client_norms_corrupted else float('inf')
        self.logger.info(f"Adaptive norm threshold (Clean): {norm_thresh_clean:.4f} ({self.norm_percentile_threshold}th percentile)")
        self.logger.info(f"Adaptive norm threshold (Corrupted): {norm_thresh_corrupted:.4f} ({self.corrupted_norm_percentile_threshold}th percentile)")

        # Apply norm filtering differentially
        filtered_updates_info = []
        accepted_indices_after_norm = []

        for update_info in client_updates_clean:
            thresh = norm_thresh_clean
            if np.isfinite(thresh) and update_info['norm'] <= thresh :
                filtered_updates_info.append(update_info)
                accepted_indices_after_norm.append(update_info['idx'])
            elif not np.isfinite(thresh): # Accept if threshold is infinite
                 filtered_updates_info.append(update_info)
                 accepted_indices_after_norm.append(update_info['idx'])
            else:
                self.logger.info(f"Client {update_info['idx']} (Clean): REJECTED by norm filter (Norm={update_info['norm']:.2f} > Thresh={thresh:.2f})")

        for update_info in client_updates_corrupted:
            thresh = norm_thresh_corrupted
            if np.isfinite(thresh) and update_info['norm'] <= thresh :
                filtered_updates_info.append(update_info)
                accepted_indices_after_norm.append(update_info['idx'])
            elif not np.isfinite(thresh): # Accept if threshold is infinite
                 filtered_updates_info.append(update_info)
                 accepted_indices_after_norm.append(update_info['idx'])
            else:
                self.logger.info(f"Client {update_info['idx']} (Corrupted): REJECTED by norm filter (Norm={update_info['norm']:.2f} > Thresh={thresh:.2f})")


        if not filtered_updates_info:
             self.logger.warning("No client updates passed the norm filter.")
             return

        self.logger.info(f"Accepted {len(filtered_updates_info)}/{num_clients_received} clients after norm filtering.")

        # --- 2. Robust Aggregation (Coordinate-wise Trimmed Mean) ---
        updates_for_cwtm = [info['update'] for info in filtered_updates_info]
        if not updates_for_cwtm:
            self.logger.warning("Update list for CWTM is empty after filtering.")
            return

        robust_mean_update = coordinate_wise_trimmed_mean(updates_for_cwtm, self.robust_trim_percentage)
        if robust_mean_update is None or robust_mean_update.numel() == 0:
             self.logger.error("Robust aggregation (CWTM) failed or returned empty tensor.")
             return

        robust_mean_update_norm = torch.norm(robust_mean_update).item()
        self.logger.info(f"Robust Mean Update (CWTM, Trim={self.robust_trim_percentage*100}%) Norm: {robust_mean_update_norm:.4f}")

        if initial_global_flat_params.shape != robust_mean_update.shape:
             self.logger.error(f"Dimension mismatch before calculating robust aggregate: Initial ({initial_global_flat_params.shape}) vs Update ({robust_mean_update.shape})")
             return

        robust_aggregated_flat_params = initial_global_flat_params + robust_mean_update

        # --- 3. Post-Aggregation Reliability Calculation ---
        client_final_params = {} # Store params to be averaged at the end
        reliability_scores_updated = {} # Track scores for logging/FM buffer

        # (Reliability calculation logic remains similar, based on similarity to robust_mean_update)
        if robust_mean_update_norm < 1e-9:
             self.logger.warning("Robust mean update norm is near zero. Skipping reliability calculation, using previous scores.")
             for info in filtered_updates_info:
                  client_idx = info['idx']
                  if client_idx < 0 or client_idx >= len(self.clients): continue
                  client = self.clients[client_idx]
                  reliability_scores_updated[client_idx] = getattr(client, 'reliability_score', 1.0)
                  client_final_params[client_idx] = info['original_params']
        else:
            for info in filtered_updates_info:
                client_idx = info['idx']
                if client_idx < 0 or client_idx >= len(self.clients): continue

                client_update = info['update']
                client_update_norm = info['norm']

                if client_update_norm < 1e-9:
                    similarity = 0.0
                else:
                    try:
                        if client_update.shape != robust_mean_update.shape:
                             self.logger.warning(f"Client {client_idx}: Shape mismatch for reliability dot product ({client_update.shape} vs {robust_mean_update.shape}). Skipping reliability update.")
                             similarity = 0.0
                        else:
                             dot_product = torch.dot(client_update, robust_mean_update)
                             similarity = dot_product / (client_update_norm * robust_mean_update_norm)
                             similarity = similarity.item()
                    except Exception as e:
                         self.logger.error(f"Client {client_idx}: Error calculating dot product/similarity: {e}")
                         similarity = 0.0

                similarity = max(-1.0, min(1.0, similarity))

                client = self.clients[client_idx]
                current_reliability = getattr(client, 'reliability_score', 1.0)
                # --- OPTIMIZATION: Potentially cap reliability for corrupted clients ---
                # if client.is_corrupted: similarity = min(similarity, 0.8) # Example cap

                client.reliability_score = self.reliability_ema_alpha * current_reliability + (1 - self.reliability_ema_alpha) * similarity
                reliability_scores_updated[client_idx] = client.reliability_score
                self.logger.debug(f"Client {client_idx} {'(C)' if client.is_corrupted else '(N)'}: Update Sim={similarity:.3f}, New Reliability={client.reliability_score:.3f}")

                client_final_params[client_idx] = info['original_params']


        # --- 4. Conditional FlowMatch Refinement (Adaptive) ---
        fm_applied_count = 0
        fm_loss_is_finite = np.isfinite(self.fm_current_loss)
        fm_trustworthy = self.use_flow_match and \
                         (current_round > self.fm_warmup_rounds) and \
                         fm_loss_is_finite and \
                         (self.fm_current_loss < self.fm_trust_loss_threshold)

        if fm_trustworthy:
            self.fm_model.eval() # Set FM model to eval mode

            # --- OPTIMIZATION: Adaptive refinement based on percentile ---
            reliabilities = [(idx, reliability_scores_updated.get(idx, 0.0)) for idx in accepted_indices_after_norm]
            reliabilities.sort(key=lambda item: item[1]) # Sort by reliability, ascending
            num_to_refine = int(len(reliabilities) * (self.fm_refine_bottom_percentile / 100.0))
            clients_to_refine_indices = {item[0] for item in reliabilities[:num_to_refine]}

            self.logger.info(f"Applying FlowMatch refinement to bottom {self.fm_refine_bottom_percentile}% ({num_to_refine} clients).")

            for client_idx in accepted_indices_after_norm:
                if client_idx not in clients_to_refine_indices:
                    continue # Skip if not in the bottom percentile

                # Ensure client index is valid and original params exist
                if client_idx < 0 or client_idx >= len(self.clients) or client_idx not in original_params_map:
                    continue

                client = self.clients[client_idx]
                current_reliability = reliability_scores_updated.get(client_idx, 0.0) # Use the score from this round
                original_params = original_params_map[client_idx]

                # Prepare conditional inputs
                params_input = original_params.unsqueeze(0).to(self.device)
                time_input = torch.tensor([[current_round / self.global_rounds]], device=self.device, dtype=torch.float32)
                reliability_input = torch.tensor([[current_reliability]], device=self.device, dtype=torch.float32)
                is_corrupted_flag_val = 1 if client.is_corrupted else 0
                is_corrupted_flag = torch.tensor([is_corrupted_flag_val], dtype=torch.long, device=self.device)

                try:
                    predicted_target_params = self.fm_model(params_input, time_input, reliability_input, is_corrupted_flag)
                    predicted_target_params = predicted_target_params.squeeze(0).detach()
                except Exception as e:
                    self.logger.error(f"Client {client_idx}: Error during FM model inference: {e}")
                    continue

                # Calculate mixing alpha (logic unchanged, but based on current_reliability)
                if current_reliability >= 1.0:
                    mix_alpha = self.mix_alpha_max
                # --- ADJUSTMENT: Base scaling on the percentile threshold maybe? Or keep simple ---
                # For simplicity, keep the previous scaling logic based on a hypothetical fixed threshold
                # A more complex adaptive alpha could be used here.
                # Let's use a fixed threshold for alpha calculation for now, e.g., 0.5
                hypothetical_alpha_threshold = 0.5
                if current_reliability < hypothetical_alpha_threshold:
                    mix_alpha = self.mix_alpha_min
                else:
                    denom = (1.0 - hypothetical_alpha_threshold)
                    if denom < 1e-6: scale = 1.0
                    else: scale = (current_reliability - hypothetical_alpha_threshold) / denom
                    mix_alpha = self.mix_alpha_min + (self.mix_alpha_max - self.mix_alpha_min) * scale
                mix_alpha = max(self.mix_alpha_min, min(self.mix_alpha_max, mix_alpha))

                # Mix original parameters and FM prediction
                if original_params.shape != predicted_target_params.shape:
                    self.logger.error(f"Client {client_idx}: Shape mismatch for mixing: Original ({original_params.shape}) vs Predicted ({predicted_target_params.shape}). Skipping refinement.")
                    continue

                refined_params = mix_alpha * original_params + (1 - mix_alpha) * predicted_target_params.to(original_params.device)
                client_final_params[client_idx] = refined_params # Update the parameters to be averaged
                fm_applied_count += 1
                self.logger.debug(f"Client {client_idx} {'(C)' if client.is_corrupted else '(N)'}: Refined with Mix Alpha={mix_alpha:.3f} (Reliability={current_reliability:.3f})")

            if fm_applied_count > 0:
                 self.logger.info(f"Applied FlowMatch refinement to {fm_applied_count} clients.")
        elif self.use_flow_match and current_round > self.fm_warmup_rounds:
             fm_loss_str = f"{self.fm_current_loss:.4f}" if fm_loss_is_finite else "inf"
             self.logger.info(f"FlowMatch refinement skipped (FM Loss {fm_loss_str} {'<' if not fm_loss_is_finite or self.fm_current_loss >= self.fm_trust_loss_threshold else '>='} Trust Threshold {self.fm_trust_loss_threshold:.4f})")


        # --- 5. Final Weighted Aggregation (Differential Weighting) ---
        total_weight = 0.0
        final_aggregated_flat_params = torch.zeros_like(initial_global_flat_params, device=initial_global_flat_params.device, dtype=initial_global_flat_params.dtype)
        use_reliability_weights = True # Make this a hyperparameter if needed

        if not client_final_params:
             self.logger.warning("No client parameters available for final aggregation. Global model not updated.")
             final_aggregated_flat_params = initial_global_flat_params
        else:
            num_aggregated = 0
            for client_idx, final_params in client_final_params.items():
                if client_idx < 0 or client_idx >= len(self.clients): continue
                client = self.clients[client_idx]

                weight = getattr(client, 'num_train', 1.0)
                if weight <= 0: weight = 1.0

                if use_reliability_weights:
                     client_reliability = reliability_scores_updated.get(client_idx, 0.0) # Use updated score
                     reliability_weight_factor = max(0.0, client_reliability)
                     weight *= reliability_weight_factor

                # --- OPTIMIZATION: Apply weight factor for corrupted clients ---
                if client.is_corrupted:
                    weight *= self.corrupted_client_weight_factor
                    self.logger.debug(f"Client {client_idx} (C): Applying weight factor {self.corrupted_client_weight_factor}. Original weight contribution: {getattr(client, 'num_train', 1.0):.2f}, Reliability: {reliability_scores_updated.get(client_idx, 0.0):.3f}, Final Weight: {weight:.3f}")


                if weight < 1e-9: continue

                if not isinstance(final_params, torch.Tensor):
                     self.logger.warning(f"Client {client_idx}: final_params is not a tensor (type: {type(final_params)}). Skipping.")
                     continue
                if final_params.shape != final_aggregated_flat_params.shape:
                     self.logger.warning(f"Client {client_idx}: Shape mismatch in final aggregation ({final_params.shape} vs {final_aggregated_flat_params.shape}). Skipping.")
                     continue

                try:
                    final_aggregated_flat_params += weight * final_params.to(final_aggregated_flat_params.device, dtype=final_aggregated_flat_params.dtype)
                    total_weight += weight
                    num_aggregated += 1
                except RuntimeError as e:
                     self.logger.error(f"Client {client_idx}: Error during weighted sum in final aggregation: {e}")
                     continue

            if total_weight > 1e-9:
                final_aggregated_flat_params /= total_weight
                self.logger.info(f"Aggregated {num_aggregated} clients. Final Total Weight: {total_weight:.2f} (Method: {'Reliability-Weighted' if use_reliability_weights else 'Train Samples'} w/ Corrupted Factor)")
            else:
                self.logger.warning("Total aggregation weight is near zero. Global model not updated.")
                final_aggregated_flat_params = initial_global_flat_params


        # Update global model
        if total_weight > 1e-9 or not client_final_params:
             self.set_flat_params(self.model, final_aggregated_flat_params)
             self.logger.debug("Global model updated with aggregated parameters.")
        else:
             self.logger.warning("Aggregation resulted in zero weight, global model remains unchanged from start of round.")


        # --- 6. Update FlowMatch Buffer ---
        # (Buffer update logic remains similar, uses reliability_scores_updated and client.is_corrupted)
        can_buffer = self.use_flow_match and \
                     (current_round >= self.fm_warmup_rounds // 2) and \
                     (self.param_buffer.maxlen is None or len(self.param_buffer) < self.param_buffer.maxlen)

        if can_buffer:
            added_to_buffer = 0
            time_step = torch.tensor([current_round / self.global_rounds], dtype=torch.float32)
            if robust_aggregated_flat_params is None or robust_aggregated_flat_params.numel() == 0:
                self.logger.warning("Skipping FM buffer update as robust aggregated params are invalid.")
            else:
                target_params_cpu = robust_aggregated_flat_params.cpu()

                for client_idx in accepted_indices_after_norm:
                     if client_idx not in original_params_map: continue
                     if client_idx < 0 or client_idx >= len(self.clients): continue
                     client = self.clients[client_idx]

                     original_params = original_params_map[client_idx].cpu()
                     reliability = reliability_scores_updated.get(client_idx, 0.0)
                     is_corrupted_flag_val = 1 if client.is_corrupted else 0

                     if self.param_buffer.maxlen is None or len(self.param_buffer) < self.param_buffer.maxlen:
                         self.param_buffer.append((original_params, target_params_cpu.clone(),
                                                     torch.tensor([reliability], dtype=torch.float32),
                                                     torch.tensor(is_corrupted_flag_val, dtype=torch.long),
                                                     time_step.cpu().clone() ))
                         added_to_buffer += 1
                     else:
                         self.logger.warning("FM Buffer is full. Cannot add more entries.")
                         break

            if added_to_buffer > 0:
                self.logger.debug(f"Added {added_to_buffer} entries to FlowMatch buffer. Size: {len(self.param_buffer)}")

    def sample_active_clients(self):
        """ Samples clients for the current round. """
        num_available_clients = len(self.clients)
        if num_available_clients == 0:
             self.logger.error("No clients available for sampling.")
             self.active_clients_indices = []
             self.active_clients = []
             return

        num_sampled = max(1, min(num_available_clients, int(self.sampling_prob * num_available_clients)))
        try:
            self.active_clients_indices = sorted(random.sample(range(num_available_clients), num_sampled))
            self.active_clients = [self.clients[i] for i in self.active_clients_indices]
            self.logger.debug(f"Sampled {len(self.active_clients)} clients: {self.active_clients_indices}")
        except ValueError as e:
             self.logger.error(f"Error sampling clients (requested {num_sampled} from {num_available_clients}): {e}")
             self.active_clients_indices = []
             self.active_clients = []


    # --- Main Training Loop ---
    def train(self):
        """ Main server training loop implementing RobustFlow. """
        self.logger.info("Starting RobustFlow Training...")
        # Ensure model is on the correct device before getting initial params
        self.model.to(self.device)
        initial_global_flat_params = self.get_flat_params(self.model.state_dict()).clone().detach() # Initial state


        for r in range(1, self.global_rounds + 1):
            self.logger.info(f"\n--- Round {r}/{self.global_rounds} ---")
            start_time = time.time()

            # Store initial model state for update calculation
            # Ensure model is on correct device before getting state dict
            self.model.to(self.device)
            initial_model_state = deepcopy(self.model.state_dict())
            initial_global_flat_params = self.get_flat_params(initial_model_state).clone().detach()
            # Handle case where flattening might fail
            if initial_global_flat_params.numel() == 0 and self.fm_param_dim > 0 :
                 self.logger.error(f"Round {r}: Failed to get initial global flat parameters. Aborting round.")
                 continue # Skip to next round


            # 1. Sample and Send Models
            # Adjust sampling probability if needed (e.g., for final round)
            # if r == self.global_rounds: self.sampling_prob = 1.0
            self.sample_active_clients()
            self.send_models() # Send current global model

            # 2. Train Clients
            # Handle potential errors from train_clients
            try:
                avg_train_acc, avg_train_loss, client_results = self.train_clients()
            except Exception as train_err:
                 self.logger.exception(f"Round {r}: Error during client training phase: {train_err}")
                 continue # Skip aggregation if client training failed broadly


            # 3. Aggregate Models (RobustFlow Logic)
            # Handle potential errors during aggregation
            try:
                 self.aggregate_models(client_results, initial_global_flat_params, r)
            except Exception as agg_err:
                 self.logger.exception(f"Round {r}: Error during model aggregation: {agg_err}")
                 # Decide how to proceed: skip FM training? revert model? continue?
                 # For now, let's continue to FM training but log the error.
                 pass # Continue execution


            # 4. Train FlowMatch Model (using updated buffer)
            try:
                 self.train_fm_model(r)
            except Exception as fm_train_err:
                 self.logger.exception(f"Round {r}: Error during FM model training: {fm_train_err}")
                 # Continue the round even if FM training fails


            round_time = time.time() - start_time
            self.round_times.append(round_time)
            self.logger.info(f"Round {r} completed in {round_time:.2f} seconds.")

            # 5. Evaluation (Periodically)
            if r % self.eval_gap == 0 or r == self.global_rounds:
                self.logger.info(f"\n--- Evaluating Round {r}/{self.global_rounds} ---")
                # We care most about personalized performance vs FedAvgFT
                try:
                    ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized() # Simulates FedAvgFT
                except Exception as p_eval_err:
                    self.logger.exception(f"Round {r}: Error during personalized evaluation: {p_eval_err}")
                    ptest_acc, ptest_loss, ptest_acc_std = 0.0, float('inf'), 0.0

                # Also evaluate global model performance
                try:
                    test_acc, test_loss, test_acc_std = self.evaluate()
                except Exception as g_eval_err:
                    self.logger.exception(f"Round {r}: Error during global evaluation: {g_eval_err}")
                    test_acc, test_loss, test_acc_std = 0.0, float('inf'), 0.0

                self.logger.info(f"Round [{r}/{self.global_rounds}] Results:")
                self.logger.info(f"  Train Loss (Avg): {avg_train_loss:.4f} | Train Acc (Avg): {avg_train_acc:.2f}%") # Log avg train acc/loss
                self.logger.info(f"  Test Loss (Global): {test_loss:.4f} | Test Acc (Global): {test_acc:.2f}% (Std: {test_acc_std:.2f}%)")
                self.logger.info(f"  Personalized Test Loss: {ptest_loss:.4f} | Personalized Test Acc: {ptest_acc:.2f}% (Std: {ptest_acc_std:.2f}%)")

                # Log results to history or files if needed
                if hasattr(self, 'results') and isinstance(self.results, dict):
                    self.results.setdefault('global_acc', []).append(test_acc)
                    self.results.setdefault('global_loss', []).append(test_loss)
                    self.results.setdefault('p_global_acc', []).append(ptest_acc) # Personalized accuracy
                    self.results.setdefault('p_global_loss', []).append(ptest_loss)
                    self.results.setdefault('train_acc', []).append(avg_train_acc) # Log train accuracy
                    self.results.setdefault('train_loss', []).append(avg_train_loss) # Log train loss

                if hasattr(self, 'log_results') and callable(self.log_results):
                    self.log_results(r) # Assuming you have this method

        self.logger.info("RobustFlow Training Finished.")
        if hasattr(self, 'save_results') and callable(self.save_results):
             self.save_results() # Assuming you have this method
        if hasattr(self, 'save_model') and callable(self.save_model):
             self.save_model() # Assuming you have this method

    @torch.no_grad()
    def get_flat_params(self, model_state):
        """ Helper to get flattened parameters from a state dict. """
        if not model_state: return torch.tensor([]) # Handle empty state dict
        try:
             # Ensure all values are tensors before concatenation
             valid_tensors = [p.data.view(-1) for p in model_state.values() if isinstance(p, torch.Tensor)]
             if not valid_tensors: return torch.tensor([]) # Handle case where state dict has no tensors
             return torch.cat(valid_tensors)
        except Exception as e:
            self.logger.error(f"Error flattening parameters: {e}. State dict keys: {model_state.keys()}")
            # Attempt to debug which parameter caused the issue if possible
            for k, v in model_state.items():
                if not isinstance(v, torch.Tensor):
                    self.logger.error(f" Non-tensor value found in state_dict for key '{k}': {type(v)}")
            return torch.tensor([]) # Return empty tensor on error


    @torch.no_grad()
    def set_flat_params(self, model, flat_params):
        """ Helper to set model parameters from a flattened vector. """
        if flat_params is None or flat_params.numel() == 0:
             self.logger.warning("Attempted to set flat params with None or empty tensor.")
             return
        offset = 0
        for param in model.parameters():
            numel = param.numel()
            if offset + numel > flat_params.numel():
                 self.logger.error(f"Parameter size mismatch: flat_params ({flat_params.numel()}) too small for model ({offset+numel} needed).")
                 return # Stop to prevent index out of bounds
            try:
                 param.data.copy_(flat_params[offset:offset + numel].view_as(param.data))
            except RuntimeError as e:
                 self.logger.error(f"Error copying flat params to model param (shape {param.data.shape}): {e}")
                 self.logger.error(f" Slice shape: {flat_params[offset:offset + numel].shape}, Target shape: {param.data.shape}")
                 return # Stop on error
            offset += numel
        if offset != flat_params.numel():
             self.logger.warning(f"Parameter size mismatch: flat_params ({flat_params.numel()}) larger than model ({offset}). Extra data ignored.")


    def train_clients(self):
        """ Coordinates the training process on active clients. """
        if not self.active_clients:
             self.logger.warning("No active clients to train.")
             return 0.0, 0.0, []

        total_loss = AverageMeter()
        total_acc = AverageMeter()
        client_params_list = [] # Store tuples (client_idx, params_state_dict)

        active_client_indices_set = set(self.active_clients_indices) # For quick lookup
        # Use tqdm for progress bar if desired
        # for client_idx in tqdm(self.active_clients_indices, desc="Client Training", leave=False):
        for client_idx in self.active_clients_indices:
            if client_idx not in active_client_indices_set: continue # Should not happen with current logic, but safe check

            # Ensure client index is valid before accessing self.clients
            if client_idx < 0 or client_idx >= len(self.clients):
                self.logger.error(f"Invalid client index {client_idx} encountered during training.")
                continue

            c = self.clients[client_idx]
            try:
                # Client training step
                # Ensure client.train() returns acc, loss and client.latest_params is updated
                # Pass finetune_mode=False explicitly if needed
                acc, loss = c.train() # Assuming c.train() updates c.latest_params internally


                # Verify that latest_params was updated and is a state_dict
                if not hasattr(c, 'latest_params') or c.latest_params is None:
                     self.logger.warning(f"Client {client_idx} did not update latest_params attribute. Skipping.")
                     continue
                if not isinstance(c.latest_params, dict): # Check if it's a dictionary (like state_dict)
                     self.logger.warning(f"Client {client_idx} latest_params is not a dictionary (type: {type(c.latest_params)}). Skipping.")
                     continue


                client_params_list.append({'idx': client_idx, 'params': c.latest_params})

                # Logging training progress (optional per-client logging)
                if acc is not None and loss is not None:
                     acc_val = acc.item() if isinstance(acc, torch.Tensor) else acc
                     loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                     if np.isfinite(acc_val) and np.isfinite(loss_val):
                         # Ensure client has num_train attribute
                         client_weight = getattr(c, 'num_train', 1) # Default to 1 if not present
                         if client_weight <= 0: client_weight = 1 # Avoid zero weight

                         total_acc.update(acc_val, client_weight)
                         total_loss.update(loss_val, client_weight)
                     else:
                         self.logger.warning(f"Client {client_idx} returned non-finite acc/loss during training.")
                else:
                    self.logger.warning(f"Client {client_idx} did not return valid acc/loss.")

            except Exception as e:
                self.logger.exception(f"Error during training client {client_idx}: {e}\n{traceback.format_exc()}")
                # Optionally remove the client from the list if it failed catastrophically
                client_params_list = [p for p in client_params_list if p['idx'] != client_idx]


        self.logger.info(f"Clients Training Avg Acc: {total_acc.avg:.2f}%, Avg Loss: {total_loss.avg:.4f}")
        return total_acc.avg, total_loss.avg, client_params_list


    # # Make sure evaluate_personalized does some local training steps
    def evaluate_personalized(self):
        """ Simulates FedAvgFT: Load global, finetune locally, evaluate. """
        total_samples = sum(c.num_test for c in self.clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        finetune_epochs = getattr(self.args, 'finetune_epochs', 1) # Add arg for finetune epochs (e.g., 1 or 5)

        for c in self.clients:
            # Store original state ONLY if needed for comparison later (deepcopy is expensive)
            # old_model_state = deepcopy(c.model.state_dict()) # Less ideal

            # Load global model state
            c.set_model(self.model) # Load current global model parameters

            # Fine-tune locally
            self.logger.debug(f"Personalized Eval: Fine-tuning client {c.client_idx} for {finetune_epochs} epochs...")
            # --- Ensure client train method can run for specific epochs ---
            # Need to modify client.train() or have a separate finetune method
            # For simplicity, let's assume client.train() uses self.local_epochs
            # We temporarily set it for finetuning
            original_local_epochs = c.local_epochs
            c.local_epochs = finetune_epochs
            try:
                _, _ = c.train() # Run fine-tuning steps
            except Exception as ft_err:
                 self.logger.error(f"Error fine-tuning client {c.client_idx} for personalized eval: {ft_err}")
            finally:
                 c.local_epochs = original_local_epochs # Restore original epochs

            # Evaluate the fine-tuned model
            try:
                acc, loss = c.evaluate()
                accs.append(acc)
                weighted_loss += (c.num_test / total_samples) * loss.detach()
                weighted_acc += (c.num_test / total_samples) * acc
            except Exception as eval_err:
                 self.logger.error(f"Error evaluating client {c.client_idx} after fine-tuning: {eval_err}")
                 accs.append(torch.tensor(0.0)) # Append 0 accuracy on error


            # Restore original model state if you need the client's previous state
            # c.model.load_state_dict(old_model_state) # Less ideal

        std = torch.std(torch.stack(accs)) if accs else torch.tensor(0.0)
        return weighted_acc, weighted_loss, std

    def train_fm_model(self, current_round):
        """ Trains the conditional Flow Matching model using the buffer. """
        # Check conditions for training
        should_train = self.use_flow_match and \
                       current_round >= self.fm_warmup_rounds and \
                       len(self.param_buffer) >= self.fm_batch_size


        if not should_train:
            if self.use_flow_match and current_round >= self.fm_warmup_rounds:
                 self.logger.info(f"Skipping FM training (Buffer size {len(self.param_buffer)} < Batch size {self.fm_batch_size})")
            self.fm_current_loss = float('inf') # Reset loss if not training
            return

        self.logger.info(f"--- Training FlowMatch Model (Round {current_round}) ---")
        self.fm_model.train()
        fm_loss_meter = AverageMeter()

        # Create DataLoader from buffer for batching
        # Convert deque to list for DataLoader compatibility only if needed (DataLoader can handle deques)
        # buffer_list = list(self.param_buffer) # Not strictly necessary
        try:
             # Prepare data for TensorDataset - ensure all components are tensors
             tensors = []
             valid_buffer = True
             if self.param_buffer:
                 # Check types and shapes of the first element as a sample
                 first_item = self.param_buffer[0]
                 if not all(isinstance(t, torch.Tensor) for t in first_item):
                      self.logger.error("Items in FM buffer are not all tensors.")
                      valid_buffer = False
                 else:
                      ref_shapes = [t.shape for t in first_item]

                 if valid_buffer:
                     all_items = list(self.param_buffer) # Convert once for stacking
                     try:
                         tensors = [torch.stack([item[i] for item in all_items]) for i in range(len(first_item))]
                     except Exception as stack_err:
                          self.logger.error(f"Error stacking buffer items: {stack_err}")
                          # Attempt to diagnose shape issues
                          for i in range(len(first_item)):
                              shapes = [item[i].shape for item in all_items]
                              if len(set(shapes)) > 1:
                                   self.logger.error(f" Inconsistent shapes found for buffer item index {i}: {set(shapes)}")
                          valid_buffer = False


             if not valid_buffer:
                 self.logger.error("Cannot create FM DataLoader due to invalid buffer content.")
                 self.fm_current_loss = float('inf')
                 return

             dataset = TensorDataset(*tensors) # Pass unpacked list of stacked tensors
             data_loader = DataLoader(dataset, batch_size=self.fm_batch_size, shuffle=True, drop_last=True) # drop_last might help with shape consistency if batch size doesn't divide buffer size
        except Exception as e:
             self.logger.exception(f"Error creating FM DataLoader: {e}")
             self.fm_current_loss = float('inf') # Set loss to indicate failure
             return


        # Simple scheduler example (optional)
        # scheduler = CosineAnnealingLR(self.fm_optimizer, T_max=self.fm_epochs * len(data_loader))

        for epoch in range(self.fm_epochs):
            epoch_loss = 0.0
            num_batches = 0
            # Use tqdm for inner loop progress if desired
            # for batch in tqdm(data_loader, desc=f"FM Epoch {epoch+1}/{self.fm_epochs}", leave=False):
            for i, batch in enumerate(data_loader):
                try:
                    # Unpack batch and move to device
                    original_params, target_params, reliability, is_corrupted_flag, time_step = [d.to(self.device) for d in batch]

                    self.fm_optimizer.zero_grad()

                    # Predict the target parameters
                    predicted_target = self.fm_model(original_params, time_step, reliability, is_corrupted_flag)

                    # Loss: Mean Squared Error between predicted target and actual target (robust aggregate)
                    loss = F.mse_loss(predicted_target, target_params)

                    # Check for NaN loss
                    if torch.isnan(loss):
                         self.logger.warning(f"NaN loss detected during FM training epoch {epoch+1}, batch {i}. Skipping batch.")
                         continue # Skip backprop and optimizer step for this batch


                    loss.backward()
                    # Gradient clipping (optional but recommended)
                    torch.nn.utils.clip_grad_norm_(self.fm_model.parameters(), max_norm=1.0)
                    self.fm_optimizer.step()
                    # scheduler.step() # Uncomment if using scheduler

                    fm_loss_meter.update(loss.item(), original_params.size(0))
                    epoch_loss += loss.item()
                    num_batches += 1
                except Exception as batch_err:
                    self.logger.exception(f"Error during FM training epoch {epoch+1}, batch {i}: {batch_err}")
                    # Optionally continue to next batch or break epoch
                    continue


            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            self.logger.debug(f"FM Epoch {epoch+1}/{self.fm_epochs}, Avg Batch Loss: {avg_epoch_loss:.4f}")


        self.fm_current_loss = fm_loss_meter.avg if fm_loss_meter.count > 0 else float('inf') # Use avg only if batches were processed
        # Handle case where fm_current_loss might still be inf (e.g., all batches failed)
        if not np.isfinite(self.fm_current_loss): self.fm_current_loss = float('inf')


        self.logger.info(f"FlowMatch Model Training Complete. Final Avg Loss: {self.fm_current_loss:.4f}")
        self.fm_model.eval() # Set back to eval mode


    def evaluate(self):
        total_samples = sum(c.num_test for c in self.clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        for c in self.clients:
            old_model = deepcopy(c.model)
            c.model = deepcopy(self.model)
            acc, loss = c.evaluate()
            accs.append(acc)
            weighted_loss += (c.num_test / total_samples) * loss.detach()
            weighted_acc += (c.num_test / total_samples) * acc
            c.model = old_model
        std = torch.std(torch.stack(accs))
        return weighted_acc, weighted_loss, std
