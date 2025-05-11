import time
from copy import deepcopy
import torch
import numpy as np
from statsmodels.stats.correlation_tools import cov_nearest
from servers.server_base import Server
from clients.client_fedfda import ClientFedFDA
from torch.distributions.multivariate_normal import MultivariateNormal


import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, TensorDataset
from utils.util import AverageMeter
from torch.nn.utils.spectral_norm import spectral_norm

class FlowMatchingModel(nn.Module):
    def __init__(self, param_dim, hidden_dim=1024, rank=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64)
        )
        
        # 添加输入归一化层
        self.input_norm = nn.LayerNorm(param_dim)
        self.low_rank_proj = nn.Linear(param_dim, rank)  # 降维
        # 更稳定的网络结构
        self.main_net = nn.Sequential(
            nn.Linear(rank + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, rank)
        )
        self.inv_proj = nn.Linear(rank, param_dim)  # 升维
        # 初始化参数
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def compute_flow_loss(self, real_params, noise_params, t):
        target_flow = real_params - noise_params # Common target for velocity field v(x_t, t)
        
        # 添加输入值裁剪
        noise_params = torch.clamp(noise_params, -5.0, 5.0)
        
        pred_flow = self(noise_params, t)
        # Use standard MSE loss for flow matching
        loss = F.mse_loss(pred_flow, target_flow) 
        return loss
    
    def forward(self, z, t):
        """ 
        输入维度修正：
        z: [batch_size, param_dim] 
        t: [batch_size, 1]
        """
        # 确保输入维度正确
        if z.dim() == 1:
            z = z.unsqueeze(0)  # [D] -> [1, D]
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B] -> [B, 1]
        
        t_emb = self.time_embed(t)  # [B, 64]
        z = self.low_rank_proj(z)
        x = torch.cat([z, t_emb], dim=-1)  # 确保在最后一维拼接
        return self.inv_proj(self.main_net(x))

    @torch.no_grad()
    def generate(self, init_params, num_steps=100):
        z = init_params.clone()
        
        # 改进的采样器
        dt = 1.0 / num_steps
        noise_decay = 0.98
        
        for step in range(num_steps):
            current_noise = (0.5 * (noise_decay ** step)) * torch.randn_like(z)
            t = torch.ones(z.size(0), 1).to(z.device) * (1.0 - step/num_steps)
            
            # 分阶段预测 - Removed output_scale multiplication
            pred = self(z + current_noise, t) # * self.output_scale 
            z = z + pred * dt
            
            # 强约束
            z = torch.clamp(z, -5.0, 5.0)
            z = torch.nan_to_num(z, nan=0.0)
        
        return z


class ServerFedFDA(Server):
    def __init__(self, args):
        super().__init__(args)
        self.clients = [
            ClientFedFDA(args, i) for i in range(self.num_clients)
        ]
        self.global_means = torch.Tensor(torch.rand([self.num_classes, self.D]))
        self.global_covariance = torch.Tensor(torch.eye(self.D))
        self.global_priors = torch.ones(self.num_classes) / self.num_classes
        self.r = 0
        self.flow_model = None
        self.param_buffer = []  # 参数历史缓冲区
        self.init_flow_model()
        self.args = args

    def init_flow_model(self):
        # 获取模型参数维度
        with torch.no_grad():
            # Create a dummy model on CPU first to avoid potential CUDA errors if device is not ready
            dummy_model = copy.deepcopy(self.model).to('cpu') 
            # Determine input shape dynamically or ensure self.model has it
            if not hasattr(self.model, 'input_shape'):
                 # Assuming CIFAR10 default if not specified
                 self.model.input_shape = (3, 32, 32) 
                 print("Warning: Model input_shape not found, assuming (3, 32, 32) for CIFAR10.")
            dummy_input = torch.randn(1, *self.model.input_shape)
            _ = dummy_model(dummy_input) # Dry run to initialize potentially lazy modules
            param_dim = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad) # Sum only trainable parameters
            del dummy_model # Free memory
        
        print(f"Initializing FlowMatchingModel with param_dim={param_dim}")
        self.flow_model = FlowMatchingModel(param_dim).to(self.device)
        # Consider adjusting LR, weight decay for AdamW
        self.optimizer = torch.optim.AdamW(self.flow_model.parameters(), lr=1e-4, weight_decay=1e-5) 
        self.param_dim = param_dim # Store param_dim for later use

    def train(self):
        for r in range(1, self.global_rounds+1):
            start_time = time.time()
            self.r = r
            if r == (self.global_rounds): # full participation on last round
                self.sampling_prob = 1.0
            self.sample_active_clients()
            # Send the *current* global model (potentially updated by previous round's aggregation)
            self.send_models() 

            # train clients
            train_acc, train_loss = self.train_clients()
            train_time = time.time() - start_time

            # aggregate parameters using the collected client models
            self.aggregate_parameters() # This now updates self.model

            round_time = time.time() - start_time
            self.train_times.append(train_time)
            self.round_times.append(round_time)

            # logging
            if r % self.eval_gap == 0 or r == self.global_rounds:
                ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized()  
                test_acc, test_loss, test_acc_std = self.evaluate() 
                print(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Test Loss [{test_loss:.4f}|{ptest_loss:.4f}]\t Test Acc [{test_acc:.2f}({test_acc_std:.2f})|{ptest_acc:.2f}({ptest_acc_std:.2f})]\t Train Time [{train_time:.2f}]")
            else:
                print(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Train Time [{train_time:.2f}]")

    def _flatten_params(self, state_dict):
        """ 将模型参数展平为向量 """
        return torch.cat([p.flatten() for p in state_dict.values()])
    
    def _unflatten_params(self, flat_params, model):
        """ 将向量恢复为模型参数 """
        pointer = 0
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            numel = param.numel()
            state_dict[name] = flat_params[pointer:pointer+numel].view_as(param)
            pointer += numel
        return state_dict
    
    def train_flow_model(self):
        """ 训练流匹配模型 """
        # Check buffer size against a configurable minimum
        min_buffer_size = getattr(self.args, 'fm_min_buffer', 50) 
        if len(self.param_buffer) < min_buffer_size:
             print(f"Buffer size {len(self.param_buffer)} < {min_buffer_size}. Skipping Flow Model training.")
             return
        
        # 准备训练数据
        # Convert deque or list to tensor
        if isinstance(self.param_buffer, torch.Tensor):
            # Should ideally be storing CPU tensors in buffer
            params_tensor = self.param_buffer.cpu() 
        else:
            # Assuming buffer contains list/deque of CPU tensors
            params_list = list(self.param_buffer)
            params_tensor = torch.stack(params_list)
        
        # Normalize parameters for training stability
        self.param_mean = params_tensor.mean(dim=0, keepdim=True)
        self.param_std = params_tensor.std(dim=0, keepdim=True) + 1e-8
        params_tensor_normalized = (params_tensor - self.param_mean) / self.param_std

        dataset = TensorDataset(params_tensor_normalized)
        batch_size = getattr(self.args, 'batch_size', 32)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # 训练循环
        self.flow_model.train()
        losses = AverageMeter()
        num_fm_epochs = getattr(self.args, 'fm_epochs', 5) 

        for epoch in range(num_fm_epochs):
            for batch in loader:
                # Renamed for clarity, these are normalized
                real_params_normalized = batch[0].to(self.device)
                
                # 流匹配训练
                # Sample t from [eps, 1-eps] for stability
                t = torch.rand(real_params_normalized.size(0), 1).to(self.device) * 0.98 + 0.01
                noise = torch.randn_like(real_params_normalized) # Noise in normalized space
                # Interpolate in normalized space
                noise_params_normalized = real_params_normalized * t + noise * (1 - t) 
                
                # Pass normalized params to flow loss
                flow_loss = self.flow_model.compute_flow_loss(real_params_normalized, noise_params_normalized, t)
                
                
                # 总损失 (Only Flow Loss now)
                total_loss = flow_loss
                
                losses.update(total_loss.item(), real_params_normalized.size(0))
                self.optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping is often helpful for FM/Diffusion
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), 1.0) # Clip norm to 1.0
                self.optimizer.step()
                
        print(f"Round {self.r} Flow Model Training Avg Loss: {losses.avg:.4f}")

    def aggregate_parameters(self):
        # --- 1. Parameter Collection & Preprocessing ---
        client_params_list = []
        client_weights = []
        total_samples = 0
        global_params_flat = self._flatten_params(self.model.state_dict()).to(self.device) # Current global model

        for c in self.active_clients:
            try:
                params_flat = self._flatten_params(c.model.state_dict())
                params_flat = torch.nan_to_num(params_flat, nan=0.0) # Handle NaNs
                 # Moderate clamping, consider making this adaptive or configurable
                params_flat = torch.clamp(params_flat, -10.0, 10.0)

                # Optional: Abnormal update detection (keep if useful)
                param_diff = params_flat.to(self.device) - global_params_flat
                global_norm = global_params_flat.norm()
                 # Adaptive threshold, maybe tune multiplier (e.g., 2 instead of 3?)
                threshold = max(3 * global_norm, 50.0) # Lowered minimum threshold
                if param_diff.norm() > threshold:
                    print(f"Client {c.client_idx} update norm {param_diff.norm():.1f} > threshold {threshold:.1f}. Clipping update.")
                    # Norm clipping instead of rejection
                    clipped_param_diff = param_diff * (threshold / param_diff.norm())
                    params_flat = global_params_flat + clipped_param_diff # Apply clipped diff to global params
                
                # Store valid params (on CPU) and weights
                client_params_list.append(params_flat.cpu()) # Ensure params_flat is on CPU before appending
                client_weights.append(c.num_train)
                total_samples += c.num_train

                # Add to history buffer (limit size)
                if len(self.param_buffer) >= getattr(self.args, 'fm_buffer_size', 1000):
                    self.param_buffer.pop(0) # Remove oldest if buffer is full
                self.param_buffer.append(params_flat.cpu().clone()) # Store CPU copy

            except Exception as e:
                print(f"Error collecting params from Client {c.client_idx}: {str(e)}. Skipping client.")
                # Optionally add global params with zero weight or handle differently

        if not client_params_list:
            print("No valid client parameters collected. Skipping aggregation.")
            return # Cannot aggregate if no clients sent valid params

        # --- 2. Flow Model Training ---
        # Train if buffer has sufficient data and FM is enabled
        if self.flow_model and len(self.param_buffer) >= getattr(self.args, 'fm_min_buffer', 50): 
            print("Training Flow Model...")
            self.train_flow_model()
        
        # --- 3. Parameter Generation / Refinement ---
        processed_params_list = []
        if self.flow_model and self.flow_model.training == False: # Check if model was trained (or loaded)
             self.flow_model.eval() # Set to eval mode for generation

        # Calculate client weights
        client_weights = torch.tensor(client_weights, dtype=torch.float32) / total_samples
        
        # Use normalization parameters if FM model was trained
        use_normalization = hasattr(self, 'param_mean') and hasattr(self, 'param_std')

        for i, params_flat_cpu in enumerate(client_params_list):
            try:
                params_flat = params_flat_cpu.to(self.device)
                
                # Normalize if required by the trained FM model
                if use_normalization:
                    # Ensure mean/std are 1D for subtraction/division
                    mean_1d = self.param_mean.squeeze(0).to(self.device)
                    std_1d = self.param_std.squeeze(0).to(self.device)
                    params_norm = (params_flat - mean_1d) / std_1d
                else:
                    params_norm = params_flat # Use directly if no normalization context

                # --- Generation Step ---
                # Input to generate should be [B, D], B=1 here
                input_for_generate = params_norm.unsqueeze(0)
                
                # Option A: Refine client params (current logic)
                if self.flow_model and torch.rand(1) > getattr(self.args, 'fm_skip_prob', 0.1): # Skip prob
                     # generate expects [B, D], outputs [B, D]
                    generated_norm_batched = self.flow_model.generate(input_for_generate)
                    # Remove batch dim, result is [D]
                    generated_norm = generated_norm_batched.squeeze(0) 
                    
                     # Denormalize the output
                    if use_normalization:
                         # Use the same 1D mean/std
                         generated_flat = generated_norm * std_1d + mean_1d
                    else:
                         generated_flat = generated_norm # Output is already [D]
                else:
                     # Skip generation: Use original params or previous global? Using original seems safer.
                     generated_flat = params_flat # Use the original params (shape [D]) if skipping or no FM model

                # --- Interpolation Step (Optional) ---
                alpha = max(0.8, 0.95 ** self.r) # Keep schedule or make fixed/configurable
                # Both params_flat and generated_flat should now be [D]
                final_flat_params = alpha * params_flat + (1 - alpha) * generated_flat
                final_flat_params = torch.clamp(final_flat_params, -5.0, 5.0) # Clamp final result
                
                processed_params_list.append(final_flat_params.detach()) # Store processed params (on device)

            except Exception as e:
                print(f"Error generating/processing params for client {self.active_clients[i].client_idx}: {e}. Using original params.")
                processed_params_list.append(params_flat_cpu.to(self.device).detach()) # Fallback to original params on device

        # --- 4. Weighted Aggregation ---
        if not processed_params_list:
             print("No parameters processed. Skipping model update.")
             return

        # Initialize global params accumulator
        aggregated_flat_params = torch.zeros(self.param_dim, device=self.device) 
        
        # Ensure weights are on the correct device
        client_weights = client_weights.to(self.device)

        for i, final_flat in enumerate(processed_params_list):
             # Weighted sum
             aggregated_flat_params += client_weights[i] * final_flat 

        # --- 5. Update Global Model ---
        try:
            aggregated_state_dict = self._unflatten_params(aggregated_flat_params, self.model)
            self.model.load_state_dict(aggregated_state_dict)
            print("Global model updated via Flow-Assisted Aggregation.")
        except Exception as e:
            print(f"Error loading aggregated state dict: {e}. Global model not updated.")

        total_samples = sum(c.num_train for c in self.active_clients)
        self.global_means.data = torch.zeros_like(self.clients[0].means)
        self.global_covariance.data = torch.zeros_like(self.clients[0].covariance)
        
        for c in self.active_clients:
            self.global_means.data = self.global_means.data + (c.num_train / total_samples)*c.adaptive_means.data
            self.global_covariance.data = self.global_covariance.data + (c.num_train / total_samples)*c.adaptive_covariance.data
        # Note: Sending updated model to clients happens in `send_models()` call at the start of the next round.
        # The old logic mixing aggregation and client updates is removed.

    
    def send_models(self):
        # send base model
        super().send_models()
        # send global gaussian estiamtes
        for c in self.active_clients:
            c.global_means.data = self.global_means.data
            c.global_covariance.data = self.global_covariance.data
            c.set_model(self.model)
            if self.r == 1:
                c.means.data = self.global_means.data
                c.covariance.data = self.global_covariance.data
                c.adaptive_means.data = self.global_means.data
                c.adaptive_covariance.data = self.global_covariance.data

    def evaluate_personalized(self):
        """
        Update global \beta and compute local interpolated (\mu, \Sigma) 
        Evaluate using interpolated (\mu, \Sigma) and local prior
        """
        total_samples = sum(c.num_test for c in self.clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        kl_divs = []
        for c in self.clients:
            old_model = deepcopy(c.model)
            c.model = deepcopy(self.model)
            c.global_means.data = self.global_means.data
            c.global_covariance.data = self.global_covariance.data
            c.global_means = c.global_means.to(self.device)
            c.global_covariance = c.global_covariance.to(self.device)
            c.model.eval()
            # solve for beta and use adaptive statistics for classifier
            c_feats, c_labels = c.compute_feats(split="train")
            c.solve_beta(feats=c_feats, labels=c_labels)
            means_mle, scatter_mle, priors, counts = c.compute_mle_statistics(feats=c_feats, labels=c_labels)
            means_mle = torch.stack([means_mle[i] if means_mle[i] is not None and counts[i] > c.min_samples else c.global_means[i] for i in range(self.num_classes)])
            cov_mle = (scatter_mle / (np.sum(counts)-1)) + 1e-4 + torch.eye(self.D).to(self.device)
            cov_psd = cov_nearest(cov_mle.cpu().numpy(), method="clipped")
            cov_psd = torch.Tensor(cov_psd).to(self.device)
            c.update(means_mle, cov_psd)
            c.set_lda_weights(c.adaptive_means, c.adaptive_covariance)
            with torch.no_grad():
                acc, loss = c.evaluate()
                accs.append(acc)
                weighted_loss += (c.num_test / total_samples) * loss.detach()
                weighted_acc += (c.num_test / total_samples) * acc
                c.model = old_model
                # 确保所有张量在同一设备
                local_means = means_mle.to(self.device)
                local_cov = cov_psd.to(self.device)
                global_means = self.global_means.to(self.device)
                global_cov = self.global_covariance.to(self.device)
                
                local_dist = MultivariateNormal(local_means, local_cov)
                global_dist = MultivariateNormal(global_means, global_cov)
                kl_div = torch.distributions.kl.kl_divergence(local_dist, global_dist)
                kl_divs.append(kl_div)
            c.model = c.model.to("cpu")
            c.global_means = c.global_means.to("cpu")
            c.global_covariance = c.global_covariance.to("cpu")
            c.adaptive_means = c.adaptive_means.to("cpu")
            c.adaptive_covariance = c.adaptive_covariance.to("cpu")
            c.means = c.means.to("cpu")
            c.covariance = c.covariance.to("cpu")
        print(f"KL Divergence: μ={torch.mean(torch.stack(kl_divs)):.4f}")
        std = torch.std(torch.stack(accs))
        return weighted_acc, weighted_loss, std