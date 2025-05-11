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
        target_flow = real_params - noise_params
        
        # 添加参数范围检查
        noise_params = torch.clamp(noise_params, -3.0, 3.0)
        
        pred_flow = self(noise_params, t)
        
        # 使用L1损失代替MSE，对异常值更鲁棒
        loss = F.l1_loss(pred_flow, target_flow)
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
        
        # 减小步长和噪声
        dt = 0.5 / num_steps  # 更小的步长
        noise_decay = 0.95
        
        for step in range(num_steps):
            # 降低噪声强度
            current_noise = (0.2 * (noise_decay ** step)) * torch.randn_like(z)
            t = torch.ones(z.size(0), 1).to(z.device) * (1.0 - step/num_steps)
            
            # 添加异常值检测
            if torch.isnan(z).any():
                print(f"NaN detected at step {step}, resetting to initial")
                return init_params.clone() # 完全回退到初始参数
            
            pred = self(z + current_noise, t)
            # 限制单步更新幅度
            pred = torch.clamp(pred, -0.5, 0.5)
            z = z + pred * dt
            
            # 更严格的clamp值
            z = torch.clamp(z, -3.0, 3.0)
            z = torch.nan_to_num(z, nan=0.0)
        
        return z


class ServerFedAvg(Server):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.clients = []
        for client_idx in range(self.num_clients):
            c = ClientFedAvg(args, client_idx)
            self.clients.append(c)
        self.flow_model = None
        self.param_buffer = []  # 参数历史缓冲区
        self.init_flow_model()
        self.r = 1
    
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

    def send_models(self):
        # Send the current global model state to active clients
        for c in self.active_clients:
            # Ensure model is on the correct device for the client BEFORE sending state_dict
            c.set_model(self.model) # This method should handle loading the state_dict

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
        if len(self.param_buffer) > min_buffer_size * 2:
            # 随机采样+最近参数采样结合
            recent_params = self.param_buffer[-min_buffer_size//2:]
            random_indices = torch.randperm(len(self.param_buffer)-len(recent_params))[:min_buffer_size//2]
            random_params = [self.param_buffer[i] for i in random_indices]
            sampled_params = recent_params + random_params
            params_tensor = torch.stack(sampled_params)
        else:
            params_tensor = torch.stack(list(self.param_buffer))
        
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
                # 降低梯度裁剪阈值
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), 0.5)
                # 检查梯度是否包含NaN
                if any(p.grad is not None and torch.isnan(p.grad).any() for p in self.flow_model.parameters()):
                    print("NaN梯度检测，跳过此批次")
                    self.optimizer.zero_grad()
                    continue
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
                    print(f"Client {c.client_idx} update norm {param_diff.norm():.1f} > threshold {threshold:.1f}. Using global params.")
                    params_flat = global_params_flat.cpu().clone() # Use CPU version of global params
                
                # Store valid params (on CPU) and weights
                client_params_list.append(params_flat.cpu()) 
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
            # 检测NaN并替换
            if torch.isnan(params_flat_cpu).any():
                print(f"Client {self.active_clients[i].client_idx} has NaN params, using global model.")
                params_flat_cpu = global_params_flat.cpu().clone()
            
            params_flat = params_flat_cpu.to(self.device)
            
            # 更严格的参数裁剪
            params_flat = torch.clamp(params_flat, -3.0, 3.0)

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
            
            # Default to original parameters
            generated_flat = params_flat # Shape [D]
            
            # Option A: Refine client params (current logic) - Modified
            if self.flow_model and torch.rand(1) > getattr(self.args, 'fm_skip_prob', 0.1): # Skip prob
                try:
                    # generate expects [B, D], outputs [B, D]
                    generated_norm_batched = self.flow_model.generate(input_for_generate)
                    # Remove batch dim, result is [D]
                    generated_norm = generated_norm_batched.squeeze(0) 
                    
                    # Denormalize the output
                    if use_normalization:
                        # Use the same 1D mean/std
                        generated_flat_denorm = generated_norm * std_1d + mean_1d
                    else:
                        generated_flat_denorm = generated_norm # Output is already [D]
                    
                    # Check for NaN/Inf in generated parameters after denormalization
                    if torch.isnan(generated_flat_denorm).any() or torch.isinf(generated_flat_denorm).any():
                         print(f"Warning: NaN/Inf detected in generated params for client {self.active_clients[i].client_idx}. Using original params.")
                         generated_flat = params_flat # Fallback to original if generation failed
                    else:
                         generated_flat = generated_flat_denorm # Use the successfully generated & denormalized params

                except Exception as gen_e:
                    print(f"Error during generation for client {self.active_clients[i].client_idx}: {gen_e}. Using original params.")
                    generated_flat = params_flat # Fallback on error

            # --- Interpolation Step ---
            # Define an interpolation factor alpha. Start conservatively (e.g., 0.2)
            # You can make this adaptive later if needed.
            alpha = getattr(self.args, 'fm_interpolation_alpha', 0.2) 
            
            # Interpolate: alpha * generated + (1-alpha) * original
            # Ensure both tensors are on the same device ([D])
            final_flat_params = alpha * generated_flat + (1 - alpha) * params_flat
            
            # Clamp final result (adjust range if needed)
            final_flat_params = torch.clamp(final_flat_params, -5.0, 5.0) 
            
            processed_params_list.append(final_flat_params.detach()) # Store processed params (on device)

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


        # Note: Sending updated model to clients happens in `send_models()` call at the start of the next round.
        # The old logic mixing aggregation and client updates is removed.

    def _split_params(self, flat_params):
        """分离基础层和个性化层参数"""
        # 假设前60%是特征提取层，后40%是分类层
        split_idx = int(flat_params.shape[0] * 0.6)
        return flat_params[:split_idx], flat_params[split_idx:]

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
                
    def evaluate(self):
        total_samples = sum(c.num_test for c in self.clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        for c in self.clients:
            # Evaluate using the current global model held by the server
            # Temporarily set client's model for evaluation function compatibility
            original_client_model_state = deepcopy(c.model.state_dict()) 
            c.model.load_state_dict(deepcopy(self.model.state_dict()))
            c.model.to(c.device) # Ensure model is on client's device for evaluation
            
            acc, loss = c.evaluate() # evaluate should handle model.eval() and device placement
            accs.append(acc.cpu()) # Collect accuracy on CPU
            
            weighted_loss += (c.num_test / total_samples) * loss.item() # Use .item()
            weighted_acc += (c.num_test / total_samples) * acc.item() # Use .item()
            
            # Restore original client model state if necessary (though likely overwritten next round)
            c.model.load_state_dict(original_client_model_state) 
            
        std = torch.std(torch.stack(accs)).item() # Calculate std dev
        return weighted_acc, weighted_loss, std

    def evaluate_personalized(self):
        total_samples = sum(c.num_test for c in self.clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        for c in self.clients:
            # Personalized evaluation: Fine-tune the current global model on client's data
            original_client_model_state = deepcopy(c.model.state_dict())
            c.model.load_state_dict(deepcopy(self.model.state_dict())) # Start from global
            c.model.to(c.device)

            # Perform local fine-tuning (ensure c.train() does this)
            # Make sure c.train() runs for a small number of steps/epochs suitable for personalization
            c.train() # Re-uses the client's train method - check if it's suitable for fine-tuning
            
            # Evaluate the fine-tuned model
            acc, loss = c.evaluate() 
            accs.append(acc.cpu())
            
            weighted_loss += (c.num_test / total_samples) * loss.item()
            weighted_acc += (c.num_test / total_samples) * acc.item()
            
            # Restore original client model state
            c.model.load_state_dict(original_client_model_state)

        std = torch.std(torch.stack(accs)).item()
        return weighted_acc, weighted_loss, std
    