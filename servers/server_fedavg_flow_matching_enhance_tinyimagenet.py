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
import random # Import random for potential sampling strategies

# --- FlowMatchingModel 改进 ---
class FlowMatchingModel(nn.Module):
    def __init__(self, param_dim, hidden_dim=1024, rank=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64)
        )
        
        self.input_norm = nn.LayerNorm(param_dim)
        self.low_rank_proj = nn.Linear(param_dim, rank)  # 降维
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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # --- 使用 Huber Loss 代替 MSE Loss ---
    def compute_flow_loss(self, real_params, noise_params, t, delta=1.0):
        """
        使用 Huber Loss 计算 Flow Matching 损失。
        delta: Huber Loss 的阈值，小于 delta 使用平方误差，大于 delta 使用绝对误差。
               可以根据参数的范数或差异的分布来调整。
        """
        target_flow = real_params - noise_params 
        
        # 添加输入值裁剪 (保留原有，增强稳定性)
        noise_params = torch.clamp(noise_params, -5.0, 5.0)
        
        pred_flow = self(noise_params, t)
        
        # Huber Loss
        diff = pred_flow - target_flow
        loss = torch.where(torch.abs(diff) < delta, 0.5 * diff ** 2, delta * (torch.abs(diff) - 0.5 * delta))
        return torch.mean(loss) # 对所有元素求平均

    def forward(self, z, t):
        """ 
        输入维度修正：
        z: [batch_size, param_dim] 
        t: [batch_size, 1]
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        t_emb = self.time_embed(t)
        z = self.low_rank_proj(z)
        x = torch.cat([z, t_emb], dim=-1)
        return self.inv_proj(self.main_net(x))

    @torch.no_grad()
    def generate(self, init_params, num_steps=100, conditioning_info=None):
        """
        改进的生成方法，可以考虑加入条件信息 (虽然这里没有实际使用 conditioning_info，
        但保留接口用于后续扩展)。
        """
        z = init_params.clone()
        
        # 改进的采样器
        dt = 1.0 / num_steps
        # 调整噪声衰减策略，使其更平缓，或者考虑在后期减少噪声
        noise_decay = 0.99 
        
        for step in range(num_steps):
            # 可以在这里根据 conditioning_info 调整噪声或步长
            current_noise = (0.5 * (noise_decay ** step)) * torch.randn_like(z)
            t = torch.ones(z.size(0), 1).to(z.device) * (1.0 - step/num_steps)
            
            pred = self(z + current_noise, t)
            z = z + pred * dt
            
            # 强约束 (保留，但可以根据需要调整范围)
            z = torch.clamp(z, -10.0, 10.0) # 适当放宽裁剪范围
            z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0) # 增加对 inf 的处理
        
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
        # 使用 deque 更好地管理 buffer，限制大小
        from collections import deque
        self.param_buffer = deque(maxlen=getattr(self.args, 'fm_buffer_size', 1000)) 
        self.init_flow_model()
        self.r = 1
        # 添加一个参数来控制 Flow Model 的训练频率
        self.fm_train_freq = getattr(self.args, 'fm_train_freq', 1) # 默认每轮都训练

    def init_flow_model(self):
        with torch.no_grad():
            dummy_model = copy.deepcopy(self.model).to('cpu') 
            if not hasattr(self.model, 'input_shape'):
                 self.model.input_shape = (3, 32, 32) 
                 print("Warning: Model input_shape not found, assuming (3, 32, 32) for CIFAR10.")
            dummy_input = torch.randn(1, *self.model.input_shape)
            _ = dummy_model(dummy_input)
            param_dim = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad) 
            del dummy_model 
        
        print(f"Initializing FlowMatchingModel with param_dim={param_dim}")
        self.flow_model = FlowMatchingModel(param_dim).to(self.device)
        # 适当调整学习率和权重衰减
        self.optimizer = torch.optim.AdamW(self.flow_model.parameters(), lr=5e-5, weight_decay=1e-4) 
        self.param_dim = param_dim

    def send_models(self):
        for c in self.active_clients:
            c.set_model(self.model)

    def _flatten_params(self, state_dict):
        """ 将模型参数展平为向量 """
        # Ensure parameters are on CPU before flattening for buffer storage
        return torch.cat([p.flatten().cpu() for p in state_dict.values()])
    
    def _unflatten_params(self, flat_params, model):
        """ 将向量恢复为模型参数 """
        pointer = 0
        state_dict = model.state_dict()
        # Ensure flat_params is on the correct device before unflattening
        flat_params = flat_params.to(self.device) 
        for name, param in state_dict.items():
            numel = param.numel()
            state_dict[name].copy_(flat_params[pointer:pointer+numel].view_as(param)) # Use copy_ for safety
            pointer += numel
        return state_dict
    
    def train_flow_model(self):
        """ 训练流匹配模型 """
        min_buffer_size = getattr(self.args, 'fm_min_buffer', 50) 
        if len(self.param_buffer) < min_buffer_size:
             print(f"Buffer size {len(self.param_buffer)} < {min_buffer_size}. Skipping Flow Model training.")
             return
        
        # Convert deque to tensor (assuming CPU tensors are stored)
        params_tensor_cpu = torch.stack(list(self.param_buffer))
        
        # Normalize parameters
        self.param_mean = params_tensor_cpu.mean(dim=0, keepdim=True)
        self.param_std = params_tensor_cpu.std(dim=0, keepdim=True) + 1e-8
        params_tensor_normalized = (params_tensor_cpu - self.param_mean) / self.param_std

        dataset = TensorDataset(params_tensor_normalized)
        batch_size = getattr(self.args, 'fm_batch_size', 64) # Increase batch size for better gradient estimation
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self.flow_model.train()
        losses = AverageMeter()
        num_fm_epochs = getattr(self.args, 'fm_epochs', 10) # 增加训练 epochs

        for epoch in range(num_fm_epochs):
            for batch in loader:
                real_params_normalized = batch[0].to(self.device)
                
                t = torch.rand(real_params_normalized.size(0), 1).to(self.device) * 0.98 + 0.01
                noise = torch.randn_like(real_params_normalized)
                noise_params_normalized = real_params_normalized * t + noise * (1 - t) 
                
                # 可以尝试动态调整 Huber Loss 的 delta 参数
                delta = 1.0 # 初始 delta 值
                if self.param_std is not None:
                     # 考虑基于参数标准差的 delta 调整
                     avg_param_std = self.param_std.mean().item()
                     delta = max(1.0, avg_param_std * 0.5) # Example: delta related to average std dev
                
                flow_loss = self.flow_model.compute_flow_loss(real_params_normalized, noise_params_normalized, t, delta=delta)
                
                total_loss = flow_loss
                
                losses.update(total_loss.item(), real_params_normalized.size(0))
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), getattr(self.args, 'fm_grad_norm', 1.0)) # 可配置的梯度裁剪
                self.optimizer.step()
                
        print(f"Round {self.r} Flow Model Training Avg Loss: {losses.avg:.4f}")
        
    def aggregate_parameters(self):
        client_params_list = []
        client_weights = []
        total_samples = 0
        global_params_flat = self._flatten_params(self.model.state_dict()).to(self.device) # Current global model on device

        for c in self.active_clients:
            try:
                params_flat_cpu = self._flatten_params(c.model.state_dict()) # Flattened params on CPU
                
                # --- 更稳健的参数校验和滤波 ---
                params_flat = params_flat_cpu.to(self.device) # Move to device for calculations
                param_diff = params_flat - global_params_flat
                diff_norm = param_diff.norm().item() # Use item() to get scalar value
                global_norm = global_params_flat.norm().item() # Use item()

                # 动态调整阈值，并增加一个绝对阈值
                # 提高对异常更新的检测能力
                threshold = max(getattr(self.args, 'param_norm_threshold_ratio', 5.0) * global_norm, 
                                getattr(self.args, 'param_norm_threshold_abs', 100.0)) # 可配置的相对和绝对阈值
                
                if diff_norm > threshold:
                    print(f"Client {c.client_idx} update norm {diff_norm:.1f} > threshold {threshold:.1f}. Skipping this update.")
                    # Option 1: Skip the client's update entirely
                    continue 
                    # Option 2: Use the previous global model's parameters for this client
                    # params_flat_cpu = self._flatten_params(self.model.state_dict()) 
                    # print(f"Client {c.client_idx} update norm {diff_norm:.1f} > threshold {threshold:.1f}. Using global params.")

                # Add valid params to buffer (on CPU)
                self.param_buffer.append(params_flat_cpu.clone()) 

                # Store valid params (on device) and weights
                client_params_list.append(params_flat.detach()) # Store detached tensor on device
                client_weights.append(c.num_train)
                total_samples += c.num_train

            except Exception as e:
                print(f"Error collecting params from Client {c.client_idx}: {str(e)}. Skipping client.")
                continue # Skip the client if an error occurs

        if not client_params_list:
            print("No valid client parameters collected. Skipping aggregation.")
            return 

        # --- Flow Model Training ( conditionally based on frequency) ---
        if self.flow_model and len(self.param_buffer) >= getattr(self.args, 'fm_min_buffer', 50) and self.r % self.fm_train_freq == 0: 
            print("Training Flow Model...")
            self.train_flow_model()
        
        # --- Parameter Generation / Refinement ---
        processed_params_list = []
        # Only use FM for generation if it exists and was trained at least once
        use_fm_generation = self.flow_model is not None and hasattr(self, 'param_mean') 

        if use_fm_generation:
             self.flow_model.eval() 

        # Calculate client weights
        client_weights_tensor = torch.tensor(client_weights, dtype=torch.float32).to(self.device) # Weights on device
        client_weights_tensor /= total_samples

        for i, params_flat in enumerate(client_params_list): # params_flat is already on device
            try:
                if use_fm_generation and torch.rand(1).item() > getattr(self.args, 'fm_skip_prob', 0.05): # 降低跳过概率
                     # Normalize the client's parameters using the calculated mean/std
                     mean_1d = self.param_mean.squeeze(0).to(self.device)
                     std_1d = self.param_std.squeeze(0).to(self.device)
                     params_norm = (params_flat - mean_1d) / std_1d

                     # Generate refined parameters
                     generated_norm_batched = self.flow_model.generate(params_norm.unsqueeze(0))
                     generated_norm = generated_norm_batched.squeeze(0) 
                     
                     # Denormalize the output
                     generated_flat = generated_norm * std_1d + mean_1d
                     
                     # --- 混合原始参数和生成参数 ---
                     # 引入一个混合因子 (例如，随轮次衰减)
                     # 可以根据客户端的可靠性进一步调整这个因子 (未在代码中实现)
                     mix_alpha = max(0.6, 0.95 ** self.r) # 更高的起始混合比例
                     final_flat_params = mix_alpha * params_flat + (1 - mix_alpha) * generated_flat
                     
                else:
                     # If not using FM generation, use the original client parameters
                     final_flat_params = params_flat 

                # 最终裁剪 (可以根据参数的实际范围调整)
                final_flat_params = torch.clamp(final_flat_params, 
                                                getattr(self.args, 'final_clamp_min', -10.0), 
                                                getattr(self.args, 'final_clamp_max', 10.0)) 
                final_flat_params = torch.nan_to_num(final_flat_params, nan=0.0) # Ensure no NaNs after processing
                
                processed_params_list.append(final_flat_params.detach()) 

            except Exception as e:
                print(f"Error generating/processing params for client {self.active_clients[i].client_idx if i < len(self.active_clients) else 'N/A'}: {e}. Using original params.")
                # Fallback to original params on device if processing fails
                processed_params_list.append(client_params_list[i].detach()) 

        # --- Weighted Aggregation ---
        if not processed_params_list:
             print("No parameters processed. Skipping model update.")
             return

        aggregated_flat_params = torch.zeros(self.param_dim, device=self.device) 
        
        # Ensure weights match the number of processed params
        if len(processed_params_list) != len(client_weights_tensor):
             print("Mismatch in processed params and weights. Using simple average of processed params.")
             # Simple average as fallback
             aggregated_flat_params = torch.mean(torch.stack(processed_params_list), dim=0)
        else:
            for i, final_flat in enumerate(processed_params_list):
                 # Weighted sum
                 aggregated_flat_params += client_weights_tensor[i] * final_flat 

        # --- Update Global Model ---
        try:
            # Ensure model is on the correct device before unflattening
            self.model.to(self.device) 
            aggregated_state_dict = self._unflatten_params(aggregated_flat_params, self.model)
            self.model.load_state_dict(aggregated_state_dict)
            print("Global model updated via Flow-Assisted Aggregation.")
        except Exception as e:
            print(f"Error loading aggregated state dict: {e}. Global model not updated.")

    # train, evaluate, evaluate_personalized 方法保持不变
    def train(self):
        for r in range(1, self.global_rounds+1):
            start_time = time.time()
            self.r = r
            if r == (self.global_rounds): 
                self.sampling_prob = 1.0
            self.sample_active_clients()
            self.send_models() 

            train_acc, train_loss = self.train_clients()
            train_time = time.time() - start_time

            self.aggregate_parameters() 

            round_time = time.time() - start_time
            self.train_times.append(train_time)
            self.round_times.append(round_time)

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
            original_client_model_state = deepcopy(c.model.state_dict()) 
            c.model.load_state_dict(deepcopy(self.model.state_dict()))
            c.model.to(c.device) 
            
            acc, loss = c.evaluate() 
            accs.append(acc.cpu()) 
            
            weighted_loss += (c.num_test / total_samples) * loss.item() 
            weighted_acc += (c.num_test / total_samples) * acc.item() 
            
            c.model.load_state_dict(original_client_model_state) 
            
        std = torch.std(torch.stack(accs)).item() 
        return weighted_acc, weighted_loss, std

    def evaluate_personalized(self):
        total_samples = sum(c.num_test for c in self.clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        for c in self.clients:
            original_client_model_state = deepcopy(c.model.state_dict())
            c.model.load_state_dict(deepcopy(self.model.state_dict())) 
            c.model.to(c.device)

            c.train() 
            
            acc, loss = c.evaluate() 
            accs.append(acc.cpu())
            
            weighted_loss += (c.num_test / total_samples) * loss.item()
            weighted_acc += (c.num_test / total_samples) * acc.item()
            
            c.model.load_state_dict(original_client_model_state)

        std = torch.std(torch.stack(accs)).item()
        return weighted_acc, weighted_loss, std