import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn
import torch.nn.functional as F # Needed for FlowMatchingModel
from sklearn.decomposition import PCA
import traceback # For debugging if needed

# ... (之前的 FlowMatchingModel, load_flow_model, get_param_dim_from_model_weights 函数保持不变) ...
# --- FlowMatchingModel class definition (as corrected in previous step) ---
class FlowMatchingModel(nn.Module):
    def __init__(self, param_dim, hidden_dim=1024, rank=128, dropout_rate=0.1): # Adjusted defaults
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),       # Matches checkpoint: [32, 1]
            nn.SiLU(),
            nn.Linear(32, 64)       # Matches checkpoint: [64, 32] -> output 64
        )
        self.input_norm = nn.LayerNorm(param_dim)
        self.low_rank_proj = nn.Linear(param_dim, rank) # rank default 128, matches checkpoint

        # Input to main_net's first linear is rank + time_embed_output = 128 + 64 = 192
        self.main_net = nn.Sequential(
            nn.Linear(rank + 64, hidden_dim), # hidden_dim default 1024. Input: 128+64=192. Matches checkpoint [1024,192]
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate), # dropout_rate default 0.1
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, rank) # Output is rank (128)
        )
        self.inv_proj = nn.Linear(rank, param_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, z, t): # Adapted from previous script, ensure consistency
        if z.dim() == 1: z = z.unsqueeze(0)
        if t.dim() == 0: t = t.view(1, 1).expand(z.size(0), 1)
        elif t.dim() == 1 and t.size(0) == z.size(0) and t.size(-1) != 1 : t = t.unsqueeze(-1)
        elif t.dim() == 1 and t.size(0) == 1: t = t.expand(z.size(0), 1)
        elif t.dim() == 2 and t.size(0) == 1 and t.size(1) == 1: t = t.expand(z.size(0), 1)
        elif t.dim() != 2 or t.size(0) != z.size(0) or t.size(1) != 1:
             raise ValueError(f"Unexpected shape for t: {t.shape}, expected ({z.size(0)}, 1)")

        z_norm = self.input_norm(z)
        z_proj = self.low_rank_proj(z_norm)
        t_emb = self.time_embed(t.float())
        x = torch.cat([z_proj, t_emb], dim=-1)
        output = self.inv_proj(self.main_net(x))
        return output

    @torch.no_grad()
    def generate(self, init_params, num_steps=100, clamp_val=20.0):
        z = init_params.clone()
        dt_step = 1.0 / num_steps
        self.eval()
        trajectory = [z.cpu().clone()]
        for step in range(num_steps):
            t_val_model = 1.0 - step / num_steps
            t_tensor = torch.ones(z.size(0), 1, device=z.device) * t_val_model
            pred = self(z, t_tensor)
            z = z + pred * dt_step
            z = torch.clamp(z, -clamp_val, clamp_val)
            z = torch.nan_to_num(z, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
            trajectory.append(z.cpu().clone())
        return trajectory
# --- End of FlowMatchingModel class ---

# --- load_flow_model and get_param_dim_from_model_weights functions (as corrected in previous step) ---
def load_flow_model(model_path, param_dim, model_hyperparams=None):
    if model_hyperparams is None:
        # Default to the corrected smaller architecture if not specified
        model_hyperparams = {'hidden_dim': 1024, 'rank': 128, 'dropout_rate': 0.1}
        print(f"Warning: Using default model hyperparameters for loading: {model_hyperparams}")
        print("Ensure these match the saved model's architecture.")

    model = FlowMatchingModel(param_dim, **model_hyperparams)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model to: {device}")
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Strict loading failed: {e}. Attempting non-strict loading.")
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model, device

def get_param_dim_from_model_weights(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
         state_dict = state_dict['model_state_dict']
    param_dim = None
    key_to_check = 'input_norm.weight'
    possible_prefixes = ["", "flow_model."]
    for prefix in possible_prefixes:
        if prefix + key_to_check in state_dict:
            param_dim = state_dict[prefix + key_to_check].shape[0]
            print(f"Found key '{prefix + key_to_check}', param_dim: {param_dim}")
            break
    if param_dim is None:
        raise KeyError(f"Could not find '{key_to_check}' ... Available keys: {list(state_dict.keys())[:10]}...")
    return param_dim
# --- End of load_flow_model and get_param_dim_from_model_weights ---

def visualize_flow_trajectory(flow_model, device, param_dim, n_visualization_steps=8, n_samples=300,
                              generation_num_steps=100, clamp_val_generate=20.0,
                              save_path='flow_model_trajectory.png'):
    """
    Visualizes the flow trajectory using the model's generate method.
    Plots in a 2-row grid. Number of columns adjusts.
    For a 2x4 grid, ensure n_visualization_steps is 8.
    """
    if n_visualization_steps <= 0:
        print("n_visualization_steps must be positive.")
        return
    if n_visualization_steps > generation_num_steps + 1:
        print(f"Warning: n_visualization_steps ({n_visualization_steps}) > generation_num_steps+1 ({generation_num_steps+1})."
              f" Setting n_visualization_steps to {generation_num_steps+1}.")
        n_visualization_steps = generation_num_steps + 1

    initial_noise = torch.randn(n_samples, param_dim, device=device)
    full_trajectory_cpu = flow_model.generate(initial_noise, num_steps=generation_num_steps, clamp_val=clamp_val_generate)
    indices_to_visualize = np.linspace(0, generation_num_steps, n_visualization_steps, dtype=int)
    zs_for_visualization_cpu = [full_trajectory_cpu[i] for i in indices_to_visualize]

    all_z_for_pca_np = np.concatenate([z.numpy() for z in zs_for_visualization_cpu], axis=0)
    pca = PCA(n_components=2)
    try:
        all_z_2d = pca.fit_transform(all_z_for_pca_np)
    except Exception as e:
        print(f"PCA failed: {e}")
        # ... (PCA error handling as before) ...
        return

    zs_2d_viz = [all_z_2d[i*n_samples:(i+1)*n_samples] for i in range(n_visualization_steps)]

    # --- Plotting Adjustments for 2 Rows ---
    nrows = 2
    # Calculate number of columns needed
    ncols = (n_visualization_steps + nrows - 1) // nrows  # Equivalent to ceil(n_visualization_steps / nrows)
    
    # Adjust figsize: (width_per_plot * ncols, height_per_plot * nrows)
    # Original was (3 * n_visualization_steps, 3.5) for 1 row.
    # For 2 rows, if each plot is ~3 wide and 3.5 high:
    fig_width = 3 * ncols
    fig_height = 3.5 * nrows 
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    
    # If n_visualization_steps = 1, axes might not be an array. Flatten for easier iteration.
    # If nrows*ncols = 1, axes is not an array. If nrows=1 or ncols=1, it's 1D. Otherwise 2D.
    if n_visualization_steps == 1:
        axes_flat = [axes]
    elif nrows == 1 or ncols == 1:
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes] # handles single plot case
    else: # nrows > 1 and ncols > 1
        axes_flat = axes.flatten()


    min_x_overall, min_y_overall = np.min(all_z_2d, axis=0)
    max_x_overall, max_y_overall = np.max(all_z_2d, axis=0)
    margin_x = (max_x_overall - min_x_overall) * 0.1 if (max_x_overall - min_x_overall) > 1e-6 else 0.1
    margin_y = (max_y_overall - min_y_overall) * 0.1 if (max_y_overall - min_y_overall) > 1e-6 else 0.1
    final_xlim = (min_x_overall - margin_x, max_x_overall + margin_x)
    final_ylim = (min_y_overall - margin_y, max_y_overall + margin_y)

    time_points_plot = np.linspace(0, 1.0, n_visualization_steps)

    for i, z2d_step in enumerate(zs_2d_viz):
        ax = axes_flat[i] # Use the flattened list of axes
        ax.scatter(z2d_step[:, 0], z2d_step[:, 1], s=10, alpha=0.7)
        ax.set_title(f't = {time_points_plot[i]:.2f}')
        ax.set_xlim(final_xlim)
        ax.set_ylim(final_ylim)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide any unused subplots if n_visualization_steps is not a multiple of ncols*nrows
    for i in range(n_visualization_steps, nrows * ncols):
        fig.delaxes(axes_flat[i])

    # plt.suptitle(f"Flow Model Trajectory ({os.path.basename(model_path)})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved flow trajectory visualization to {save_path}")
    print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}, Sum: {np.sum(pca.explained_variance_ratio_)}")


if __name__ == '__main__':
    # model_path = '/home/xiongzc/Desktop/pFedFDA-main/eval_model/fedavg_cifar10_dir01_CNN_global_flow_weights_round200.pth'
    # 使用你之前成功加载的 round20 的模型路径进行测试，确保图片能出来
    model_path = '/home/xiongzc/Desktop/pFedFDA-main/eval_model/fedavg_cifar10_dir01_CNN_global_flow_weights_round120.pth'


    try:
        param_dim = get_param_dim_from_model_weights(model_path)
        print(f"Successfully inferred param_dim: {param_dim}")

        model_hyperparams_saved_model = {
            'hidden_dim': 1024,
            'rank': 128,
            'dropout_rate': 0.1
        }
        flow_model, device = load_flow_model(model_path, param_dim,
                                             model_hyperparams=model_hyperparams_saved_model)
        
        # --- To get 2 rows and 4 columns, set n_visualization_steps to 8 ---
        visualize_flow_trajectory(
            flow_model,
            device,
            param_dim,
            n_visualization_steps=8, # Changed to 8 for a 2x4 grid
            n_samples=200,
            generation_num_steps=100,
            clamp_val_generate=20.0,
            save_path='flow_model_trajectory_2x4.png' # New save path
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()