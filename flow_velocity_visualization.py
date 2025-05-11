import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn
# import torch.nn.functional as F # Not directly used in FlowMatchingModel in this version
from sklearn.decomposition import PCA
import traceback # For debugging if needed

# --- FlowMatchingModel class definition ---
class FlowMatchingModel(nn.Module):
    def __init__(self, param_dim, hidden_dim=1024, rank=128, dropout_rate=0.1):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64)
        )
        self.input_norm = nn.LayerNorm(param_dim)
        self.low_rank_proj = nn.Linear(param_dim, rank)

        self.main_net = nn.Sequential(
            nn.Linear(rank + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, rank)
        )
        self.inv_proj = nn.Linear(rank, param_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, z, t):
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
        predictions_for_viz = []

        for step in range(num_steps):
            t_val_model = 1.0 - step / num_steps
            t_tensor = torch.ones(z.size(0), 1, device=z.device) * t_val_model
            pred = self(z, t_tensor)
            predictions_for_viz.append(pred.cpu().clone())
            delta_z = pred * dt_step
            z = z + delta_z
            z = torch.clamp(z, -clamp_val, clamp_val)
            z = torch.nan_to_num(z, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
            trajectory.append(z.cpu().clone())
        return trajectory, predictions_for_viz

# --- load_flow_model and get_param_dim_from_model_weights functions ---
def load_flow_model(model_path, param_dim, model_hyperparams=None):
    if model_hyperparams is None:
        model_hyperparams = {'hidden_dim': 1024, 'rank': 128, 'dropout_rate': 0.1}
        print(f"Warning: Using default model hyperparameters for loading: {model_hyperparams}")
        print("Ensure these match the saved model's architecture.")

    model = FlowMatchingModel(param_dim, **model_hyperparams)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model to: {device}")
    state_dict = torch.load(model_path, map_location=device)
    clean_state_dict = {}
    if isinstance(state_dict, dict) and 'state_dict' in state_dict: # Common pattern for PL checkpoints
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model_state_dict' in state_dict: # Another common pattern
        state_dict = state_dict['model_state_dict']

    # Remove "flow_model." prefix if present (e.g. from some saving methods)
    for k, v in state_dict.items():
        if k.startswith("flow_model."):
            clean_state_dict[k[len("flow_model."):]] = v
        else:
            clean_state_dict[k] = v
    state_dict = clean_state_dict

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Strict loading failed: {e}. Attempting non-strict loading.")
        model.load_state_dict(state_dict, strict=False) # Allow partial loads or mismatches
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
    # Try to infer param_dim from keys related to input layer or normalization
    possible_keys_prefixes = [
        "input_norm.weight", "low_rank_proj.weight", # From current model
        "flow_model.input_norm.weight", "flow_model.low_rank_proj.weight" # If wrapped
    ]
    for key_suffix in ["input_norm.weight", "low_rank_proj.weight"]: # Add more if needed
        for prefix in ["", "flow_model."]:
            full_key = prefix + key_suffix
            if full_key in state_dict:
                if key_suffix == "input_norm.weight":
                    param_dim = state_dict[full_key].shape[0]
                elif key_suffix == "low_rank_proj.weight": # input features to this linear layer
                    param_dim = state_dict[full_key].shape[1]

                if param_dim is not None:
                    print(f"Found key '{full_key}', inferred param_dim: {param_dim}")
                    return param_dim

    if param_dim is None:
        # Fallback if specific keys aren't found, inspect all keys
        for key in state_dict.keys():
            if key.endswith("input_norm.weight"):
                param_dim = state_dict[key].shape[0]
                print(f"Found key '{key}' via endswith, param_dim: {param_dim}")
                break
            elif key.endswith("low_rank_proj.weight") and len(state_dict[key].shape) > 1:
                param_dim = state_dict[key].shape[1]
                print(f"Found key '{key}' via endswith (from low_rank_proj), param_dim: {param_dim}")
                break
    if param_dim is None:
        raise KeyError(f"Could not automatically infer 'param_dim'. Inspected keys like 'input_norm.weight', 'low_rank_proj.weight'. Available keys (first 10): {list(state_dict.keys())[:10]}...")
    return param_dim


def visualize_flow_trajectory(flow_model, device, param_dim, n_visualization_steps=8, n_samples=300,
                              generation_num_steps=100, clamp_val_generate=20.0,
                              save_path='flow_model_trajectory_neurips_small_arrows.png', # 新文件名
                              arrow_scale=400.0, # 大幅增加 scale 使箭头变短
                              arrow_headwidth=2.5, # 减小箭头头部宽度
                              arrow_headlength=4,  # 减小箭头头部长度
                              arrow_shaftwidth_factor=0.0015, # 进一步减小箭头杆的宽度
                              dpi=300):
    """
    Visualizes the flow trajectory with smaller, complete velocity arrows.
    """
    if n_visualization_steps <= 0:
        print("n_visualization_steps must be positive.")
        return
    if n_visualization_steps > generation_num_steps:
        print(f"Warning: n_visualization_steps ({n_visualization_steps}) > generation_num_steps ({generation_num_steps}). "
              f"Setting n_visualization_steps to {generation_num_steps}.")
        n_visualization_steps = generation_num_steps
    if n_visualization_steps == 0 and generation_num_steps > 0:
        n_visualization_steps = 1

    initial_noise = torch.randn(n_samples, param_dim, device=device)
    full_trajectory_cpu, full_predictions_cpu = flow_model.generate(
        initial_noise,
        num_steps=generation_num_steps,
        clamp_val=clamp_val_generate
    )

    if n_visualization_steps == 1 and generation_num_steps > 0:
        indices_to_sample = np.array([0], dtype=int)
    elif n_visualization_steps > 1:
        indices_to_sample = np.linspace(0, generation_num_steps - 1, n_visualization_steps, dtype=int)
    else:
        print("No steps to visualize.")
        if os.path.exists(save_path):
            try: os.remove(save_path)
            except OSError: pass
        return

    zs_for_visualization_cpu = [full_trajectory_cpu[i] for i in indices_to_sample]
    vs_for_visualization_cpu = [full_predictions_cpu[i] for i in indices_to_sample]

    all_z_for_pca_np = np.concatenate([z.numpy() for z in zs_for_visualization_cpu], axis=0)
    all_v_for_pca_np = np.concatenate([v.numpy() for v in vs_for_visualization_cpu], axis=0)

    if param_dim < 2:
        print("param_dim < 2, cannot perform PCA. Aborting visualization.")
        return

    pca = PCA(n_components=2)
    all_z_2d, all_v_2d = None, None
    try:
        pca.fit(all_z_for_pca_np)
        all_z_2d = pca.transform(all_z_for_pca_np)
        all_v_2d = pca.transform(all_v_for_pca_np)
    except Exception as e:
        print(f"PCA failed: {e}"); traceback.print_exc()
        non_zero_var_cols = np.var(all_z_for_pca_np, axis=0) > 1e-9
        if np.sum(non_zero_var_cols) >= 2:
            print("PCA failed, attempting manual selection of first 2 non-zero variance components.")
            active_cols = np.where(non_zero_var_cols)[0][:2]
            all_z_2d = all_z_for_pca_np[:, active_cols]
            all_v_2d = all_v_for_pca_np[:, active_cols]
        else:
            print("PCA failed: Not enough non-zero variance components."); return

    zs_2d_viz = [all_z_2d[i*n_samples:(i+1)*n_samples] for i in range(len(indices_to_sample))]
    vs_2d_viz = [all_v_2d[i*n_samples:(i+1)*n_samples] for i in range(len(indices_to_sample))]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9, # Slightly smaller base font for more space
        'axes.titlesize': 11,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'axes.edgecolor': '#DDDDDD',
        'grid.color': '#EAEAEA',
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
    })

    scatter_color = '#A0CBE8' # Kept the light blue for points
    scatter_alpha = 0.5
    arrow_color = '#5A8FBB' # Kept the arrow color
    arrow_alpha = 0.75 # Slightly more opaque to ensure visibility if very small

    nrows = 2
    ncols = (len(indices_to_sample) + nrows - 1) // nrows
    fig_width = 3.2 * ncols  # Slightly reduced width per plot if needed
    fig_height = 3.2 * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    min_x_overall, min_y_overall = np.min(all_z_2d, axis=0) if all_z_2d.size > 0 else (0,0)
    max_x_overall, max_y_overall = np.max(all_z_2d, axis=0) if all_z_2d.size > 0 else (0,0)
    # Margins might need to be adjusted if arrows are very short and don't go out of bounds
    margin_factor = 0.1 # Reduce margin if arrows are now well contained
    margin_x = (max_x_overall - min_x_overall) * margin_factor if (max_x_overall - min_x_overall) > 1e-6 else 0.1
    margin_y = (max_y_overall - min_y_overall) * margin_factor if (max_y_overall - min_y_overall) > 1e-6 else 0.1
    final_xlim = (min_x_overall - margin_x, max_x_overall + margin_x)
    final_ylim = (min_y_overall - margin_y, max_y_overall + margin_y)

    for i, sampled_idx in enumerate(indices_to_sample):
        ax = axes_flat[i]
        z2d_step = zs_2d_viz[i]
        v2d_step = vs_2d_viz[i]

        ax.scatter(z2d_step[:, 0], z2d_step[:, 1], s=10, alpha=scatter_alpha, color=scatter_color, zorder=2, edgecolor='none')

        if v2d_step.shape[0] > 0:
            # Calculate dynamic arrow shaft width
            # This factor might need to be smaller for very small arrows
            dynamic_arrow_shaft_width = arrow_shaftwidth_factor * (50 / max(10, np.sqrt(n_samples)))

            ax.quiver(z2d_step[:, 0], z2d_step[:, 1], v2d_step[:, 0], v2d_step[:, 1],
                        color=arrow_color, alpha=arrow_alpha, zorder=3, # Arrows on top of grid, below points if zorder adjusted
                        scale=arrow_scale,           # Key parameter for arrow length
                        width=dynamic_arrow_shaft_width, # Shaft width
                        headwidth=arrow_headwidth,   # Head width as multiple of shaft width
                        headlength=arrow_headlength, # Head length as multiple of shaft width
                        minlength=0.1,             # Don't draw arrows for very tiny vectors (in pixels)
                        minshaft=0.5,                # Minimum shaft length (in head lengths)
                        pivot='tail'                 # Arrow starts at the data point
                        )

        time_progress = sampled_idx / generation_num_steps if generation_num_steps > 0 else 0
        ax.set_title(f't = {time_progress:.2f}', color='#333333')

        ax.set_xlim(final_xlim)
        ax.set_ylim(final_ylim)

        ax.tick_params(axis='both', which='major', colors='#555555', direction='out', length=3, width=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.6)
        ax.set_facecolor('white')

    for i in range(len(indices_to_sample), nrows * ncols):
        axes_flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97], pad=0.3, h_pad=0.8, w_pad=0.3) # Adjust padding
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved flow trajectory visualization to {save_path}")
    if pca and hasattr(pca, 'explained_variance_ratio_'):
        print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}, Sum: {np.sum(pca.explained_variance_ratio_)}")
    plt.style.use('default')


# --- Main execution block ---
if __name__ == '__main__':
    # Ensure this path points to your actual model file
    model_path = '/home/xiongzc/Desktop/pFedFDA-main/eval_model/fedavg_cifar10_dir01_CNN_global_flow_weights_round120.pth'
    # model_path = 'dummy_model_weights/dummy_flow_model.pth' # For testing without a real model

    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        print("A dummy model will be created for testing the visualization script.")
        dummy_param_dim = 256
        dummy_model_for_save = FlowMatchingModel(param_dim=dummy_param_dim, hidden_dim=512, rank=64)
        dummy_state_dict = {}
        for name, param in dummy_model_for_save.named_parameters():
            dummy_state_dict[name] = torch.randn_like(param)
        temp_model_dir = 'dummy_model_weights'
        os.makedirs(temp_model_dir, exist_ok=True)
        temp_model_path = os.path.join(temp_model_dir, 'dummy_flow_model.pth')
        torch.save(dummy_state_dict, temp_model_path)
        print(f"Dummy model saved to {temp_model_path}")
        model_path = temp_model_path

    try:
        param_dim = get_param_dim_from_model_weights(model_path)
        print(f"Successfully inferred param_dim: {param_dim}")

        model_hyperparams_saved_model = {
            'hidden_dim': 1024, 'rank': 128, 'dropout_rate': 0.1
        }
        if "dummy_flow_model.pth" in model_path:
            print("Using dummy model. Hyperparameters in 'model_hyperparams_saved_model' might not match exactly.")
            model_hyperparams_saved_model = {'hidden_dim': 512, 'rank': 64, 'dropout_rate': 0.1}

        flow_model, device = load_flow_model(model_path, param_dim, model_hyperparams=model_hyperparams_saved_model)
        print(f"Parameter dimension for visualization: {param_dim}")

        if param_dim < 2 :
             print("Error: param_dim is less than 2. Cannot perform 2D PCA visualization.")
        else:
            visualize_flow_trajectory(
                flow_model,
                device,
                param_dim,
                n_visualization_steps=8,
                n_samples=50, # Keep n_samples relatively low for clarity with small arrows
                generation_num_steps=100,
                clamp_val_generate=20.0,
                save_path='flow_model_trajectory_neurips_small_arrows.png',
                arrow_scale=400.0,  # << INCREASED (larger scale = shorter arrows)
                arrow_headwidth=3.0, # << DECREASED
                arrow_headlength=4.5, # << DECREASED
                arrow_shaftwidth_factor=0.0018, # << DECREASED (adjust for thinner shafts)
                dpi=300
            )
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please check the path.")
    except KeyError as e:
        print(f"KeyError: {e}. This might be due to mismatched model architecture or keys in the state_dict.")
    except Exception as e:
        print(f"An unexpected error occurred in the main execution block: {e}")
        traceback.print_exc()