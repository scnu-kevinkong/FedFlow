import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
from sklearn.manifold import TSNE
import seaborn as sns # For better color palettes and plotting
import glob # For finding client model files
import matplotlib # For custom color palette

# --- BEGIN: Class definitions ---
class CIFARNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(CIFARNet, self).__init__()
        self.input_shape = (in_channels, 32, 32)
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.flat_size = 64 * 3 * 3
        self.linear = nn.Linear(self.flat_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.D = 128
        self.cls = num_classes

    def forward(self, x, return_feat=False, return_logits=True):
        feat = self.pool(F.leaky_relu(self.conv1(x)))
        feat = self.pool(F.leaky_relu(self.conv2(feat)))
        feat = self.pool(F.leaky_relu(self.conv3(feat)))
        feat = feat.view(-1, self.flat_size)
        feat_before_fc = F.leaky_relu(self.linear(feat))
        out = self.fc(feat_before_fc)

        if return_feat:
            if return_logits:
                return feat_before_fc, out
            else:
                return feat_before_fc, F.softmax(out, dim=1)
        if return_logits:
            return out
        else:
            return F.softmax(out, dim=1)

class FlowMatchingModel(nn.Module):
    def __init__(self, param_dim, hidden_dim=1024, rank=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 64)
        )
        self.input_norm = nn.LayerNorm(param_dim)
        self.low_rank_proj = nn.Linear(param_dim, rank)
        self.main_net = nn.Sequential(
            nn.Linear(rank + 64, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
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
        if t.dim() == 1: t = t.unsqueeze(-1) if t.numel() == z.size(0) else t.view(z.size(0), 1)
        t_emb = self.time_embed(t)
        z_proj = self.low_rank_proj(z)
        x = torch.cat([z_proj, t_emb], dim=-1)
        return self.inv_proj(self.main_net(x))

    @torch.no_grad()
    def generate(self, init_params, num_steps=100):
        if init_params.dim() == 1:
            z = init_params.unsqueeze(0).clone(); input_was_1d = True
        else:
            z = init_params.clone(); input_was_1d = False
        dt = 0.5 / num_steps; noise_decay = 0.95
        for step in range(num_steps):
            current_noise = (0.2 * (noise_decay ** step)) * torch.randn_like(z)
            t_val = torch.ones(z.size(0), 1).to(z.device) * (1.0 - step / num_steps)
            if torch.isnan(z).any():
                print(f"NaN detected at step {step} during flow generation."); return init_params.clone()
            pred = self(z + current_noise, t_val)
            pred = torch.clamp(pred, -0.5, 0.5)
            z = z + pred * dt
            z = torch.clamp(z, -3.0, 3.0)
            z = torch.nan_to_num(z, nan=0.0)
        return z.squeeze(0) if input_was_1d else z

def flatten_params(model_or_state_dict):
    state_dict = {k: v.cpu() for k, v in (model_or_state_dict.state_dict() if isinstance(model_or_state_dict, torch.nn.Module) else model_or_state_dict).items()}
    return torch.cat([p.flatten() for p in state_dict.values()])

def unflatten_params(flat_params, example_model_or_state_dict_cpu):
    ref_state_dict = {k: v.cpu() for k, v in (example_model_or_state_dict_cpu.state_dict() if isinstance(example_model_or_state_dict_cpu, torch.nn.Module) else example_model_or_state_dict_cpu).items()}
    new_state_dict = {}; pointer = 0; flat_params_cpu = flat_params.cpu()
    for name, param_ref_cpu in ref_state_dict.items():
        numel = param_ref_cpu.numel()
        if pointer + numel > flat_params_cpu.numel(): raise ValueError(f"Unflatten error for '{name}'.")
        new_state_dict[name] = flat_params_cpu[pointer:pointer+numel].view_as(param_ref_cpu).clone()
        pointer += numel
    if pointer != flat_params_cpu.numel(): raise ValueError(f"Unflatten size mismatch.")
    return new_state_dict
# --- END: Class definitions ---

# --- USER CONFIGURATION ---
# !! 请确保将此路径更改为您的实际模型保存基本目录 !!
MODEL_SAVE_BASE_DIR = "./eval_model" # 例如: "/home/user/pFedFDA/eval_model"
if not os.path.exists(MODEL_SAVE_BASE_DIR) or not os.path.isdir(MODEL_SAVE_BASE_DIR):
    print(f"警告: MODEL_SAVE_BASE_DIR ('{MODEL_SAVE_BASE_DIR}') 不存在或不是一个目录。")
    print("脚本可能会因为找不到模型文件而失败。请验证此路径。")
    # 可以选择在这里创建目录:
    # os.makedirs(MODEL_SAVE_BASE_DIR, exist_ok=True)
    # print(f"已尝试创建目录: {MODEL_SAVE_BASE_DIR}")


ALL_CLIENTS_PARAMS_SUBDIR = "" # 如果客户端模型在 MODEL_SAVE_BASE_DIR 的子目录中，请指定
ROUND_NUMBER = 200

ALL_CLIENTS_FILENAME_PATTERN_FOR_STATS = f"fedavg_cifar10_dir01_CNN_client_*_personal_model_weights_round{ROUND_NUMBER}.pth"

CLIENT_IDS_TO_ANALYZE = [0, 1] # 示例: 分析客户端 0 和客户端 1

GLOBAL_MODEL_FILENAME = f"fedavg_cifar10_dir01_CNN_global_model_weights_round{ROUND_NUMBER}.pth"
FLOW_MODEL_FILENAME = f"fedavg_cifar10_dir01_CNN_global_flow_weights_round{ROUND_NUMBER}.pth"

TARGET_CLASS_NAMES_FOR_TSNE = ['cat', 'airplane'] # 选项: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
NUM_SAMPLES_PER_CLASS_FOR_TSNE = 75

CIFAR10_CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

TSNE_PERPLEXITY = 30
TSNE_N_ITER = 2500
TSNE_LEARNING_RATE = 'auto'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Helper Functions ---
def load_cifar10_data_for_multiple_classes(num_samples_per_class, target_class_names_list):
    all_images_list = []
    all_labels_list = []
    loaded_class_indices = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    try:
        print(f"\nAttempting to load CIFAR-10 data for classes: {target_class_names_list}")
        # 尝试在本地加载数据集，如果失败则尝试下载
        try:
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
        except RuntimeError:
            print("Local CIFAR-10 dataset not found or corrupted, attempting to download...")
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        full_testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

        for target_class_name in target_class_names_list:
            target_class_idx = -1
            try:
                target_class_idx = CIFAR10_CLASS_NAMES.index(target_class_name)
            except ValueError:
                print(f"Error: Target class '{target_class_name}' not found in CIFAR10_CLASS_NAMES: {CIFAR10_CLASS_NAMES}")
                continue

            class_images_for_current_class = []
            class_labels_for_current_class = []

            image_count_for_class = 0
            for images_batch, labels_batch in full_testloader:
                for i in range(images_batch.size(0)):
                    if labels_batch[i].item() == target_class_idx:
                        class_images_for_current_class.append(images_batch[i])
                        class_labels_for_current_class.append(target_class_idx)
                        image_count_for_class +=1
                    if image_count_for_class >= num_samples_per_class:
                        break
                if image_count_for_class >= num_samples_per_class:
                    break

            if not class_images_for_current_class:
                print(f"Warning: No samples found for class '{target_class_name}' (index {target_class_idx}).")
            else:
                num_loaded_for_class = len(class_images_for_current_class)
                if num_loaded_for_class < num_samples_per_class:
                    print(f"Warning: Requested {num_samples_per_class} samples for class '{target_class_name}', but only found {num_loaded_for_class}. Using {num_loaded_for_class}.")

                all_images_list.extend(class_images_for_current_class)
                all_labels_list.extend(class_labels_for_current_class)
                if target_class_idx not in loaded_class_indices:
                     loaded_class_indices.append(target_class_idx)
                print(f"Loaded {num_loaded_for_class} samples for class: '{target_class_name}' (index {target_class_idx}).")

        if not all_images_list:
            print(f"Error: No samples loaded for any of the specified classes: {target_class_names_list}")
            return None, None, []

        images_tensor = torch.stack(all_images_list)
        labels_tensor = torch.tensor(all_labels_list, dtype=torch.long)

        print(f"Total loaded {images_tensor.shape[0]} samples for t-SNE from {len(loaded_class_indices)} classes.")
        return images_tensor, labels_tensor, sorted(list(set(loaded_class_indices)))

    except Exception as e:
        print(f"Error loading CIFAR-10 data for classes '{target_class_names_list}': {e}")
        return None, None, []


def get_inference_vectors(model, images_tensor, device, return_logits=True):
    model.eval()
    model.to(device)
    images_tensor = images_tensor.to(device)
    with torch.no_grad():
        outputs = model(images_tensor, return_logits=return_logits)
    return outputs.cpu()

def load_cifar_net_from_file(model_path, device):
    if not os.path.exists(model_path):
        print(f"ERROR: CIFARNet model file not found at {model_path}")
        return None
    model = CIFARNet(num_classes=len(CIFAR10_CLASS_NAMES)).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded CIFARNet from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading CIFARNet state_dict from {model_path}: {e}")
        return None

def load_flow_model_from_file(model_path, param_dim, device):
    if not os.path.exists(model_path):
        print(f"ERROR: FlowMatchingModel file not found at {model_path}")
        return None
    flow_model = FlowMatchingModel(param_dim=param_dim).to(device)
    try:
        flow_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded FlowMatchingModel from {model_path}")
        return flow_model
    except Exception as e:
        print(f"Error loading FlowMatchingModel state_dict from {model_path}: {e}")
        return None

def calculate_param_stats_from_clients(clients_params_dir, filename_pattern, dummy_model_cpu, device):
    all_flat_params = []
    if not os.path.isdir(clients_params_dir):
        print(f"Error: Dir for client params '{clients_params_dir}' not found."); return None, None
    client_model_files = glob.glob(os.path.join(clients_params_dir, filename_pattern))
    if not client_model_files:
        print(f"No client model files in '{clients_params_dir}' with pattern '{filename_pattern}'."); return None, None
    print(f"Found {len(client_model_files)} client model files for calculating stats.")
    if len(client_model_files) <= 1: print("Warning: <=1 client models for stats. Std may be 0/NaN.")
    loaded_count = 0
    for model_file in client_model_files:
        try:
            state_dict = torch.load(model_file, map_location='cpu')
            temp_model = CIFARNet(num_classes=len(CIFAR10_CLASS_NAMES)).cpu(); temp_model.load_state_dict(state_dict)
            all_flat_params.append(flatten_params(temp_model.state_dict())); loaded_count +=1
        except Exception as e: print(f"Warning: Could not load/flatten params from {model_file}: {e}"); continue
    if not all_flat_params or loaded_count <=1 : print("Not enough client params for robust stats."); return None, None
    print(f"Successfully loaded/flattened params from {loaded_count} clients.")
    params_tensor = torch.stack(all_flat_params, dim=0)
    calculated_mean = params_tensor.mean(dim=0)
    calculated_std = params_tensor.std(dim=0, unbiased=True) + 1e-8 if params_tensor.shape[0] > 1 else torch.zeros_like(calculated_mean) + 1e-8
    if params_tensor.shape[0] <= 1: print("Warning: Std calculated from <=1 data point.")
    print("Calculated parameter mean and std from client models.")
    return calculated_mean.to(device), calculated_std.to(device)

# --- Main Script ---
if __name__ == "__main__":
    if not os.path.isdir(MODEL_SAVE_BASE_DIR):
        print(f"CRITICAL ERROR: `MODEL_SAVE_BASE_DIR` ('{MODEL_SAVE_BASE_DIR}') is not a valid directory or does not exist. Please check the path.")
        exit()
    clients_stats_dir = os.path.join(MODEL_SAVE_BASE_DIR, ALL_CLIENTS_PARAMS_SUBDIR) if ALL_CLIENTS_PARAMS_SUBDIR else MODEL_SAVE_BASE_DIR
    if not os.path.isdir(clients_stats_dir):
        print(f"CRITICAL ERROR: Directory for client param stats '{clients_stats_dir}' is invalid or does not exist.")
        exit()

    dummy_cifar_net_cpu = CIFARNet(num_classes=len(CIFAR10_CLASS_NAMES)).cpu()
    param_dim = sum(p.numel() for p in dummy_cifar_net_cpu.parameters() if p.requires_grad)

    print(f"\nCalculating param_mean/std from client models in '{clients_stats_dir}' using '{ALL_CLIENTS_FILENAME_PATTERN_FOR_STATS}'...")
    calculated_param_mean, calculated_param_std = calculate_param_stats_from_clients(
        clients_stats_dir, ALL_CLIENTS_FILENAME_PATTERN_FOR_STATS, dummy_cifar_net_cpu, DEVICE
    )
    if calculated_param_mean is None or calculated_param_std is None: print("Failed to calc param_mean/std. No normalization for Flow Model.")
    else: print("Using calculated param_mean/std for Flow Model.")

    images, true_labels, target_class_indices_loaded = load_cifar10_data_for_multiple_classes(
        NUM_SAMPLES_PER_CLASS_FOR_TSNE, TARGET_CLASS_NAMES_FOR_TSNE
    )
    if images is None or not target_class_indices_loaded:
        print(f"Failed to load CIFAR-10 data for classes '{TARGET_CLASS_NAMES_FOR_TSNE}'. Exiting."); exit()

    all_inference_vectors = []
    all_model_types = []
    all_true_labels_repeated = []

    global_model_path = os.path.join(MODEL_SAVE_BASE_DIR, GLOBAL_MODEL_FILENAME)
    global_model = load_cifar_net_from_file(global_model_path, DEVICE)
    if global_model:
        print("\nGetting inference vectors for Global Model...")
        global_vectors = get_inference_vectors(global_model, images, DEVICE, return_logits=True)
        all_inference_vectors.append(global_vectors)
        all_model_types.extend(['Global'] * global_vectors.shape[0])
        all_true_labels_repeated.extend(true_labels.tolist())

    flow_model_path = os.path.join(MODEL_SAVE_BASE_DIR, FLOW_MODEL_FILENAME)
    flow_model_instance = None
    if os.path.exists(flow_model_path):
        print("\nLoading FlowMatchingModel...")
        flow_model_instance = load_flow_model_from_file(flow_model_path, param_dim, DEVICE)
        if flow_model_instance: flow_model_instance.eval()
    else:
        print(f"\nFlow model not found at {flow_model_path}. Cannot generate flow-calibrated models.")

    for client_id in CLIENT_IDS_TO_ANALYZE:
        print(f"\n--- Processing Client ID: {client_id} ---")
        client_model_filename = f"fedavg_cifar10_dir01_CNN_client_{client_id}_personal_model_weights_round{ROUND_NUMBER}.pth"
        client_model_path_specific = os.path.join(MODEL_SAVE_BASE_DIR, client_model_filename)

        if not os.path.exists(client_model_path_specific) and ALL_CLIENTS_PARAMS_SUBDIR:
            client_model_path_specific = os.path.join(clients_stats_dir, client_model_filename)

        client_model_to_analyze = load_cifar_net_from_file(client_model_path_specific, DEVICE)

        if client_model_to_analyze:
            print(f"Getting inference vectors for Client {client_id} Model...")
            client_vectors = get_inference_vectors(client_model_to_analyze, images, DEVICE, return_logits=True)
            all_inference_vectors.append(client_vectors)
            all_model_types.extend([f'Client {client_id}'] * client_vectors.shape[0])
            all_true_labels_repeated.extend(true_labels.tolist())

            if flow_model_instance:
                print(f"Generating parameters for Client {client_id} using Flow Model...")
                client_model_flat_params_cpu = flatten_params(client_model_to_analyze.state_dict())
                params_for_flow_gen = client_model_flat_params_cpu.clone().to(DEVICE)

                if calculated_param_mean is not None and calculated_param_std is not None:
                    if torch.any(calculated_param_std < 1e-7): print("Warning: std values close to zero in normalization.")
                    print("Normalizing client parameters before flow generation...")
                    params_for_flow_gen = (params_for_flow_gen - calculated_param_mean) / (calculated_param_std)
                else: print("Skipping normalization for Flow Model input.")

                generated_params_flat_device = flow_model_instance.generate(params_for_flow_gen)

                if torch.isnan(generated_params_flat_device).any():
                    print(f"ERROR: NaN in params generated by Flow Model for Client {client_id}.")
                else:
                    if calculated_param_mean is not None and calculated_param_std is not None:
                        print("Denormalizing generated parameters...")
                        generated_params_flat_device = generated_params_flat_device * calculated_param_std + calculated_param_mean
                    else: print("Skipping denormalization for Flow Model output.")

                    flow_gen_state_dict_cpu = unflatten_params(generated_params_flat_device.cpu(), dummy_cifar_net_cpu)
                    flow_calibrated_model = CIFARNet(num_classes=len(CIFAR10_CLASS_NAMES)).to(DEVICE)
                    flow_calibrated_model.load_state_dict(flow_gen_state_dict_cpu)

                    print(f"Getting inference vectors for Flow-Generated Client {client_id} Model...")
                    flow_gen_vectors = get_inference_vectors(flow_calibrated_model, images, DEVICE, return_logits=True)

                    if torch.isnan(flow_gen_vectors).any():
                        print(f"ERROR: NaN in inference from Flow-Generated model for Client {client_id}.")
                    else:
                        all_inference_vectors.append(flow_gen_vectors)
                        all_model_types.extend([f'Client {client_id} (Flow-Gen)'] * flow_gen_vectors.shape[0])
                        all_true_labels_repeated.extend(true_labels.tolist())
        else:
            print(f"Could not load client model {client_id} from {client_model_path_specific}. Skipping.")

    if not all_inference_vectors: print("\nNo inference vectors collected. Cannot perform t-SNE. Exiting."); exit()

    print("\nConcatenating all inference vectors...")
    try:
        final_vectors_torch = torch.cat(all_inference_vectors, dim=0)
        if torch.isnan(final_vectors_torch).any():
            print("ERROR: NaN values in final_vectors_torch before t-SNE. Aborting.")
            nan_indices = torch.isnan(final_vectors_torch).any(dim=1); nan_model_types = np.array(all_model_types)[nan_indices.cpu().numpy()]
            print(f"Model types potentially contributing to NaNs: {np.unique(nan_model_types)}"); exit()
        final_vectors_np = final_vectors_torch.numpy()
    except RuntimeError as e: print(f"Error during torch.cat: {e}"); exit()

    n_samples_for_tsne = final_vectors_np.shape[0]
    current_perplexity = TSNE_PERPLEXITY
    if n_samples_for_tsne <= current_perplexity:
        new_perplexity = max(5, n_samples_for_tsne - 1)
        print(f"Warning: Number of samples ({n_samples_for_tsne}) is less than or equal to perplexity ({current_perplexity}). Adjusting perplexity to {new_perplexity}.")
        current_perplexity = new_perplexity

    print(f"Performing t-SNE on {n_samples_for_tsne} vectors (perplexity: {current_perplexity})...")

    import sklearn
    tsne_lr = TSNE_LEARNING_RATE
    # sklearn < 1.1 默认为 200.0，'auto' 是在 1.1 版本引入的
    if sklearn.__version__ < '1.1' and TSNE_LEARNING_RATE == 'auto':
        print("scikit-learn version is < 1.1, using t-SNE learning_rate=200.0 as 'auto' is not supported for perplexity method.")
        tsne_lr = 200.0 # 显式设置，以避免早期版本的警告或错误
    elif sklearn.__version__ >= '1.1' and TSNE_LEARNING_RATE == 'auto':
         # 对于 1.1+, 'auto' 会根据 n_samples 自动选择 PCA 初始化 ('pca') 时为 max(N / 12, 200)，否则为 200
         pass # 'auto' is fine

    tsne = TSNE(n_components=2, random_state=42, perplexity=current_perplexity,
                max_iter=TSNE_N_ITER, learning_rate=tsne_lr, init='pca', metric='cosine')
    try:
        tsne_results = tsne.fit_transform(final_vectors_np)
    except ValueError as e:
        print(f"ValueError during t-SNE: {e}. NaNs: {np.isnan(final_vectors_np).any()}, Infs: {np.isinf(final_vectors_np).any()}"); exit()
    print("t-SNE completed.")

    print("\nPlotting t-SNE results...")
    plt.figure(figsize=(16, 12))

    unique_model_types = sorted(list(set(all_model_types)))
    markers_list = ['o', 's', '^', 'X', 'D', 'P', '*', 'v', '<', '>']
    model_type_to_marker = {mtype: markers_list[i % len(markers_list)] for i, mtype in enumerate(unique_model_types)}

    num_total_cifar_classes = len(CIFAR10_CLASS_NAMES)
    try:
        palette_colors = list(matplotlib.colormaps['tab10'].colors[:num_total_cifar_classes])
        if num_total_cifar_classes > 2 and 2 < len(palette_colors) : palette_colors[2] = (0.0, 0.5, 0.5) # Dark Teal
        palette = sns.color_palette(palette_colors)
    except Exception as e:
        print(f"Failed to create custom tab10 palette, defaulting to 'plasma'. Error: {e}")
        palette = sns.color_palette("plasma", num_total_cifar_classes)

    class_to_color = {i: palette[i % len(palette)] for i in range(num_total_cifar_classes)}

    for i in range(tsne_results.shape[0]):
        model_type_label = all_model_types[i]
        true_label_idx = all_true_labels_repeated[i]
        point_color = class_to_color.get(true_label_idx, 'black')
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1],
                    color=point_color,
                    marker=model_type_to_marker.get(model_type_label, 'x'),
                    s=50, alpha=0.7)

    # --- MODIFIED LEGEND SETTINGS ---
    legend_font_size = 14        # 增大字体
    legend_title_font_size = 14  # 增大标题字体
    # 调整 bbox_to_anchor 的 x 值，使其更靠近图像
    # 初始值可以尝试比之前的小一些，例如 1.15-1.2 区域
    # 您可能需要根据实际出图效果微调这些值
    # bbox_x_offset_model = 1.0
    bbox_x_offset_model = 1.0
    bbox_x_offset_class = 0.155 # 可以尝试让两个 legend 的 x 偏移相同或略有不同

    if unique_model_types:
        legend_elements_model_type = [Line2D([0], [0], marker=model_type_to_marker.get(mtype,'x'), color='w',
                                             label=mtype, markerfacecolor='grey', markersize=10)
                                      for mtype in unique_model_types]
        legend1 = plt.legend(handles=legend_elements_model_type,
                             title="Model Type",
                             loc="upper right",
                             bbox_to_anchor=(bbox_x_offset_model, 1), # 调整位置
                             fontsize=legend_font_size,               # 调整字体
                             title_fontsize=legend_title_font_size)   # 调整标题字体
        plt.gca().add_artist(legend1)

    legend_elements_class = []
    for class_idx in target_class_indices_loaded:
        class_name = CIFAR10_CLASS_NAMES[class_idx]
        color = class_to_color.get(class_idx, 'black')
        legend_elements_class.append(Line2D([0], [0], marker='o', color='w',
                                           label=f"{class_name}",
                                           markerfacecolor=color, markersize=10))

    if legend_elements_class:
        # 注意：如果同时显示两个图例，第二个图例的 plt.legend() 会覆盖第一个，除非第一个被用 plt.gca().add_artist(legend1) 添加
        plt.legend(handles=legend_elements_class,
                   title="Displayed Classes",
                   loc="lower right", # 通常放在不同的位置以避免重叠
                   bbox_to_anchor=(bbox_x_offset_class, 0), # 如果有第一个图例，调整y偏移避免重叠
                   fontsize=legend_font_size,
                   title_fontsize=legend_title_font_size)
    # --- END OF MODIFIED LEGEND SETTINGS ---

    clients_str = "_".join(map(str, CLIENT_IDS_TO_ANALYZE))
    clients_title_str = ", ".join(map(str, CLIENT_IDS_TO_ANALYZE))
    displayed_class_names_str = "_".join(TARGET_CLASS_NAMES_FOR_TSNE).lower().replace(' ','_')
    displayed_class_names_title_str = ", ".join(class_name.upper() for class_name in TARGET_CLASS_NAMES_FOR_TSNE)

    # plt.title(f't-SNE: Classes: {displayed_class_names_title_str} - R{ROUND_NUMBER}\nGlobal & Clients: {clients_title_str}', fontsize=16)
    # plt.xlabel("t-SNE Component 1", fontsize=14)
    # plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 调整 tight_layout 的 rect 参数，给图例留出适当空间
    # rect 的第三个参数 (right) 控制主绘图区域的右边界
    # 如果 bbox_to_anchor 的 x 值减小了 (图例向左移)，可以适当增大此 rect 的 right 值
    # 例如，从 0.85 增加到 0.88 或 0.90，给主图更多空间
    plt.tight_layout(rect=[0, 0, 0.88, 0.96]) # (left, bottom, right, top); 调整 right 和 top

    plot_filename = f"tsne_logits_r{ROUND_NUMBER}_clients_{clients_str}_classes_{displayed_class_names_str}.pdf"
    plt.savefig(plot_filename, bbox_inches='tight', dpi=800) # bbox_inches='tight' 确保图例不会被裁剪
    print(f"\nt-SNE plot saved as {plot_filename}")
    plt.show()