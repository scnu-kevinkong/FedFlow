import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd # Using pandas can simplify data handling

# --- NIPS Style Configuration ---
plt.rcParams.update({
    # "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 10,
    "figure.titlesize": 12,
    "axes.linewidth": 0.8
})
# Uncomment if using LaTeX
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "font.size": 10,
#     "axes.labelsize": 10,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "legend.fontsize": 10,
#     "figure.titlesize": 12,
# })


# --- 1. Load and Parse Data ---
file_path = '/home/xiongzc/Desktop/pFedFDA-main/data/partition/tinyimagenet_c100_dir05.json' # cite: 1

try:
    with open(file_path, 'r') as f:
        data = json.load(f) # cite: 1
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

num_clients = data.get('num_clients', 100) # cite: 1
num_classes = data.get('num_classes', 200) # cite: 1
label_distribution = data.get('label_distribution', []) # cite: 1

# Prepare data for plotting - using a list of dictionaries first
plot_data = []
for client_id, client_data in enumerate(label_distribution): # cite: 1
    for class_info in client_data: # cite: 1
        if len(class_info) == 2: # cite: 1
            class_index, count = class_info # cite: 1
            if count > 0: # Only plot if count is greater than 0
                plot_data.append({
                    'client': client_id,
                    'class': class_index,
                    'count': count
                })

# Convert to Pandas DataFrame for easier handling (optional but convenient)
df = pd.DataFrame(plot_data)

# --- 2. Create Enhanced Scatter Plot (Bubble Grid) ---
fig, ax = plt.subplots(figsize=(10, 4)) # Adjust figsize

# --- Size Scaling ---
# Adjust the multiplier 'k' to get visually appealing bubble sizes
# Maybe scale non-linearly if counts vary extremely, e.g., using log
# Simple linear scaling for now:
min_size = 5 # Minimum size for a bubble
max_size = 500 # Maximum size for a bubble (adjust as needed)
# Scale sizes: (count / max_count) * (max_size - min_size) + min_size might work
# Or simpler: scale proportionally, adjust k
k = 0.8 # Size multiplier factor - adjust this!
sizes = df['count'] * k

# --- Color Mapping ---
# Choose a colormap (e.g., viridis, plasma, Blues, Reds)
cmap_name = 'viridis'
cmap = plt.get_cmap(cmap_name)
norm = mcolors.Normalize(vmin=df['count'].min(), vmax=df['count'].max()) # Normalize counts to 0-1 for color map
# For logarithmic color scale (if counts vary widely):
# norm = mcolors.LogNorm(vmin=df[df['count'] > 0]['count'].min(), vmax=df['count'].max())


scatter = ax.scatter(
    df['client'],
    df['class'],
    s=sizes,           # Size based on count
    c=df['count'],     # Color based on count
    cmap=cmap,         # Colormap to use
    norm=norm,         # Normalization for color scale
    alpha=0.7,         # Transparency
    edgecolors='grey', # Add subtle edge to bubbles
    linewidth=0.5
)

# --- 3. Customize Plot Style (NIPS-like) ---
ax.set_xlabel('Client Index')
ax.set_ylabel('Class Index')
# title_str = f'Data Distribution across Clients (CIFAR-10, Dirichlet $\\alpha=0.5$)' # cite: 1
# if plt.rcParams['text.usetex']:
#      ax.set_title(title_str)
# else:
#      ax.set_title(title_str.replace('\\alpha', 'alpha'))


# Set axis limits and ticks
ax.set_xlim(-1, num_clients) # Start slightly before 0
ax.set_ylim(-0.5, num_classes - 0.5)

# --- Key Change: Reduce Y Ticks ---
tick_interval_y = 10 # Adjust based on num_classes (e.g., 10 for 100, 20 for 200)
if num_classes > 50: # Only reduce ticks if there are many classes
    ax.set_yticks(np.arange(0, num_classes, tick_interval_y))
else:
    ax.set_yticks(np.arange(num_classes)) # Keep all ticks for fewer classes

ax.set_xticks(np.arange(0, num_clients, 10))
ax.tick_params(axis='x', rotation=0)

# Add a color bar
cbar = fig.colorbar(scatter, ax=ax, label='Number of Samples', pad=0.02)

# Add grid (adjust based on new ticks)
ax.grid(True, which='major', linestyle=':', linewidth=0.5, color='lightgrey')
# Optional: Turn off minor grid if it looks too busy with sparse major ticks
# ax.minorticks_off()
ax.set_axisbelow(True)

# --- 4. Show/Save Plot ---
plt.tight_layout(pad=1.0)
plt.show()

# --- (Optional) Save Plot ---
fig.savefig("tinyimagenet_dirichlet_alpha_0.5_bubble_grid_nips.pdf", bbox_inches='tight')
# fig.savefig("dirichlet_alpha_0.5_bubble_grid_nips.png", dpi=300, bbox_inches='tight')