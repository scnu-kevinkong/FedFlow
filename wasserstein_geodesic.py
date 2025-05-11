import torch

def wasserstein_geodesic(t: float, 
                         Sigma1: torch.Tensor, 
                         Sigma2: torch.Tensor, 
                         epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute the Wasserstein geodesic interpolation between SPD matrices
    
    Args:
        t:         Interpolation parameter (float or tensor in [0, 1])
        Sigma1:    Batch of SPD matrices (..., d, d)
        Sigma2:    Batch of SPD matrices (..., d, d)
        epsilon:   Regularization term for numerical stability
    
    Returns:
        Sigma_t:   Interpolated SPD matrix at time t (..., d, d)
    """
    # Add regularization to ensure positive definiteness
    Sigma1 = Sigma1 + epsilon * torch.eye(Sigma1.shape[-1], device=Sigma1.device)
    Sigma2 = Sigma2 + epsilon * torch.eye(Sigma2.shape[-1], device=Sigma2.device)

    # Compute matrix square roots using SVD for stability
    def matrix_sqrt(x):
        U, S, V = torch.svd(x)
        return U @ torch.diag_embed(S.sqrt()) @ V.transpose(-1, -2)

    # 1. Compute sqrt(Sigma1)
    sqrt_Sigma1 = matrix_sqrt(Sigma1)

    # 2. Compute inverse sqrt using Cholesky decomposition (more stable)
    try:
        L = torch.linalg.cholesky(sqrt_Sigma1)
    except RuntimeError:
        # Fallback to SVD if Cholesky fails
        U, S, V = torch.svd(sqrt_Sigma1)
        inv_sqrt_Sigma1 = V @ torch.diag_embed(1.0 / S.sqrt()) @ U.transpose(-1, -2)
    else:
        inv_sqrt_Sigma1 = torch.cholesky_inverse(L)

    # 3. Compute M = inv_sqrt_Sigma1 @ Sigma2 @ inv_sqrt_Sigma1
    M = inv_sqrt_Sigma1 @ Sigma2 @ inv_sqrt_Sigma1

    # 4. Eigen decomposition of M (symmetric)
    eigenvalues, eigenvectors = torch.linalg.eigh(M)

    # 5. Compute M^t using eigenvalue decomposition
    Mt = eigenvectors @ torch.diag_embed(eigenvalues ** t) @ eigenvectors.transpose(-1, -2)

    # 6. Reconstruct Sigma(t)
    Sigma_t = sqrt_Sigma1 @ Mt @ sqrt_Sigma1

    return Sigma_t

# ----------------------
# 验证与使用示例
# ----------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # 生成两个随机的SPD矩阵
    def random_spd(n, batch_size=None):
        shape = (n, n) if batch_size is None else (batch_size, n, n)
        A = torch.randn(shape)
        return A @ A.transpose(-1, -2) + 1e-3 * torch.eye(n)

    Sigma1 = random_spd(3)
    Sigma2 = random_spd(3)

    # 验证端点
    print("t=0 误差:", torch.norm(wasserstein_geodesic(0.0, Sigma1, Sigma2) - Sigma1).item())
    print("t=1 误差:", torch.norm(wasserstein_geodesic(1.0, Sigma1, Sigma2) - Sigma2).item())

    # 验证插值路径的正定性
    Sigma_half = wasserstein_geodesic(0.5, Sigma1, Sigma2)
    eigenvalues = torch.linalg.eigvalsh(Sigma_half)
    print("中间点特征值:", eigenvalues)
    print("是否正定:", torch.all(eigenvalues > 0).item())

    # 可视化测地线路径
    t_values = torch.linspace(0, 1, 5)
    for t in t_values:
        Sigma_t = wasserstein_geodesic(t, Sigma1, Sigma2)
        print(f"\nt={t:.1f}时的矩阵范数:", torch.norm(Sigma_t).item())