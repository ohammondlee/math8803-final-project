import torch

def build_connectivity(config, shape):
    t = config["type"]
    if t == "low_rank":
        return low_rank(shape, config["rank"])
    elif t == "svd_truncated":
        return svd_truncated(shape, config["rank"], config.get("base_std", 1.25), config.get("base_W0"))
    elif t == "spectral":
        return spectral_radius_init(shape, config["spectral_radius"])
    elif t == "orthogonal":
        return orthogonal_init(shape, config["alpha"])
    elif t == "random":
        return torch.randn(*shape)
    else:
        raise ValueError(f"Unknown connectivity type: {t}")

def low_rank(shape, rank):
    n, m = shape
    U = torch.randn(n, rank)
    V = torch.randn(rank, m)
    return (U @ V) / (rank ** 0.5)

def svd_truncated(shape, rank, base_std=1.25, base_W0=None):
    n, m = shape
    assert n == m, "SVD truncated assumes square matrix"
    if base_W0 is not None:
        W0 = base_W0
    else:
        W0 = base_std * torch.randn(n, n) / (n ** 0.5)
    U, S, VT = torch.linalg.svd(W0)
    new_S = S.clone()
    new_S[rank:] = 0
    W_trunc = U @ torch.diag(new_S) @ VT
    # Renormalize to same norm as W0
    W_trunc = W_trunc * (torch.norm(W0) / torch.norm(W_trunc))
    return W_trunc

def spectral_radius_init(shape, rho):
    W = torch.randn(*shape)
    eigvals = torch.linalg.eigvals(W)
    current = eigvals.abs().max()
    return (W * (rho / current)).real

def orthogonal_init(shape, alpha, scale=1.0):
    n, m = shape
    k = min(n, m)
    U, _ = torch.linalg.qr(torch.randn(n, k))
    V, _ = torch.linalg.qr(torch.randn(m, k))
    # log-normal singular values centered at 1
    log_s = alpha * torch.randn(k)
    s = torch.exp(log_s)  # ensures positivity
    s = s / s.mean()
    S = torch.diag(s)
    return scale * (U @ S @ V.T).real