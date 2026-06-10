import torch

def sample_interior_points(N, T=2.0, device="cpu"):
    x = torch.rand((N, 1), device=device, requires_grad=True)
    t = T * torch.rand((N, 1), device=device, requires_grad=True)

    return x, t


def sample_boundary_points_u(N, T=2.0, device="cpu"):
    N_left = N // 2
    N_right = N - N_left

    t_left = T * torch.rand((N_left, 1), device=device)
    x_left = torch.zeros((N_left, 1), device=device)

    t_right = T * torch.rand((N_right, 1), device=device)
    x_right = torch.ones((N_right, 1), device=device)

    x_boundary = torch.cat([x_left, x_right], dim=0)
    t_boundary = torch.cat([t_left, t_right], dim=0)

    return x_boundary, t_boundary

def sample_initial_points_u(N, device="cpu"):
    x_initial = torch.rand((N, 1), device=device)
    t_initial = torch.zeros((N, 1), device=device)

    return x_initial, t_initial
