import torch
import analyticSolutions


def sample_observation_points(N_obs, alpha_true, T, device, t_min=0.02, t_max=0.30):
    x_obs = torch.rand(N_obs, 1, device=device)
    t_obs = t_min + (t_max - t_min) * torch.rand(N_obs, 1, device=device)

    with torch.no_grad():
        u_obs = analyticSolutions.u(alpha_true, x_obs, t_obs)

    return x_obs.detach(), t_obs.detach(), u_obs.detach()