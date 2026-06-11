import torch

def observation_loss(model , x, t, u_obs):
    u_pred = model(x, t)
    return torch.mean((u_pred - u_obs) ** 2)

def boundary_loss(model, x_b, t_b):
    u_pred = model(x_b, t_b)
    return torch.mean(u_pred ** 2)


def initial_loss(model, x_ic, t_ic):
    u_pred = model(x_ic, t_ic)
    u_true = torch.sin(torch.pi * x_ic)
    return torch.mean((u_pred - u_true) ** 2)

def pde_loss(model, alpha, x, t):
    """
    Important:
    x and t are cloned and detached so that every PDE loss call builds
    a fresh autograd graph. This avoids backward-through-graph errors
    when fixed collocation points are reused across epochs.
    """

    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = model(x, t)

    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0]

    residual = u_t - alpha * u_xx

    return torch.mean(residual ** 2)