import torch
import analyticSolutions

def residue(alpha, u, x, t):
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    return u_t-alpha*u_xx

def pde_loss(model, alpha, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    r = residue(alpha, u, x, t)

    loss = torch.mean(r**2)

    return loss

def data_loss(model, alpha, x, t):
    u_pred = model(x, t)
    u_true = analyticSolutions.u(alpha, x, t)

    return torch.mean((u_pred - u_true) ** 2)


def total_loss(model, alpha,  initialx, boundaryx, interiorx, initialt, boundaryt, interiort):
    return 30*pde_loss(model,alpha, interiorx, interiort) + data_loss(model, alpha, initialx, initialt) +10*data_loss(model, alpha, boundaryx, boundaryt)