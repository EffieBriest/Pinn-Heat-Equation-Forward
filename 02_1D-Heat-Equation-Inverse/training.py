import torch
import sys
from pathlib import Path
from tqdm.auto import tqdm

# Add forward project path only for shared forward utilities
forward_project_path = Path(__file__).resolve().parents[1] / "01_1D-Heat-Equation-Forward"
sys.path.append(str(forward_project_path))

import modelInversePinn
import lossesInversePinn
import samplingInversePinn
import sampling
import plot
import plotInversePinn


# -----------------------------
# Configuration
# -----------------------------


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(42)

T = 2.0

ADAM_EPOCHS = 1000
USE_LBFGS = True
LBFGS_MAX_ITER = 1000

N_INTERIOR = 2000
N_BOUNDARY = 200
N_INITIAL = 200
N_OBS = 500

LAMBDA_PDE = 5.0
LAMBDA_BC = 5.0
LAMBDA_IC = 5.0
LAMBDA_OBS = 5.0

alpha_true = 1.0

LR_NET = 2e-4
LR_ALPHA = 5e-3

T_OBS_MIN = 0.02
T_OBS_MAX = 0.30


# -----------------------------
# Model and trainable parameter
# -----------------------------

pinn = modelInversePinn.InversePINN().to(device)



# -----------------------------
# Fixed training data
# -----------------------------

x_obs, t_obs, u_obs = samplingInversePinn.sample_observation_points(
    N_obs=N_OBS,
    alpha_true=alpha_true,
    T=T,
    device=device,
    t_min=T_OBS_MIN,
    t_max=T_OBS_MAX,
)

x_obs = x_obs.detach()
t_obs = t_obs.detach()
u_obs = u_obs.detach()

x_f, t_f = sampling.sample_interior_points(N_INTERIOR, T=T, device=device)
x_b, t_b = sampling.sample_boundary_points_u(N_BOUNDARY, T=T, device=device)
x_ic, t_ic = sampling.sample_initial_points_u(N_INITIAL, device=device)


# -----------------------------
# Optimizer
# -----------------------------

#optimizer = torch.optim.Adam(pinn.parameters(), lr=LR_NET)
network_params = [p for name, p in pinn.named_parameters() if name != "alpha"]

optimizer = torch.optim.Adam(
    [
        {"params": network_params, "lr": LR_NET},
        {"params": [pinn.alpha], "lr": LR_ALPHA},
    ]
)


# -----------------------------
# Loss history
# -----------------------------

loss_history = {
    "total": [],
    "pde": [],
    "bc": [],
    "ic": [],
    "obs": [],
    "alpha": [],
}


# -----------------------------
# Adam training
# -----------------------------

print("\nStarting Adam training...")

for epoch in tqdm(range(ADAM_EPOCHS)):
    pinn.train()

    loss_pde = lossesInversePinn.pde_loss(pinn, pinn.alpha, x_f, t_f)
    loss_bc = lossesInversePinn.boundary_loss(pinn, x_b, t_b)
    loss_ic = lossesInversePinn.initial_loss(pinn, x_ic, t_ic)
    loss_obs = lossesInversePinn.observation_loss(pinn, x_obs, t_obs, u_obs)

    loss = (
        LAMBDA_PDE * loss_pde
        + LAMBDA_BC * loss_bc
        + LAMBDA_IC * loss_ic
        + LAMBDA_OBS * loss_obs
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history["total"].append(loss.detach().cpu().item())
    loss_history["pde"].append(loss_pde.detach().cpu().item())
    loss_history["bc"].append(loss_bc.detach().cpu().item())
    loss_history["ic"].append(loss_ic.detach().cpu().item())
    loss_history["obs"].append(loss_obs.detach().cpu().item())
    loss_history["alpha"].append(pinn.alpha.detach().cpu().item())

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch:6d} | "
            f"Total: {loss.item():.4e} | "
            f"PDE: {loss_pde.item():.4e} | "
            f"BC: {loss_bc.item():.4e} | "
            f"IC: {loss_ic.item():.4e} | "
            f"OBS: {loss_obs.item():.4e} | "
            f"alpha: {pinn.alpha.item():.6f}"
        )


# -----------------------------
# L-BFGS refinement
# -----------------------------

if USE_LBFGS:
    print("\nStarting L-BFGS refinement...")
    print(f"Alpha before L-BFGS: {pinn.alpha.item():.6f}")

    pinn.train()

    lbfgs_optimizer = torch.optim.LBFGS(
        list(pinn.parameters()),
        lr=1.0,
        max_iter=LBFGS_MAX_ITER,
        max_eval=LBFGS_MAX_ITER * 2,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    lbfgs_counter = {"eval": 0}

    def closure():
        lbfgs_optimizer.zero_grad()

        loss_pde = lossesInversePinn.pde_loss(pinn, pinn.alpha, x_f, t_f)
        loss_bc = lossesInversePinn.boundary_loss(pinn, x_b, t_b)
        loss_ic = lossesInversePinn.initial_loss(pinn, x_ic, t_ic)
        loss_obs = lossesInversePinn.observation_loss(pinn, x_obs, t_obs, u_obs)

        loss = (
            LAMBDA_PDE * loss_pde
            + LAMBDA_BC * loss_bc
            + LAMBDA_IC * loss_ic
            + LAMBDA_OBS * loss_obs
        )

        loss.backward()

        lbfgs_counter["eval"] += 1

        loss_history["total"].append(loss.detach().cpu().item())
        loss_history["pde"].append(loss_pde.detach().cpu().item())
        loss_history["bc"].append(loss_bc.detach().cpu().item())
        loss_history["ic"].append(loss_ic.detach().cpu().item())
        loss_history["obs"].append(loss_obs.detach().cpu().item())
        loss_history["alpha"].append(pinn.alpha.detach().cpu().item())

        if lbfgs_counter["eval"] % 25 == 0:
            print(
                f"L-BFGS eval: {lbfgs_counter['eval']:5d} | "
                f"Total: {loss.item():.4e} | "
                f"PDE: {loss_pde.item():.4e} | "
                f"BC: {loss_bc.item():.4e} | "
                f"IC: {loss_ic.item():.4e} | "
                f"OBS: {loss_obs.item():.4e} | "
                f"alpha: {pinn.alpha.item():.6f}"
            )

        return loss

    lbfgs_optimizer.step(closure)

    print(f"Alpha after L-BFGS:  {pinn.alpha.item():.6f}")


# -----------------------------
# Evaluation and plotting
# -----------------------------

pinn.eval()
final_alpha = pinn.alpha.detach()

print("\nTraining finished")
print(f"True alpha:    {alpha_true:.6f}")
print(f"Learned alpha: {final_alpha.item():.6f}")

time_points = [0.0, 0.25, 0.5, 1.0]

plotInversePinn.plot_alpha_convergence(loss_history, alpha_true=alpha_true)
plot.plot_loss_history(loss_history, window=500)
plot.multi_t_plot(pinn, final_alpha, time_points)
plot.multi_t_plot_with_fd(pinn, final_alpha.item(), time_points)