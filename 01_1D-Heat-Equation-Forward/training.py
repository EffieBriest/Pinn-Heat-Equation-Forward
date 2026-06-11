import torch
from tqdm.auto import tqdm

import model
import losses
import sampling
import plot


# -----------------------------
# Configuration
# -----------------------------

FORCE_CPU = False
device = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)

alpha = 1.0

ADAM_EPOCHS = 10000
USE_LBFGS = True
LBFGS_MAX_ITER = 1000

N_INTERIOR = 10_000
N_BOUNDARY = 1_000
N_INITIAL = 1_000

LAMBDA_PDE = 1.0
LAMBDA_BC = 20.0
LAMBDA_IC = 1.0

LR_ADAM = 0.001


# -----------------------------
# Model
# -----------------------------

pinn = model.forwardPinnModel().to(device)


# -----------------------------
# Fixed training data
# -----------------------------

x_f, t_f = sampling.sample_interior_points(N_INTERIOR, device=device)
x_b, t_b = sampling.sample_boundary_points_u(N_BOUNDARY, device=device)
x_ic, t_ic = sampling.sample_initial_points_u(N_INITIAL, device=device)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.Adam(pinn.parameters(), lr=LR_ADAM)


# -----------------------------
# Loss history
# -----------------------------

loss_history = {
    "total": [],
    "pde": [],
    "bc": [],
    "ic": []
}


# -----------------------------
# Adam training
# -----------------------------

print("\nStarting Adam training...")

for epoch in tqdm(range(ADAM_EPOCHS)):
    pinn.train()

    loss_pde = losses.pde_loss(pinn, alpha, x_f, t_f)
    loss_bc = losses.data_loss(pinn, alpha, x_b, t_b)
    loss_ic = losses.data_loss(pinn, alpha, x_ic, t_ic)

    loss = (
        LAMBDA_PDE * loss_pde
        + LAMBDA_BC * loss_bc
        + LAMBDA_IC * loss_ic
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history["total"].append(loss.detach().cpu().item())
    loss_history["pde"].append(loss_pde.detach().cpu().item())
    loss_history["bc"].append(loss_bc.detach().cpu().item())
    loss_history["ic"].append(loss_ic.detach().cpu().item())

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch:5d} | "
            f"Total: {loss.item():.6e} | "
            f"PDE: {loss_pde.item():.6e} | "
            f"BC: {loss_bc.item():.6e} | "
            f"IC: {loss_ic.item():.6e}"
        )


# -----------------------------
# L-BFGS refinement
# -----------------------------

if USE_LBFGS:
    print("\nStarting L-BFGS refinement...")

    pinn.train()

    lbfgs_optimizer = torch.optim.LBFGS(
        pinn.parameters(),
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

        loss_pde = losses.pde_loss(pinn, alpha, x_f, t_f)
        loss_bc = losses.data_loss(pinn, alpha, x_b, t_b)
        loss_ic = losses.data_loss(pinn, alpha, x_ic, t_ic)

        loss = (
            LAMBDA_PDE * loss_pde
            + LAMBDA_BC * loss_bc
            + LAMBDA_IC * loss_ic
        )

        loss.backward()

        lbfgs_counter["eval"] += 1

        if lbfgs_counter["eval"] % 25 == 0:
            print(
                f"L-BFGS eval: {lbfgs_counter['eval']:5d} | "
                f"Total: {loss.item():.6e} | "
                f"PDE: {loss_pde.item():.6e} | "
                f"BC: {loss_bc.item():.6e} | "
                f"IC: {loss_ic.item():.6e}"
            )

        return loss

    lbfgs_optimizer.step(closure)


# -----------------------------
# Evaluation and plotting
# -----------------------------

pinn.eval()

time_points = [0.0, 0.25, 0.5, 1.0]

plot.plot_loss_history(loss_history, window=500)
plot.multi_t_plot(pinn, alpha, time_points)
plot.multi_t_plot_with_fd(pinn, alpha, time_points)