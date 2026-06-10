import torch
from tqdm.auto import tqdm

import model
import losses
import sampling
import plot


# -----------------------------
# Configuration
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

alpha = 1.0

epochs = 50_00
learning_rate = 0.001

N_INTERIOR = 10_000
N_BOUNDARY = 1_000
N_INITIAL = 1_000

LAMBDA_PDE = 1.0
LAMBDA_BC = 20.0
LAMBDA_IC = 1.0


# -----------------------------
# Model and optimizer
# -----------------------------
pinn = model.forwardPinnModel().to(device)

optimizer = torch.optim.Adam(
    params=pinn.parameters(),
    lr=learning_rate
)


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
# Training loop
# -----------------------------
for epoch in tqdm(range(epochs)):
    pinn.train()

    # Sample training points
    x_f, t_f = sampling.sample_interior_points(N_INTERIOR, device=device)
    x_b, t_b = sampling.sample_boundary_points_u(N_BOUNDARY, device=device)
    x_ic, t_ic = sampling.sample_initial_points_u(N_INITIAL, device=device)

    # Compute individual losses
    loss_pde = losses.pde_loss(pinn, alpha, x_f, t_f)
    loss_bc = losses.data_loss(pinn, alpha, x_b, t_b)
    loss_ic = losses.data_loss(pinn, alpha, x_ic, t_ic)

    # Weighted total loss
    loss = (
        LAMBDA_PDE * loss_pde
        + LAMBDA_BC * loss_bc
        + LAMBDA_IC * loss_ic
    )

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store losses
    loss_history["total"].append(loss.item())
    loss_history["pde"].append(loss_pde.item())
    loss_history["bc"].append(loss_bc.item())
    loss_history["ic"].append(loss_ic.item())

    # Print progress
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch:5d} | "
            f"Total: {loss.item():.6e} | "
            f"PDE: {loss_pde.item():.6e} | "
            f"BC: {loss_bc.item():.6e} | "
            f"IC: {loss_ic.item():.6e}"
        )


# -----------------------------
# Evaluation and plotting
# -----------------------------
pinn.eval()

time_points = [0.0, 0.25, 0.5, 1.0]
plot.plot_loss_history(loss_history, window=500)
plot.multi_t_plot(pinn, alpha, time_points)
plot.multi_t_plot_with_fd(pinn, alpha, time_points)