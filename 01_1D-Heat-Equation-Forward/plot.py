import matplotlib.pyplot as plt
import torch
import numpy as np
import analyticSolutions
import finiteDifference

def multi_t_plot(model1, alpha, time_points):
    model1.eval()

    device = next(model1.parameters()).device

    x_plot = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, t_fixed in zip(axes, time_points):
        t_plot = torch.full_like(x_plot, t_fixed)

        with torch.no_grad():
            u_real = analyticSolutions.u(alpha, x_plot, t_plot)
            u_pred = model1(x_plot, t_plot)
            abs_error = torch.abs(u_real - u_pred)

        x_np = x_plot.detach().cpu().numpy()
        u_real_np = u_real.detach().cpu().numpy()
        u_pred_np = u_pred.detach().cpu().numpy()
        abs_error_np = abs_error.detach().cpu().numpy()

        ax.plot(x_np, u_real_np, label="Real solution")
        ax.plot(x_np, u_pred_np, label="PINN prediction")
        ax.plot(x_np, abs_error_np, label="Absolute error")

        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"t = {t_fixed}")
        ax.legend()

    plt.tight_layout()
    plt.show()

def multi_t_plot_with_fd(model1, alpha, time_points):
    model1.eval()

    device = next(model1.parameters()).device

    alpha_value = float(alpha.detach().cpu()) if torch.is_tensor(alpha) else float(alpha)

    x_fd, fd_solutions, dt, r = finiteDifference.solve_heat_equation_fd(
        alpha=alpha_value,
        T=max(time_points),
        nx=201,
        time_points=time_points
    )

    x_plot = torch.tensor(x_fd, dtype=torch.float32, device=device).reshape(-1, 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, t_fixed in zip(axes, time_points):
        t_plot = torch.full_like(x_plot, t_fixed)

        with torch.no_grad():
            u_real = analyticSolutions.u(alpha, x_plot, t_plot)
            u_pred = model1(x_plot, t_plot)

        u_real_np = u_real.detach().cpu().numpy()
        u_pred_np = u_pred.detach().cpu().numpy()
        u_fd_np = fd_solutions[t_fixed]

        ax.plot(x_fd, u_real_np, label="Real solution")
        ax.plot(x_fd, u_pred_np, label="PINN prediction")
        ax.plot(x_fd, u_fd_np, label="Finite difference")

        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"t = {t_fixed}")
        ax.legend()

    plt.tight_layout()
    plt.show()

def moving_average(values, window=200):
    values = np.array(values)

    if len(values) < window:
        return values

    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_loss_history(loss_history, window=200):
    plt.figure(figsize=(10, 6))

    total = moving_average(loss_history["total"], window)
    pde = moving_average(loss_history["pde"], window)
    bc = moving_average(loss_history["bc"], window)
    ic = moving_average(loss_history["ic"], window)

    epochs = np.arange(len(total))

    plt.plot(epochs, total, label="Total loss")
    plt.plot(epochs, pde, label="Interior / PDE loss")
    plt.plot(epochs, bc, label="Boundary loss")
    plt.plot(epochs, ic, label="Initial loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(f"PINN Loss Development During Training, moving average window = {window}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()