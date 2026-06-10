import numpy as np


def solve_heat_equation_fd(
    alpha=1.0,
    T=1.0,
    nx=101,
    time_points=None,
    safety=0.4,
):

    if time_points is None:
        time_points = [T]

    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    if nx < 3:
        raise ValueError("nx must be at least 3.")

    if safety <= 0 or safety > 0.5:
        raise ValueError("safety must be in the interval (0, 0.5].")

    for t in time_points:
        if t < 0 or t > T:
            raise ValueError(f"time point {t} is outside [0, T].")

    # Spatial grid
    x = np.linspace(0.0, 1.0, nx)
    dx = x[1] - x[0]

    # Stable time step
    dt = safety * dx**2 / alpha
    nt = int(np.ceil(T / dt))

    # Adjust dt so that the final time is exactly T
    dt = T / nt

    # Stability number
    r = alpha * dt / dx**2

    if r > 0.5:
        raise ValueError(f"Unstable scheme: r = {r:.4f} > 0.5")

    # Initial condition
    u = np.sin(np.pi * x)

    # Boundary conditions
    u[0] = 0.0
    u[-1] = 0.0

    # Convert requested time points to nearest time-step indices
    target_steps = {
        t_target: int(round(t_target / dt))
        for t_target in time_points
    }

    snapshots = {}

    # Store initial state if requested
    for t_target, step in target_steps.items():
        if step == 0:
            snapshots[t_target] = u.copy()

    # Time stepping
    for step in range(1, nt + 1):
        u_new = u.copy()

        u_new[1:-1] = (
            u[1:-1]
            + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        )

        # Enforce boundary conditions
        u_new[0] = 0.0
        u_new[-1] = 0.0

        u = u_new

        for t_target, target_step in target_steps.items():
            if step == target_step:
                snapshots[t_target] = u.copy()

    return x, snapshots, dt, r