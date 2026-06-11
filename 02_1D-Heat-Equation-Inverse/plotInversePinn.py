import matplotlib.pyplot as plt


def plot_alpha_convergence(loss_history, alpha_true=None):
    alpha_history = loss_history["alpha"]

    plt.figure(figsize=(8, 5))
    plt.plot(alpha_history, label="learned alpha")

    if alpha_true is not None:
        plt.axhline(
            y=float(alpha_true),
            linestyle="--",
            label="true alpha"
        )

    plt.xlabel("Epoch")
    plt.ylabel("alpha")
    plt.title("Convergence of learned alpha")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()