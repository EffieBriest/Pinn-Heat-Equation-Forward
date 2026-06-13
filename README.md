# Physics-Informed Neural Networks for the 1D Heat Equation

This project implements and compares three approaches for the one-dimensional heat equation:

1. a **forward Physics-Informed Neural Network**,
2. an **inverse Physics-Informed Neural Network**, and
3. a classical **finite difference baseline**.

The forward PINN learns the solution when the heat coefficient $\alpha$ is known. The inverse PINN learns both the solution and the unknown coefficient $\alpha$ from observation data. The finite difference method serves as a classical numerical reference.

The project is designed as a clean learning-oriented implementation for understanding how PINNs work on both forward and inverse PDE problems.

---

## 1. Mathematical Problem

We consider the one-dimensional heat equation

$$
u_t = \alpha u_{xx}$$

on the domain

$$x \in [0,1], \qquad t \in [0,T].$$

The initial condition is

$$u(x,0)=\sin(\pi x),$$

and the boundary conditions are homogeneous Dirichlet conditions:

$$u(0,t)=0, \qquad u(1,t)=0.$$

For this setup, the analytical solution is known:

$$u(x,t)=e^{-\alpha \pi^2 t}\sin(\pi x).$$

This exact solution is used to evaluate the forward PINN, compare the finite difference method, and generate synthetic observation data for the inverse PINN.

---

## 2. Physics-Informed Neural Network Idea

A standard neural network is usually trained only on input-output data. A Physics-Informed Neural Network also includes the governing differential equation in the loss function.

The network approximation is denoted by

$$u_\theta(x,t),$$

where $\theta$ are the trainable neural network parameters. The PDE residual is

$$r_\theta(x,t)=\frac{\partial u_\theta}{\partial t}(x,t)-\alpha \frac{\partial^2 u_\theta}{\partial x^2}(x,t).$$

The PINN is trained to make this residual small at sampled interior collocation points.

---

## 3. Forward PINN

### Goal

In the forward problem, $\alpha$ is known. The model learns the solution

$$u_\theta(x,t) \approx u(x,t).$$

The network receives two inputs, \((x,t)\), and outputs one scalar value, \(u_\theta(x,t)\).

### Loss Function

The forward PINN uses three loss terms:

$$L_{\text{total}}=\lambda_{\text{PDE}}L_{\text{PDE}}+\lambda_{\text{BC}}L_{\text{BC}}+\lambda_{\text{IC}}L_{\text{IC}}.$$

The terms are:

$$L_{\text{PDE}}=\frac{1}{N_f}\sum_{i=1}^{N_f}\left|r_\theta(x_i,t_i)\right|^2,$$

$$L_{\text{IC}}=\frac{1}{N_{\text{IC}}}\sum_{i=1}^{N_{\text{IC}}}\left|u_\theta(x_i,0)-\sin(\pi x_i)\right|^2,$$

$$L_{\text{BC}}=\frac{1}{N_{\text{BC}}}\sum_{i=1}^{N_{\text{BC}}}\left|u_\theta(x_i,t_i)\right|^2, \qquad x_i\in\{0,1\}.$$

The PDE loss enforces the heat equation inside the domain. The initial condition loss enforces the correct starting profile. The boundary condition loss enforces zero temperature at both boundaries.

### Network Architecture

The PINN uses a fully connected multilayer perceptron:

```python
nn.Linear(2, hidden_dim)
nn.Tanh()
nn.Linear(hidden_dim, hidden_dim)
nn.Tanh()
nn.Linear(hidden_dim, 1)
```

`Tanh` is used because the PDE residual requires smooth derivatives, especially the second derivative $u_{xx}$.

### Automatic Differentiation

The PDE residual requires $u_t$ and $u_{xx}$. These are computed with PyTorch autograd:

```python
u = model(x, t)

u_t = derivative of u with respect to t
u_x = derivative of u with respect to x
u_xx = derivative of u_x with respect to x

residual = u_t - alpha * u_xx
loss_pde = mean(residual ** 2)
```

The collocation points must track gradients:

```python
x = x.clone().detach().requires_grad_(True)
t = t.clone().detach().requires_grad_(True)
```

This creates a fresh autograd graph for every PDE-loss evaluation and avoids reusing old computation graphs across epochs.

### Sampling Strategy

The forward PINN samples three types of training points:

| Point type | Domain | Purpose |
|---|---:|---|
| Interior points | $x\in[0,1],\ t\in[0,T]$ | PDE residual loss |
| Boundary points | $x=0$ or $x=1$, random $t$ | Boundary condition loss |
| Initial points | random $x$, $t=0$ | Initial condition loss |

---

## 4. Inverse PINN

### Goal

In the inverse problem, $\alpha$ is unknown. The model learns both

$$u_\theta(x,t)$$

and

$$\alpha.$$

The forward problem is

$$\alpha \text{ known} \quad \longrightarrow \quad u(x,t) \text{ learned},$$

while the inverse problem is

$$u_{\text{obs}}(x_i,t_i) \text{ partially known},\ \alpha \text{ unknown} \quad \longrightarrow \quad u_\theta(x,t) \text{ and } \alpha \text{ learned}.$$

### Observation Data

Synthetic observations are generated from the analytical solution:

$$u_{\text{obs}}=e^{-\alpha_{\text{true}}\pi^2 t_{\text{obs}}}\sin(\pi x_{\text{obs}}).$$

Optional Gaussian noise can be added:

$$u_{\text{obs,noisy}}=u_{\text{obs}}+\sigma_{\text{noise}}\varepsilon, \qquad \varepsilon\sim\mathcal{N}(0,1).$$

In code:

```python
noise_level = 0.05
u_obs_noisy = u_obs + noise_level * torch.std(u_obs) * torch.randn_like(u_obs)
```

The noise is scaled relative to the standard deviation of the clean observations.

### Observation Time Window

The inverse setup uses early-time observations:

```python
T_OBS_MIN = 0.02
T_OBS_MAX = 0.30
```

This is useful because the solution still has visible magnitude at early times. At late times, the heat equation solution becomes close to zero, making different values of \(\alpha\) harder to distinguish numerically.

### Loss Function

The inverse PINN uses four loss terms:

$$L_{\text{total}}=\lambda_{\text{PDE}}L_{\text{PDE}}+\lambda_{\text{BC}}L_{\text{BC}}+\lambda_{\text{IC}}L_{\text{IC}}+\lambda_{\text{OBS}}L_{\text{OBS}}.$$

The observation loss is

$$L_{\text{OBS}}=\frac{1}{N_{\text{OBS}}}\sum_{i=1}^{N_{\text{OBS}}}\left|u_\theta(x_i,t_i)-u_{\text{obs},i}\right|^2.$$

The observation loss makes the model fit the data. The PDE loss connects this data fit to the unknown physical parameter \(\alpha\). Together, they allow parameter recovery.

### Trainable Alpha

The heat coefficient is included as a trainable model parameter. The optimization problem is

$$\min_{\theta,\alpha} L_{\text{total}}(\theta,\alpha).$$

The implementation uses separate learning rates for the network and for \(\alpha\):

```python
network_params = [p for name, p in pinn.named_parameters() if name != "alpha"]

optimizer = torch.optim.Adam(
    [
        {"params": network_params, "lr": LR_NET},
        {"params": [pinn.alpha], "lr": LR_ALPHA},
    ]
)
```

This is useful because \(\alpha\) is a single scalar and may require a different learning rate than the neural network weights.

---

## 5. Optimization

The inverse PINN is trained in two stages:

1. **Adam phase**: robust first-order optimization to move the model into a good parameter region.
2. **L-BFGS refinement**: quasi-Newton refinement to reduce the remaining residuals more accurately.

This two-stage strategy is common for PINNs because the early training phase is often unstable, while the later phase benefits from more precise local optimization.

---

### Adam Phase

The first stage uses Adam. During this phase, the model updates both the neural network parameters $\theta$ and the unknown heat coefficient \(\alpha\).

The optimization problem is

$$
\min_{\theta,\alpha} L_{\text{total}}(\theta,\alpha),
$$

where

$$
L_{\text{total}}=
\lambda_{\text{PDE}}L_{\text{PDE}}
+
\lambda_{\text{BC}}L_{\text{BC}}
+
\lambda_{\text{IC}}L_{\text{IC}}
+
\lambda_{\text{OBS}}L_{\text{OBS}}.
$$

The PDE loss depends directly on \(\alpha\), because the heat equation residual is

$$
r(x,t)=u_t(x,t)-\alpha u_{xx}(x,t).
$$

Therefore, the optimizer does not only learn the shape of the solution $u_\theta(x,t)$, but also adjusts $\alpha$ so that the learned solution satisfies the heat equation.

During each Adam epoch, the loss terms are computed and combined:

```python
loss_pde = lossesInversePinn.pde_loss(pinn, pinn.alpha, x_f, t_f)
loss_bc = lossesInversePinn.boundary_loss(pinn, x_b, t_b)
loss_ic = lossesInversePinn.initial_loss(pinn, x_ic, t_ic)
loss_obs = lossesInversePinn.observation_loss(pinn, x_obs, t_obs, u_obs_noisy)

loss = (
    LAMBDA_PDE * loss_pde
    + LAMBDA_BC * loss_bc
    + LAMBDA_IC * loss_ic
    + LAMBDA_OBS * loss_obs
)
```

The standard PyTorch training step is then applied:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The backward pass computes gradients with respect to both trainable parts of the inverse PINN:

$$
\nabla_\theta L_{\text{total}}
$$

and

$$
\frac{\partial L_{\text{total}}}{\partial \alpha}.
$$

Adam is useful in the early phase because the different loss components can have very different magnitudes. The network may not yet satisfy the boundary condition, the initial condition, the observation data, or the PDE residual well. This makes the optimization problem badly scaled.

Adam handles this by adapting the effective step size for each parameter. For a parameter \(w\), it keeps moving averages of the gradient and the squared gradient:

$$
m_k = \beta_1 m_{k-1} + (1-\beta_1)g_k,
$$

$$
v_k = \beta_2 v_{k-1} + (1-\beta_2)g_k^2.
$$

The update has the form

$$
w_{k+1}=w_k-
\eta
\frac{\hat{m}_k}{\sqrt{\hat{v}_k}+\varepsilon}.
$$

This helps stabilize the early training phase. Parameters with large or unstable gradients receive smaller effective updates, while parameters with smaller gradients can still be updated meaningfully.

This is especially important for the inverse problem because $\theta$ and $\alpha$ behave very differently. The neural network parameters control the full function $u_\theta(x,t)$, while $\alpha$ is only a single scalar but has a direct effect on the PDE residual. For this reason, the implementation uses separate learning rates for the network and for $\alpha$.

---

### L-BFGS Refinement

After Adam, the inverse PINN can optionally be refined with L-BFGS:

```python
USE_LBFGS = True
LBFGS_MAX_ITER = 1000
```

Adam is good at reaching a reasonable parameter region, but it does not explicitly use curvature information. Once the model is already close to a good solution, L-BFGS can often reduce the remaining residuals more precisely.

L-BFGS is a quasi-Newton method. A full Newton method would use the Hessian matrix

$$
H=\nabla^2 L(\theta,\alpha)
$$

and compute an update direction of the form

$$
p_k=-H_k^{-1}\nabla L_k.
$$

For a neural network, the full Hessian is too large to compute and store. L-BFGS avoids this by building a limited-memory approximation of the inverse Hessian from recent parameter and gradient changes. In practice, this gives the optimizer some information about the local curvature of the loss landscape without explicitly forming the Hessian.

PyTorch's L-BFGS optimizer requires a closure because one optimizer step may evaluate the loss multiple times. This is needed, for example, when the optimizer performs an internal line search to choose a suitable step size.

```python
def closure():
    lbfgs_optimizer.zero_grad()

    loss_pde = lossesInversePinn.pde_loss(pinn, pinn.alpha, x_f, t_f)
    loss_bc = lossesInversePinn.boundary_loss(pinn, x_b, t_b)
    loss_ic = lossesInversePinn.initial_loss(pinn, x_ic, t_ic)
    loss_obs = lossesInversePinn.observation_loss(pinn, x_obs, t_obs, u_obs_noisy)

    loss = (
        LAMBDA_PDE * loss_pde
        + LAMBDA_BC * loss_bc
        + LAMBDA_IC * loss_ic
        + LAMBDA_OBS * loss_obs
    )

    loss.backward()
    return loss
```

The closure recomputes the full loss and its gradients whenever L-BFGS needs them. Therefore, the training points should remain fixed during the L-BFGS phase. If the collocation, boundary, initial, or observation points were resampled inside the closure, the optimizer would see a different objective function at each evaluation. This would make the curvature approximation unreliable.


## 6. Finite Difference Baseline

The finite difference method is used as a classical reference solution.

The spatial and temporal grids are

$$x_i=i\Delta x, \qquad t_n=n\Delta t,$$

and the numerical approximation is

$$u_i^n \approx u(x_i,t_n).$$

Using a forward difference in time and a centered difference in space gives the explicit FTCS scheme:

$$u_i^{n+1}=u_i^n+r\left(u_{i+1}^n-2u_i^n+u_{i-1}^n\right),$$

where

$$r=\alpha\frac{\Delta t}{\Delta x^2}.$$

The method is stable only if

$$r\leq \frac{1}{2}.$$

For this simple one-dimensional benchmark, the finite difference solution is expected to be very accurate when the grid is sufficiently fine and the stability condition is satisfied.

---

## 7. Project Structure

```text
pinn-heat-equation/
│
├── 01_1D-Heat-Equation-Forward/
│   ├── training.py
│   ├── model.py
│   ├── losses.py
│   ├── sampling.py
│   ├── analyticSolutions.py
│   ├── finiteDifference.py
│   ├── plot.py
│   └── README.md
│
├── 02_1D-Heat-Equation-Inverse/
│   ├── trainingInversePinn.py
│   ├── modelInversePinn.py
│   ├── lossesInversePinn.py
│   ├── samplingInversePinn.py
│   ├── plotInversePinn.py
│   └── README.md
│
├── 03_1D-Heat-Equation-Finite-Difference/
│   └── finiteDifference.py
│
├── requirements.txt
└── README.md
```

### Main Files

| File | Purpose |
|---|---|
| `model.py` | Defines the forward PINN architecture |
| `losses.py` | Defines PDE, initial, and boundary losses |
| `sampling.py` | Samples interior, boundary, and initial points |
| `analyticSolutions.py` | Provides the exact heat equation solution |
| `finiteDifference.py` | Implements the FTCS finite difference method |
| `plot.py` | Plots forward PINN results and comparisons |
| `modelInversePinn.py` | Defines the inverse PINN with trainable \(\alpha\) |
| `lossesInversePinn.py` | Adds observation loss for inverse training |
| `samplingInversePinn.py` | Generates synthetic observation data |
| `plotInversePinn.py` | Plots alpha convergence and inverse results |
| `training.py` | Runs forward PINN training |
| `trainingInversePinn.py` | Runs inverse PINN training and alpha recovery |

---

## 8. Installation

Install the dependencies with:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` is:

```text
torch
numpy
matplotlib
tqdm
```

---

## 9. How to Run

### Forward PINN

From the forward project folder:

```bash
python training.py
```

The script trains the forward PINN and can plot:

- training loss history,
- PINN prediction versus analytical solution,
- PINN prediction versus finite difference solution,
- absolute error comparisons.

### Inverse PINN

From the inverse project folder:

```bash
python trainingInversePinn.py
```

The script trains the inverse PINN, prints the learned value of \(\alpha\), and plots alpha convergence.

---

## 10. Current Inverse Configuration

```python
T = 2.0

ADAM_EPOCHS = 1000
USE_LBFGS = True
LBFGS_MAX_ITER = 1000

N_INTERIOR = 2000
N_BOUNDARY = 200
N_INITIAL = 200
N_OBS = 500

LAMBDA_PDE = 1.0
LAMBDA_BC = 1.0
LAMBDA_IC = 1.0
LAMBDA_OBS = 1.0

alpha_true = 1.0
noise_level = 0.0

LR_NET = 2e-4
LR_ALPHA = 5e-3

T_OBS_MIN = 0.02
T_OBS_MAX = 0.30
```

The observation data, collocation points, boundary points, and initial points are sampled once and then reused during training. This makes experiments easier to compare across different noise levels.

---

## 11. Results

Note: For the Forward PINN, we only used ADAM as optimizer.

### Forward PINN Training

![PINN loss](https://hackmd.io/_uploads/S1UJ208ZGg.png)

The total loss decreases strongly at the beginning and then approaches a plateau. The initial and boundary losses become small, showing that the network learns the initial sine profile and the homogeneous boundary conditions well.

The PDE loss decreases more slowly because satisfying the differential equation throughout the full interior domain is more difficult than matching sampled initial and boundary values.

### Forward PINN vs Analytical and Finite Difference Solutions

![Comparison with finite difference](https://hackmd.io/_uploads/SJ8z2R8-zg.png)

The PINN captures the overall heat-equation behavior: the initial sine profile decays over time. At early times, the match is very good. At later times, small deviations become more visible because the exact solution is already close to zero.

The finite difference approximation almost overlaps with the analytical solution at the shown time points. This is expected for a smooth one-dimensional problem on a simple domain.

### Forward PINN Absolute Error

![Function behavior](https://hackmd.io/_uploads/HJpMnCLbzg.png)

The absolute error is small for most of the domain. The largest visible errors occur near later times and close to the boundaries. This shows that the PINN learns the qualitative solution behavior well, but does not match the exact solution as accurately as the finite difference method.

### Inverse PINN Loss development during training

The figure shows the development of the total loss and the individual PINN loss components during training. A moving average with window size 500 is used to make the overall trend easier to see.

All loss components decrease steadily over the course of training. Since the y-axis is logarithmic, this corresponds to a reduction by several orders of magnitude. The initial condition loss is relatively large at the beginning, which is expected because the network has not yet learned the correct solution shape. As training progresses, the PDE loss, boundary loss, and initial condition loss are all reduced consistently.

Around epoch 1000, the loss decreases much more sharply. This marks the transition from the first optimization phase to the refinement phase. In this stage, the already pre-trained network is improved further, leading to significantly smaller residuals.

The final losses are very small, with the total loss reaching approximately \(10^{-5}\). This indicates that the trained PINN satisfies the heat equation, the boundary condition, and the initial condition accurately in the noise-free case.
![loss over training inverse pinn](https://hackmd.io/_uploads/rk_nQjYWfg.png)

### PINN prediction compared to the analytical solution

The figure compares the trained PINN prediction with the analytical solution at several fixed time points.

At early times, especially at $t=0$ and $t=0.25$, the PINN matches the real solution very well. The predicted curve almost overlaps with the analytical solution, and the absolute error is close to zero. This shows that the network learned the initial condition and the early-time behaviour of the heat equation accurately.

At later times, the solution of the heat equation becomes very small because the initial sine profile decays exponentially. Around $t=0.5$, the PINN still captures the correct overall shape, but a small amplitude error becomes visible. The prediction is slightly below the real solution in the interior and shows small deviations near the boundary.

At $t=1.0$, the real solution is already close to zero. Although the absolute error is still small in numerical terms, it becomes large compared to the magnitude of the true solution. Therefore, the relative error is much more visible at late times. The PINN still produces values of order $10^{-4}$, while the true solution is almost fully decayed.

This shows an important limitation of the training setup: minimizing an absolute mean squared error can lead to very small total losses while still producing noticeable relative errors when the true solution itself is close to zero.
![comparison pinn and eal solution](https://hackmd.io/_uploads/SySrVotWfx.png)

### Inverse PINN Learning of $\alpha$

Since $\alpha$ is only a single scalar but directly affects the PDE residual, its learning rate is especially sensitive.

The plot shows that $\alpha$ does not converge smoothly from the beginning. During the early training phase, the network approximation is still inaccurate, so the gradients with respect to $\alpha$ are not yet very reliable. This causes an initial overshoot and correction. After the PINN has learned a better approximation of the solution, the updates of $\alpha$ become more stable and the parameter gradually moves toward the true value.

In the noise-free case, the chosen learning rate is stable enough to recover the correct value:

$$
\alpha_{\text{true}} = 1.0
$$


After the initial transient phase, the learned parameter converges very close to the true diffusion coefficient. This confirms that, without observation noise, the inverse PINN can correctly identify the physical parameter when the learning rate is chosen carefully.
![alpha learning](https://hackmd.io/_uploads/B1zbmjF-zg.png)
### Inverse PINN Alpha Recovery

The inverse PINN is evaluated by comparing the learned coefficient with the true value:

$$\alpha_{\text{learned}} \approx \alpha_{\text{true}}.$$

The relative error is computed as

$$\text{relative error}=\frac{|\alpha_{\text{learned}}-\alpha_{\text{true}}|}{|\alpha_{\text{true}}|}.$$

| Data noise | Sensor count | $\alpha_{\text{true}}$ | $\alpha_{\text{learned}}$ | Relative error |
|---:|---:|---:|---:|---:|
| 0% | 500 | 1.0 | 0.999920 | 0.000080 |
| 1% | 500 | 1.0 | 0.999637 | 0.000363 |
| 5% | 500 | 1.0 | 0.998339 | 0.001661 |
| 10% | 500 | 1.0 | 0.996637 | 0.003363 |

The results show that $\alpha$ is recovered accurately for clean data and remains stable under moderate noise.

### Influence of the Number of Observation Points

In the noise-free case, changing the number of observation points had only a small effect on the recovered value of $\alpha$. Once the observations cover the relevant time interval well enough, even a small number of clean data points can already identify the decay behaviour of the solution.

This is why the results for 10, 100, or even 10000 observation points can be very similar. Adding more points does not automatically make the observation loss more important, because the loss is averaged over all observation points. More points mainly improve how well the observation region is sampled.

The situation changes when noise is added. With noisy observations, a small number of points can give a distorted impression of the true solution behaviour. In that case, increasing the number of observation points becomes much more relevant, because the random noise is averaged out over more samples.
| Data noise | Sensor count | alpha learned | Relative error |
|---:|---:|---:|---:|
| 10% | 10 | 1.465745 | 0.465745 |
| 10% | 100 | 3.729761 | 2.729761 |
| 10% | 10000 | 0.863107 | 0.136893 |
| 10% | 1000000 | 0.867541 | 0.132459|


## 12. Possible Improvements

Possible extensions include:

- using a deeper or wider MLP,
- applying adaptive loss weighting,
- using adaptive collocation sampling,
- increasing the number of collocation points,
- enforcing positivity of $\alpha$ with a parametrization such as `softplus`,


---

## 13. Disclaimer

This project is a learning-focused baseline implementation. It is not intended to prove that PINNs outperform classical numerical solvers on this simple one-dimensional problem.

---

## 14. References

- M. Raissi, P. Perdikaris, and G. E. Karniadakis, *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*, Journal of Computational Physics, 2019.
- N. Thuerey et al., *Physics-based Deep Learning*, online book.
- PyTorch documentation on automatic differentiation.
