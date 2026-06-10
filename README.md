# Simple Physics-Informed Neural Network for the 1D Heat Equation in PyTorch

## Project Summary
(work in process, the inverse direction is not done yet. Also the finite difference method is not complete yet)
This project implements a simple Physics-Informed Neural Network (PINN) in PyTorch to approximate the solution of the one-dimensional heat equation.  
The project also includes a classical finite difference method as a numerical baseline.

The goal is to compare three different solution approaches:

$$
\text{Analytical solution}
\quad vs \quad
\text{PINN approximation}
\quad vs \quad
\text{Finite Difference approximation}
$$

The implementation is intended as a clean introductory project for understanding how neural networks can be trained with physical constraints and how their results compare to a traditional numerical method.

---

## Problem Definition

We consider the one-dimensional heat equation

$$
u_t = \alpha u_{xx}
$$

on the spatial domain

$$
x \in [0,1]
$$

and time interval

$$
t \in [0,T].
$$

Equivalently, the PDE residual is written as

$$
r(x,t) = u_t(x,t) - \alpha u_{xx}(x,t).
$$

The initial condition is

$$
u(x,0)=\sin(\pi x),
$$

and the boundary conditions are homogeneous Dirichlet conditions:

$$
u(0,t)=0,
\qquad
u(1,t)=0.
$$

For this setup, the analytical solution is

$$
u(x,t)=e^{-\alpha \pi^2 t}\sin(\pi x).
$$

This known solution is used to evaluate both the PINN and the finite difference method.

---

## Physics-Informed Neural Network Idea

A standard neural network is usually trained with labeled input-output pairs.

A Physics-Informed Neural Network also uses the governing differential equation as part of the training objective.

The neural network approximates the unknown solution

$$
u_\theta(x,t),
$$

where $\theta$ denotes the trainable weights and biases of the network.

The model receives two inputs:

$$
(x,t)
$$

and predicts one scalar output:

$$
u_\theta(x,t).
$$

The derivatives needed for the PDE residual are computed with PyTorch automatic differentiation.

---

## PINN Loss Function

The total PINN loss consists of three parts:

$$
L_{\text{total}}=
\lambda_{\text{PDE}} L_{\text{PDE}}
+
\lambda_{\text{BC}} L_{\text{BC}}
+
\lambda_{\text{IC}} L_{\text{IC}}.
$$

The loss terms enforce different parts of the PDE problem.


---

### PDE Residual Loss

The PDE residual is

$$
r_\theta(x,t)=
\frac{\partial u_\theta}{\partial t}(x,t)
-\alpha\frac{\partial^2 u_\theta}{\partial x^2}(x,t).
$$

The PDE loss is the mean squared residual over interior collocation points:

$$
L_{\text{PDE}}=
\frac{1}{N_f}
\sum_{i=1}^{N_f}
\left|
r_\theta(x_i,t_i)
\right|^2.
$$

This term enforces the heat equation inside the space-time domain.

---

### Initial Condition Loss

The initial condition is

$$
u(x,0)=\sin(\pi x).
$$

The initial condition loss is

$$
L_{\text{IC}}=
\frac{1}{N_{\text{IC}}}
\sum_{i=1}^{N_{\text{IC}}}
\left|
u_\theta(x_i,0)-\sin(\pi x_i)
\right|^2.
$$

This term enforces the correct starting state at \(t=0\).

---

### Boundary Condition Loss

The boundary conditions are

$$
u(0,t)=0,
\qquad
u(1,t)=0.
$$

The boundary condition loss is

$$
L_{\text{BC}}=
\frac{1}{N_{\text{BC}}}
\sum_{i=1}^{N_{\text{BC}}}
\left|
u_\theta(x_i,t_i)
\right|^2,
\qquad
x_i \in \{0,1\}.
$$

This term enforces the solution to stay close to zero at the spatial boundaries.

---

## Neural Network Architecture

The PINN uses a fully connected multilayer perceptron.

A typical architecture is:

```python
nn.Linear(2, hidden_dim)
nn.Tanh()
nn.Linear(hidden_dim, hidden_dim)
nn.Tanh()
nn.Linear(hidden_dim, 1)
```

The input dimension is 2 because the model receives both \(x\) and \(t\).  
The output dimension is 1 because the model predicts the scalar value \(u(x,t)\).

The `Tanh` activation is used because PINNs require smooth derivatives. This is important because the heat equation contains the second derivative \(u_{xx}\).

---

## Automatic Differentiation

The PDE residual requires derivatives of the neural network output with respect to the inputs.

For the heat equation, the required derivatives are:

$$
u_t
$$

and

$$
u_{xx}.
$$

Conceptually, the computation is:

```python
u = model(x, t)

u_t = derivative of u with respect to t
u_x = derivative of u with respect to x
u_xx = derivative of u_x with respect to x

residual = u_t - alpha * u_xx
loss_pde = mean(residual ** 2)
```

This allows the network to learn from the PDE itself, even without labeled solution values at interior points.

---

## Sampling Strategy

The project samples three different types of points.

### Interior Points

Interior collocation points are sampled from the space-time domain:

$$
x \in [0,1],
\qquad
t \in [0,T].
$$

These points are used for the PDE residual loss.

---

### Boundary Points

Boundary points are sampled at

$$
x=0
$$

and

$$
x=1
$$

with random time values \(t\).

These points are used for the boundary condition loss.

---

### Initial Points

Initial points are sampled at

$$
t=0
$$

with random spatial values \(x\).

These points are used for the initial condition loss.

---

## Finite Difference Method Baseline

To compare the PINN with a classical numerical method, this project also implements an explicit finite difference method for the one-dimensional heat equation.

The heat equation is

$$
u_t = \alpha u_{xx},
$$

on the spatial domain

$$
0 < x < 1
$$

with homogeneous Dirichlet boundary conditions

$$
u(0,t)=0,
\qquad
u(1,t)=0,
$$

and initial condition

$$
u(x,0)=\sin(\pi x).
$$

The finite difference method does not assume that the full solution \(u(x,t)\) is known.  
Instead, it starts from the known initial condition and then uses the discretized heat equation to propagate the solution forward in time.

We discretize the spatial and temporal domains by defining grid points

$$
x_i = i\Delta x,
\qquad
i=0,1,\dots,N,
$$

and

$$
t_n = n\Delta t,
\qquad
n=0,1,\dots,M.
$$

The numerical approximation is denoted by

$$
u_i^n \approx u(x_i,t_n).
$$

Here, $(u_i^n)$ represents the approximate solution value at spatial point $(x_i)$ and time $(t_n)$.

Using a forward difference in time gives

$$
u_t(x_i,t^n)
\approx
\frac{u_i^{n+1}-u_i^n}{\Delta t}.
$$

Using a centered difference in space gives

$$
u_{xx}(x_i,t^n)
\approx
\frac{
u_{i+1}^n - 2u_i^n + u_{i-1}^n
}{
\Delta x^2
}.
$$

Substituting these approximations into the heat equation gives the FTCS scheme:

$$
\frac{u_i^{n+1}-u_i^n}{\Delta t}=
\alpha
\frac{
u_{i+1}^n - 2u_i^n + u_{i-1}^n
}{
\Delta x^2
}.
$$

Solving for the next time step gives the explicit update formula

$$
u_i^{n+1}=
u_i^n
+
r
\left(
u_{i+1}^n - 2u_i^n + u_{i-1}^n
\right),
$$

where

$$
r = \alpha \frac{\Delta t}{\Delta x^2}.
$$

This formula is applied only to the interior grid points

$$
i=1,2,\dots,N-1.
$$

The boundary values are fixed for all time steps:

$$
u_0^n = 0,
\qquad
u_N^n = 0.
$$

The initial values are given by the initial condition:

$$
u_i^0 = \sin(\pi x_i).
$$

Therefore, the method first constructs the initial row of values at \(t=0\), then repeatedly computes the next time row from the previous one.

The explicit FTCS scheme is easy to implement, but it is only stable if the time step is sufficiently small. For the one-dimensional heat equation, the stability condition is

$$
r=
\alpha \frac{\Delta t}{\Delta x^2}
\leq
\frac{1}{2}.
$$

This means that when the spatial grid is refined, the time step usually has to be reduced as well.

The finite difference approximation becomes more accurate when both $(\Delta x)$ and $(\Delta t)$ are made smaller. However, this also increases the computational cost because more spatial grid points and more time steps are required.

In this project, the finite difference method is used as a classical numerical baseline. Its result is compared with both the analytical solution and the PINN approximation.
---



## Comparison Strategy

The project compares the three solution types at selected time points:

```python
time_points = [0.0, 0.25, 0.5, 1.0]
```

For each fixed time \(t\), the following curves can be plotted:

$$
u_{\text{exact}}(x,t),
$$

$$
u_{\text{PINN}}(x,t),
$$

and

$$
u_{\text{FD}}(x,t).
$$

This allows a visual comparison of:

- the analytical solution,
- the neural network approximation,
- the finite difference approximation.

The absolute errors can also be compared:

$$
|u_{\text{PINN}} - u_{\text{exact}}|
$$

and

$$
|u_{\text{FD}} - u_{\text{exact}}|.
$$

---

## Project Structure

```text
simple-pinn-heat-equation-pytorch/
│
├── training.py
├── model.py
├── losses.py
├── sampling.py
├── analyticSolutions.py
├── finite_difference.py
├── plot.py
├── requirements.txt
└── README.md
```

---

## Files

### `model.py`

Defines the neural network architecture used to approximate

$$
u_\theta(x,t).
$$

The model is a fully connected MLP. It receives `x` and `t`, concatenates them, and outputs the predicted solution value.

---

### `losses.py`

Contains the physics-informed loss functions:

- PDE residual loss,
- initial condition loss,
- boundary condition loss,
- total weighted loss construction.

The PDE loss uses PyTorch autograd to compute the derivatives required by the heat equation.

---

### `sampling.py`

Contains functions for sampling:

- interior collocation points,
- boundary points,
- initial condition points.

The training points are randomly resampled during training.

---

### `analyticSolutions.py`

Contains the analytical solution

$$
u(x,t)=e^{-\alpha \pi^2 t}\sin(\pi x).
$$

This is used for comparison with both the PINN and the finite difference method.

---

### `finiteDifference.py`
Implements the explicit finite difference method for the 1D heat equation.

The spatial and temporal grid points are defined as

$$
x_i = i\Delta x,
\qquad
t_n = n\Delta t.
$$

The finite difference method computes a grid-based approximation

$$
u_i^n \approx u(x_i,t_n),
$$

where (U_i^n) denotes the numerical approximation at spatial grid point (x_i) and time step (t_n).

The values are advanced in time using the FTCS update rule.

---

### `plot.py`

Contains plotting functions for:

- PINN vs analytical solution,
- PINN vs finite difference method,
- absolute error comparison,
- loss history visualization.

---

### `training.py`

Runs the PINN training loop.

The training loop repeatedly:

1. samples interior, boundary, and initial points,
2. computes the PDE, boundary, and initial losses,
3. combines them into a weighted total loss,
4. performs backpropagation,
5. updates the model parameters with Adam,
6. stores the loss history.

---

## Installation

Install the required packages with:

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

## How to Run

Train the PINN with:

```bash
python training.py
```

During training, the script prints the total loss and the individual loss components:

```text
Total loss
PDE loss
Boundary loss
Initial loss
```

After training, the project can plot:

- the training loss history,
- the exact solution compared with the PINN prediction,
- the exact solution compared with the finite difference solution,
- the absolute errors of both methods.

---

## GPU Support

The project can use a CUDA-capable GPU if available.

The device is selected with:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

The model and training tensors are moved to the same device.  
During plotting, GPU tensors are moved back to CPU before converting them to NumPy arrays.

---

## Results and Interpretation

The results are evaluated in three parts. First, the training behavior of the PINN is analyzed through the loss curves. Then, the learned PINN solution is compared with the analytical solution and the finite difference approximation. Finally, the absolute error is visualized to better understand where the PINN deviates from the exact solution.

### Training Behavior

The following plot shows the development of the different loss components during training.

![pinnloss](https://hackmd.io/_uploads/S1UJ208ZGg.png)

The total loss decreases strongly during the first part of training and then gradually approaches a plateau. This indicates that the PINN successfully learns the main constraints of the problem.

The initial condition loss decreases very quickly. This shows that the network learns the initial sine profile

$$
u(x,0)=\sin(\pi x)
$$

well. The boundary loss also becomes small, meaning that the network approximately satisfies the boundary conditions

$$
u(0,t)=0,
\qquad
u(1,t)=0.
$$

The PDE loss decreases more slowly and remains one of the larger components near the end of training. This is expected because satisfying the differential equation throughout the full interior domain is more difficult than matching the initial and boundary values at sampled points.

Overall, the loss curves show that the training process is stable. However, the plateau in the total loss also indicates that the model does not reach an exact solution. Small approximation errors remain, especially at later time values.

### Comparison with Analytical and Finite Difference Solutions

The following plots compare the analytical solution, the PINN prediction, and the finite difference approximation at different time points.

![comparison with finite difference](https://hackmd.io/_uploads/SJ8z2R8-zg.png)

At (t=0), the PINN prediction matches the initial condition very closely. This is consistent with the small initial condition loss observed during training.

At (t=0.25), the PINN still follows the analytical solution very well. The solution has already decayed significantly compared to the initial state, and the PINN captures this decay correctly.

At (t=0.5), small deviations become more visible, especially close to the boundary points. The PINN still captures the correct overall sine-like shape, but it does not match the analytical solution perfectly.

At (t=1.0), the analytical solution is already very close to zero because the heat equation causes the initial temperature profile to decay over time. In this regime, even small absolute errors become visually significant. The PINN slightly undershoots the true solution and produces small negative values. This does not mean that the model completely fails, but it shows that the relative error becomes more noticeable when the true solution itself is very small.

The finite difference approximation almost overlaps with the analytical solution at all shown time points. This is expected because the problem is one-dimensional, smooth, and defined on a simple domain. For this type of problem, classical finite difference methods are very accurate when the grid resolution is sufficiently fine and the stability condition is satisfied.

### Absolute Error of the PINN

The following plot shows the analytical solution, the PINN prediction, and the absolute error between them.

![function behaviour](https://hackmd.io/_uploads/HJpMnCLbzg.png)

The absolute error is small for most of the domain at early and intermediate time values. This confirms that the PINN learns the qualitative behavior of the heat equation well.

At (t=0.5), the largest errors appear near the boundary points. This suggests that the boundary conditions are approximately learned, but not enforced perfectly.

At (t=1.0), the absolute error becomes more important compared to the magnitude of the analytical solution. The true solution is almost zero, while the PINN still produces small negative values. Therefore, the absolute error is still small in numerical size, but it is large relative to the exact solution.

### Overall Interpretation

The PINN successfully learns the main behavior of the one-dimensional heat equation. It captures the initial sine-shaped temperature profile and its decay over time. The results show that the model is able to approximate the solution using the PDE residual, the initial condition, and the boundary conditions as training constraints.

However, the comparison also shows that the finite difference method is more accurate for this simple benchmark problem. This is not surprising. For low-dimensional PDEs on simple domains, classical numerical methods such as finite differences are highly efficient, stable, and accurate.

---

## Possible Improvements

Possible extensions include:

- using a deeper or wider MLP,
- testing different activation functions,
- applying adaptive loss weighting,
- using adaptive sampling strategies,
---


## Disclaimer

This project is a learning-focused baseline implementation.  
The goal is to understand the mechanics of PINNs and compare them with a simple classical numerical method.

---

## References

- M. Raissi, P. Perdikaris, and G. E. Karniadakis, *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*, Journal of Computational Physics, 2019.
- N. Thuerey et al., *Physics-based Deep Learning*, online book.
- PyTorch documentation on automatic differentiation.
