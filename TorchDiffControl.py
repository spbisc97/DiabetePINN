# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler  # Mixed-precision utilities
import time

from BergmanTrueDynamics import BergmanTrueDynamics

# %%
# Disturbance (Meal) and Control (Insulin) functions
def meal_input(t):
    if t == 60 or t == 180:  # Meals at minute 60 and 180
        return 4.0  # Increase in glucose level
    return 0.0


def insulin_input(t):
    if 29 <t< 31 or 119 <t <121:  # Insulin injections at minute 30 and 120
        return 5.0  # Increase in insulin level
    return 0.0


# # %% imported from BergmanTrueDynamics.py
# class BergmanTrueDynamics(nn.Module):
#     def __init__(self, p1=0.028735, p2=0.028344, p3=5.035e-5, G_b=100.0, I_b=15.0, I_X=15.0):
#         super().__init__()
#         self.p1 = p1
#         self.p2 = p2
#         self.p3 = p3
#         self.G_b = G_b
#         self.I_b = I_b
#         self.I_X = I_X 
#         self.V1 = 12# % L
#         self.n = 5/54#; % min

#     def forward(self, t, state, D, U):
#         G, X, I = state
#         dGdt = -self.p1 * (G - self.G_b) - (X-self.I_X) * G + D
#         dXdt = -self.p2 * (X-self.I_X) + self.p3 * (I - self.I_b)
#         dIdt = U/self.V1 - self.n*I
#         return torch.stack([dGdt, dXdt, dIdt])


# %%
# Neural ODE function to fit the dynamics
class NeuralODEFunc(nn.Module):
    def __init__(self, state_dim, hidden_dim, ext_input_dim):
        super(NeuralODEFunc, self).__init__()
        # Adjust for external inputs: control + disturbance
        self.net = nn.Sequential(
            nn.Linear(state_dim + ext_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t, state, ext_input):
        # Concatenate state and external inputs (control + disturbance)
        combined_input = torch.cat((state, ext_input.to(state.device)), dim=-1)
        return self.net(combined_input)


# %%
def dynamic_system(t, state, model, control_input, disturbance_input):
    D = disturbance_input
    U = control_input
    return model(t, state, D, U)


# %%
def generate_data(model, time_points, G0, X0, I0, control_inputs, disturbance_inputs):
    initial_state = torch.tensor([G0, X0, I0], dtype=torch.float32)
    true_solution = odeint(
        lambda t, y: dynamic_system(
            t, y, model, control_inputs(t), disturbance_inputs(t)
        ),
        initial_state,
        time_points,
        method="rk4",
    )
    return time_points, true_solution


# %%
def loss_function(predicted, true):
    return ((predicted - true) ** 2).mean()


# %%
def train(
    ode_func,
    time_points,
    true_solution,
    control_inputs,
    disturbance_inputs,
    optimizer,
    epochs=10,
):
    losses = []

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    ode_func.to(device)
    time_points = time_points.to(device)
    true_solution = true_solution.to(device)
    initial_state = true_solution[0].to(device)

    scaler = GradScaler()
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        with autocast(device_name):
            predicted_solution = odeint(
                lambda t, y: ode_func(
                    t,
                    y,
                    torch.tensor(
                        [control_inputs(t), disturbance_inputs(t)],
                        dtype=torch.float32,
                    ),
                ),
                initial_state,
                time_points,
                method="rk4",
            )

            loss = loss_function(predicted_solution, true_solution)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        if epoch % 100 == 0:
            elapsed_time = time.time() - start_time
            print(
                f"Epoch {epoch}, Loss: {loss.item()}, Time: {elapsed_time:.2f} seconds"
            )
            start_time = time.time()

    return losses


def plot_results(
    time_points,
    true_solution,
    predicted_solution,
    control_inputs=0,
    disturbance_inputs=0,
):
    plt.figure(figsize=(12, 8))
    if control_inputs is 0:
        control_inputs = torch.zeros_like(time_points)
    if disturbance_inputs is 0:
        disturbance_inputs = torch.zeros_like(time_points)

    # Plotting Glucose levels
    plt.subplot(2, 1, 1)  # Two rows, one column, first plot
    plt.plot(time_points.numpy(), true_solution[:, 0].numpy(), label="True Glucose")
    plt.plot(
        time_points.numpy(),
        predicted_solution[:, 0].numpy(),
        "r--",
        label="Predicted Glucose",
    )
    plt.title("Glucose Dynamics Over Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Glucose Level (mg/dL)")
    plt.legend()

    # Plotting Insulin and Meal inputs
    plt.subplot(2, 1, 2)  # Two rows, one column, second plot
    plt.step(time_points.numpy(), control_inputs, label="Insulin Input", where="post")
    plt.step(
        time_points.numpy(),
        disturbance_inputs,
        label="Meal Input",
        where="post",
        linestyle="--",
    )
    plt.title("Control (Insulin) and Disturbance (Meal) Inputs Over Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Input Levels")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_losses(losses):
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label="Loss")
    plt.yscale("log")
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
# Main script
if __name__ == "__main__":


    # Initial conditions (initial glucose, insulin action, insulin)
    G0 = 5.0  # Initial glucose level (mg/dL)
    X0 = 15.0  # Initial insulin action
    I0 = 15.0  # Initial insulin level (uU/mL)
    time_span = 600  # Simulate for 180 minutes
    time_points = torch.linspace(0, time_span, time_span+1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    control_inputs = torch.tensor([insulin_input(t.item()) for t in time_points])
    disturbance_inputs = torch.tensor([meal_input(t.item()) for t in time_points])

    true_model = BergmanTrueDynamics()
    time_points, true_solution = generate_data(
        true_model, time_points, G0, X0, I0, insulin_input, meal_input
    )

    ode_func = NeuralODEFunc(state_dim=3, hidden_dim=50, ext_input_dim=2)
    optimizer = optim.Adam(ode_func.parameters(), lr=0.01)
    losses = train(
        ode_func,
        time_points,
        true_solution,
        insulin_input,
        meal_input,
        optimizer,
        epochs=100,
    )

    with torch.no_grad():
        predicted_solution = odeint(
            lambda t, y: ode_func(
                t,
                y,
                torch.tensor(
                    [insulin_input(t), meal_input(t)],
                    dtype=torch.float32,
                ),
            ),
            true_solution[0].to(device),
            time_points,
            method="rk4",
        ).cpu()

    plot_results(time_points, true_solution, predicted_solution, control_inputs, disturbance_inputs)

    plot_losses(losses)

# %%
