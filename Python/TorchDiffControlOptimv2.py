import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler  # Mixed-precision utilities
import time
from BergmanTrueDynamics import BergmanTrueDynamics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Precompute meal and insulin arrays
def precompute_meal_input(time_points):
    meal_array = torch.zeros_like(time_points)
    meal_array[time_points == 60] = 50.0
    meal_array[time_points == 180] = 50.0
    return meal_array

def precompute_insulin_input(time_points):
    insulin_array = torch.zeros_like(time_points)
    insulin_array[(time_points > 29) & (time_points < 31)] = 2.0
    insulin_array[(time_points > 119) & (time_points < 121)] = 2.0
    return insulin_array



class NeuralODEFunc(nn.Module):
    def __init__(self, state_dim, hidden_dim, ext_input_dim):
        super(NeuralODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + ext_input_dim, hidden_dim),
            nn.ReLU(),  # Faster than Tanh
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t, state, ext_input):
        # Ensure state has a batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, 3]

        # ext_input should be [1, 2]
        combined_input = torch.cat((state, ext_input), dim=-1)
        return self.net(combined_input)

@torch.no_grad()
def generate_data(true_model, time_points, G0, X0, I0, control_inputs, disturbance_inputs):
    initial_state = torch.tensor([G0, X0, I0], dtype=torch.float32, device=device)
    ext_inputs = torch.stack([control_inputs, disturbance_inputs], dim=-1)  # [T, 2]

    def true_ode(t, y):
        idx = t.long()  # since t matches time_points exactly
        D = ext_inputs[idx, 1]  # disturbance input
        U = ext_inputs[idx, 0]  # control input
        return true_model(t, y, D, U)

    true_solution = odeint(true_ode, initial_state, time_points, method='rk4')
    return time_points, true_solution

def loss_function(predicted, true):
    return ((predicted - true) ** 2).mean()

def train(ode_func, time_points, true_solution, control_inputs, disturbance_inputs, optimizer, epochs=1000):
    ode_func.to(device)
    true_solution = true_solution.to(device)
    initial_state = true_solution[0]

    # Precompute external inputs in one tensor for indexing
    ext_inputs = torch.stack([control_inputs, disturbance_inputs], dim=-1).to(device)

    # ODE closure without searchsorted
    def ode_closure(t, y):
        idx = t.long()
        # Ensure y has batch dimension
        if y.dim() == 1:
            y = y.unsqueeze(0)
        ext_input = ext_inputs[idx].unsqueeze(0)  # [1, 2]
        return ode_func(t, y, ext_input)

    scaler = GradScaler()
    losses = []
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            predicted_solution = odeint(ode_closure, initial_state, time_points, method="rk4")
            loss = loss_function(predicted_solution, true_solution)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        if epoch % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Time: {elapsed_time:.2f}s")
            start_time = time.time()

    return losses

def plot_results(time_points, true_solution, predicted_solution, control_inputs, disturbance_inputs):
    plt.figure(figsize=(12, 8))

    # Plotting Glucose levels
    plt.subplot(2, 1, 1)
    plt.plot(time_points.cpu().numpy(), true_solution[:, 0].cpu().numpy(), label="True Glucose")
    plt.plot(time_points.cpu().numpy(), predicted_solution[:, 0].cpu().numpy(), "r--", label="Predicted Glucose")
    plt.title("Glucose Dynamics Over Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Glucose Level (mg/dL)")
    plt.legend()

    # Plotting Insulin and Meal inputs
    plt.subplot(2, 1, 2)
    plt.step(time_points.cpu().numpy(), control_inputs.cpu().numpy(), label="Insulin Input", where="post")
    plt.step(time_points.cpu().numpy(), disturbance_inputs.cpu().numpy(), label="Meal Input", where="post", linestyle="--")
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

if __name__ == "__main__":


    # Initial conditions
    G0 = 5.0
    X0 = 15.0
    I0 = 15.0
    time_span = 360
    time_points = torch.linspace(0, time_span, time_span + 1, device=device)

    control_inputs = precompute_insulin_input(time_points).to(device)
    disturbance_inputs = precompute_meal_input(time_points).to(device)

    true_model = BergmanTrueDynamics().to(device)
    time_points, true_solution = generate_data(true_model, time_points, G0, X0, I0, control_inputs, disturbance_inputs)

    ode_func = NeuralODEFunc(state_dim=3, hidden_dim=50, ext_input_dim=2)
    optimizer = optim.Adam(ode_func.parameters(), lr=0.01)

    losses = train(ode_func, time_points, true_solution, control_inputs, disturbance_inputs, optimizer, epochs=100)

    with torch.no_grad():
        ext_inputs = torch.stack([control_inputs, disturbance_inputs], dim=-1)
        def ode_closure_eval(t, y):
            idx = t.long()
            if y.dim() == 1:
                y = y.unsqueeze(0)
            ext_input = ext_inputs[idx].unsqueeze(0)
            return ode_func(t, y, ext_input)

        predicted_solution = odeint(ode_closure_eval, true_solution[0], time_points, method="rk4")

    plot_results(time_points, true_solution, predicted_solution*0, control_inputs, disturbance_inputs)
    plot_losses(losses)
