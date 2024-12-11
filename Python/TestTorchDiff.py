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
    if t == 30 or t == 120:  # Insulin injections at minute 30 and 120
        return 2.0  # Increase in insulin level
    return 0.0
# # %%
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

#     def forward(self, t, state, D=0, U=0):
#         G, X, I = state
#         dGdt = -self.p1 * (G - self.G_b) - (X-self.I_X) * G + D
#         dXdt = -self.p2 * (X-self.I_X) + self.p3 * (I - self.I_b)
#         dIdt = U/self.V1 - self.n*I
#         return torch.stack([dGdt, dXdt, dIdt])



# %%
# Neural ODE function to fit the dynamics
class NeuralODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, t, y):
        return self.net(y)

# %%
def generate_data(model, time_span, G0, X0, I0):
    initial_state = torch.tensor([G0, X0, I0], dtype=torch.float32)
    time_points = torch.linspace(0, time_span, steps=300)
    with torch.no_grad():
        true_solution = odeint(model, initial_state, time_points,method='rk4')
    return time_points, true_solution

# %%
def loss_function(predicted, true):
    return ((predicted - true)**2).mean()

# %%
def train(ode_func, time_points, true_solution, optimizer, epochs=1000):
    losses = []
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    ode_func.to(device)
    time_points = time_points.to(device)
    true_solution = true_solution.to(device)

    scaler = GradScaler()
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        with autocast(device_name):
            predicted_solution = odeint(ode_func, true_solution[0], time_points, method='rk4')
            loss = loss_function(predicted_solution, true_solution)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        if epoch % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch}, Loss: {loss.item()}, Time: {elapsed_time:.2f} seconds")
            start_time = time.time() 

    return losses


def plot_results(time_points, true_solution, predicted_solution, control_inputs=0, disturbance_inputs=0):
    plt.figure(figsize=(12, 8))
    if control_inputs == 0:
        control_inputs = torch.zeros_like(time_points)
    if disturbance_inputs == 0:
        disturbance_inputs = torch.zeros_like(time_points)

    # Plotting Glucose levels
    plt.subplot(2, 1, 1)  # Two rows, one column, first plot
    plt.plot(time_points.numpy(), true_solution[:, 0].numpy(), label="True Glucose")
    plt.plot(time_points.numpy(), predicted_solution[:, 0].numpy(), 'r--', label="Predicted Glucose")
    plt.title("Glucose Dynamics Over Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Glucose Level (mg/dL)")
    plt.legend()

    # Plotting Insulin and Meal inputs
    plt.subplot(2, 1, 2)  # Two rows, one column, second plot
    plt.step(time_points.numpy(), control_inputs, label="Insulin Input", where="post")
    plt.step(time_points.numpy(), disturbance_inputs, label="Meal Input", where="post", linestyle='--')
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
    G0 = 10.0   # Initial glucose level (mg/dL)
    X0 = 15.0     # Initial insulin action
    I0 = 15.0    # Initial insulin level (uU/mL)
    time_span = 360  # Simulate for 180 minutes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    true_model = BergmanTrueDynamics()
    time_points, true_solution = generate_data(true_model, time_span, G0, X0, I0)

    ode_func = NeuralODEFunc(input_dim=3, hidden_dim=50)
    optimizer = optim.Adam(ode_func.parameters(), lr=0.01)
    losses = train(ode_func, time_points, true_solution, optimizer, epochs=200)

    with torch.no_grad():
        predicted_solution = odeint(ode_func, true_solution[0].to(device), time_points, method='rk4').cpu()
        

    plot_results(time_points, true_solution, predicted_solution)
    
    plot_losses(losses)
