# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler  # Mixed-precision utilities
import time
# %%
class BergmanTrueDynamics(nn.Module):
    def __init__(self, p1, p2, p3, G_b, I_b):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.G_b = G_b
        self.I_b = I_b

    def forward(self, t, state):
        G, X, I = state
        dGdt = -self.p1 * (G - self.G_b) - X * G
        dXdt = -self.p2 * X + self.p3 * (I - self.I_b) 
        dIdt = torch.tensor(0.0, device=state.device)  # Keeping the device consistent
        return torch.stack([dGdt, dXdt, dIdt])

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

def generate_data(model, time_span, G0, X0, I0):
    initial_state = torch.tensor([G0, X0, I0], dtype=torch.float32)
    time_points = torch.linspace(0, time_span, steps=300)
    with torch.no_grad():
        true_solution = odeint(model, initial_state, time_points)
    return time_points, true_solution

def loss_function(predicted, true):
    return ((predicted - true)**2).mean()

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
            print(f"Epoch {epoch}, Loss: {loss.item()}, Elapsed Time: {elapsed_time:.2f} seconds")
            start_time = time.time()

    return losses

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_model = BergmanTrueDynamics(p1=0.02, p2=0.025, p3=0.0001, G_b=100.0, I_b=15.0)
    time_points, true_solution = generate_data(true_model, 360, 300.0, 0.0, 15.0)
    ode_func = NeuralODEFunc(input_dim=3, hidden_dim=50)
    optimizer = optim.Adam(ode_func.parameters(), lr=0.01)
    losses = train(ode_func, time_points, true_solution, optimizer, epochs=100)

    with torch.no_grad():
        predicted_solution = odeint(ode_func, true_solution[0].to(device), time_points, method='rk4').cpu()
    plt.plot(time_points.numpy(), true_solution[:, 0].numpy(), label="True Glucose")
    plt.plot(time_points.numpy(), predicted_solution[:, 0].numpy(), '--', label="Predicted Glucose")
    plt.xlabel('Time (minutes)')
    plt.ylabel('Glucose Level (mg/dL)')
    plt.legend()
    plt.show()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# %%
