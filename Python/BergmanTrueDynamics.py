from torch import nn
import torch


class BergmanTrueDynamics(nn.Module):
    def __init__(self, p1=0.028735, p2=0.028344, p3=5.035e-5, G_b=4.5, I_b=15.0, I_X=15.0):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.G_b = G_b
        self.I_b = I_b
        self.I_X = I_X 
        self.V1 = 12# % L
        self.n = 5/54#; % min

    def forward(self, t, state, D=0, U=3):
        G, X, I = state
        dGdt = -self.p1 * (G - self.G_b) - (X-self.I_X) * G + D
        dXdt = -self.p2 * (X-self.I_X) + self.p3 * (I - self.I_b)
        dIdt = U/self.V1 - self.n*I
        return torch.stack([dGdt, dXdt, dIdt])
    

def generate_data(model, time_span, G0, X0, I0):
    initial_state = torch.tensor([G0, X0, I0], dtype=torch.float32)
    time_points = torch.linspace(0, time_span, steps=300)
    with torch.no_grad():
        true_solution = odeint(model, initial_state, time_points, method='rk4',rtol=1e-10, atol=1e-9)
    return time_points, true_solution
    
def plot_data(time_points, true_solution):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axs[0].plot(time_points, true_solution[:, 0], label='Glucose (G)')
    axs[0].set_ylabel('Glucose (G)')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(time_points, true_solution[:, 1], label='Insulin (X)')
    axs[1].set_ylabel('Insulin (X)')
    axs[1].legend()
    axs[1].grid(True)
    
    axs[2].plot(time_points, true_solution[:, 2], label='Plasma Insulin (I)')
    axs[2].set_ylabel('Plasma Insulin (I)')
    axs[2].set_xlabel('Time')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    

    
from torchdiffeq import odeint
import numpy as np

model = BergmanTrueDynamics()
time_span = 100
G0 = 10.0
X0 = 15.0
I0 = 15.0
time_points, true_solution = generate_data(model, time_span, G0, X0, I0)
plot_data(time_points, true_solution)
    
    

    
    

    