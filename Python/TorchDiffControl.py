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
    # Simulate meal intake with a more realistic profile
    if 60 <= t < 90 or 180 <= t < 210:  # Meals at minute 60-90 and 180-210
        return 4.0  # Increase in glucose level
    if 150 <= t < 180:
        return 3.0
    if 240 <= t < 270:
        return 2.0
    if 330 <= t < 360:
        return 1.0
    if 420 <= t < 450:
        return 2.0
    if 510 <= t < 540:
        return 3.0
    if 600 <= t < 630:
        return 4.0
    if 690 <= t < 720:
        return 3.0
    if 780 <= t < 810:
        return 2.0
    return 0.0


def insulin_input(t):
    # Simulate insulin injections with a more realistic profile
    if 30 <= t < 35 or 120 <= t < 125:  # Insulin injections at minute 30-35 and 120-125
        return 5.0  # Increase in insulin level
    if 210 <= t < 215:
        return 1.0
    if 270 <= t < 275:
        return 2.0
    if 330 <= t < 335:
        return 3.0
    if 390 <= t < 395:
        return 4.0
    if 450 <= t < 455:
        return 5.0
    if 510 <= t < 515:
        return 6.0
    if 570 <= t < 575:
        return 2.0
    if 630 <= t < 635:
        return 3.0
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
    plt.subplot(4, 1, 1)  # Two rows, one column, first plot
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
    
    # Plotting Insulin Action levels
    plt.subplot(4, 1, 2)  # Four rows, one column, second plot
    plt.plot(time_points.numpy(), true_solution[:, 1].numpy(), label="True Insulin Action")
    plt.plot(
        time_points.numpy(),
        predicted_solution[:, 1].numpy(),
        "r--",
        label="Predicted Insulin Action",
    )
    plt.title("Insulin Action Dynamics Over Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Insulin Action Level")
    plt.legend()

    # Plotting Insulin levels
    plt.subplot(4, 1, 3)  # Four rows, one column, third plot
    plt.plot(time_points.numpy(), true_solution[:, 2].numpy(), label="True Insulin")
    plt.plot(
        time_points.numpy(),
        predicted_solution[:, 2].numpy(),
        "r--",
        label="Predicted Insulin",
    )
    plt.title("Insulin Dynamics Over Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Insulin Level (uU/mL)")
    plt.legend()
    
        # Plotting Insulin and Meal inputs
    plt.subplot(4, 1, 4)  # Two rows, one column, second plot
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

def status_plot(ode_func, true_solution, time_points, insulin_input, meal_input, device):
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

    plot_results(time_points.cpu(), true_solution.cpu(), predicted_solution, torch.tensor([insulin_input(t.item()) for t in time_points], dtype=torch.float32), torch.tensor([meal_input(t.item()) for t in time_points], dtype=torch.float32))

# %%
def train(
    ode_func,
    time_points,
    true_solution,
    control_inputs,
    disturbance_inputs,
    optimizer,
    epochs=10,
    plot_freq=10,
    info_freq=10,
):
    losses = []

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    ode_func.to(device)
    
    time_points = time_points.to(device)
    true_solution = true_solution.to(device)
    initial_state = true_solution[0].to(device)
    
    minibatch_size = 128
    # dynamics=torch.cat((time_points,true_solution,control_inputs, disturbance_inputs)).to(device)
    minibatch_generator = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(time_points, true_solution),
        batch_size=minibatch_size,
        shuffle=False,
    )

    scaler = GradScaler()
    start_time = time.time()

    for epoch in range(epochs):
        for batch_time_points, batch_true_solution in minibatch_generator:
            if np.random.rand() < 0.1:
                pass
            else:
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
                        batch_true_solution[0].to(device),
                        batch_time_points.to(device),
                        method="rk4",
                        rtol=1e-10, atol=1e-9
                    )
                    loss = torch.nn.L1Loss()(predicted_solution, batch_true_solution.to(device))
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        losses.append(loss.item())
        if epoch % info_freq == 0:
            elapsed_time = time.time() - start_time
            print(
                f"Epoch {epoch}, Loss: {loss.item()}, Time: {elapsed_time:.2f} seconds"
            )
            start_time = time.time()
        if epoch % plot_freq == 0:
            status_plot(ode_func, true_solution, time_points, insulin_input, meal_input, device)
            #save the model with date and time as the name
            plot_losses(losses)
            torch.save(ode_func.state_dict(), f"models/model_{time.strftime('%Y%m%d-%H%M%S')}.pt")


    return losses
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
    control_inputs = torch.tensor([insulin_input(t.item()) for t in time_points], dtype=torch.float32)
    disturbance_inputs = torch.tensor([meal_input(t.item()) for t in time_points], dtype=torch.float32)

    true_model = BergmanTrueDynamics()
    time_points, true_solution = generate_data(
        true_model, time_points, G0, X0, I0, insulin_input, meal_input
    )

    ode_func = NeuralODEFunc(state_dim=3, hidden_dim=50, ext_input_dim=2)
    optimizer = optim.Adam(ode_func.parameters(), lr=0.001, weight_decay=1e-4)
    losses = train(
        ode_func,
        time_points,
        true_solution,
        insulin_input,
        meal_input,
        optimizer,
        epochs=1000,
        plot_freq=20,
        info_freq=10,
    )

    status_plot(ode_func, true_solution, time_points, insulin_input, meal_input, device)
    
    plot_losses(losses)

# %%
