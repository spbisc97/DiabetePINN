# %%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# %%
# Define the model with disturbances and controls
def bergman_model_with_control(y, t, p, D, U):
    G, X, I = y
    V1 = 12# % L
    n = 5/54#; % min
    dGdt = -p["S_G"] * (G - p["G_b"]) - (X-p["X_b"]) * G + D(t)
    dXdt = -p["S_X"] * (X-p["X_b"]) + p["S_I"] * (I - p["I_b"])
    dIdt = -n * I + U(t)/V1
    return [dGdt, dXdt,dIdt]

# %%
# Control and disturbance functions
def meal_input(t):
    if 60 <= t < 120:
        return 10  # mg/dL per minute input
    return 0

def insulin_input(t):
    if t == 30:
        return 1  # mU/L increase
    return 0

# %%
# Initial conditions and parameters
initial_conditions = [300,0.0, 15.0]  # Initial glucose and insulin levels
params = {"S_G": 0.028735,"S_X": 0.028344, "S_I": 5.035e-5,'I_b':15.0,"X_b":15.0,"G_b":100.0}  # Model parameters for Glucose Effectiveness and Insulin Sensitivity ; % mU/L

# Time points
t = np.linspace(0, 1000, 1001)  # 8.3 hours, one measurement per minute

# %%
# Solve ODE with controls and disturbances
solution = odeint(
    bergman_model_with_control,
    initial_conditions,
    t,
    args=(params, meal_input, insulin_input),
)

# %%
# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(t, solution[:, 0], label="Glucose", color='b')
plt.plot(t, solution[:, 1], label="Insulin", color='g')

# Adding markers for meal and insulin events
# Plot meal times
meal_times = [time for time in t if 60 <= time < 120]
meal_glucose = [meal_input(time) for time in meal_times]
plt.scatter(meal_times, [max(solution[:, 0]) + 20] * len(meal_times), c='orange', label='Meal Intake', marker='o')

# Plot insulin injections
insulin_times = [time for time in t if insulin_input(time) > 0]
insulin_levels = [insulin_input(time) for time in insulin_times]
plt.scatter(insulin_times, [max(solution[:, 1]) + 1] * len(insulin_times), c='red', label='Insulin Injection', marker='^')

plt.xlabel("Time (minutes)")
plt.ylabel("Concentration")
plt.title("Glucose and Insulin dynamics with controls and disturbances")
plt.legend()
plt.grid(True)
plt.show()
# %%
