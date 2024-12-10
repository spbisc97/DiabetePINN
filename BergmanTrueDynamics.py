from torch import nn
import torch


class BergmanTrueDynamics(nn.Module):
    def __init__(self, p1=0.028735, p2=0.028344, p3=5.035e-5, G_b=100.0, I_b=15.0, I_X=15.0):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.G_b = G_b
        self.I_b = I_b
        self.I_X = I_X 
        self.V1 = 12# % L
        self.n = 5/54#; % min

    def forward(self, t, state, D=0, U=0):
        G, X, I = state
        dGdt = -self.p1 * (G - self.G_b) - (X-self.I_X) * G + D
        dXdt = -self.p2 * (X-self.I_X) + self.p3 * (I - self.I_b)
        dIdt = U/self.V1 - self.n*I
        return torch.stack([dGdt, dXdt, dIdt])