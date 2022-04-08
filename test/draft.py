import matplotlib.pyplot as plt
import numpy as np

P = np.linspace(60, 130)

k2 = 0.187572
Gamma = 0.618
Target = 60
T_env = 32

u = (P / (k2 * (Target - T_env))) ** (1/Gamma)
Nx = u/2.0
action = Nx.copy()

for i, v in enumerate(Nx):
    if v >= 100:
        action[i] = 19
    elif v <= 20:
        action[i] = 0
    else:
        action[i] = int((v - 20) / 4.0)

plt.plot(P, action, "o--")

plt.show()
