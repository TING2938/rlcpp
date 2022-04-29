import matplotlib.pyplot as plt
import numpy as np
import torch
import math

torch.manual_seed(22)

mu = 10
sigma = 100

# dist = torch.distributions.Normal(mu, sigma)
dist = torch.distributions.Normal(0, 1)


x = dist.sample((10,))
prob = dist.log_prob(x)

# prob1 = -(x - mu)**2 / (2*sigma**2) - math.log(math.sqrt(2*math.pi) * sigma)
prob2 = -x**2 / 2 - math.log(math.sqrt(2*math.pi) * sigma)

print("prob = ", prob)
print("prob2 = ", prob2)

# x = x * sigma + mu


plt.plot(x)
plt.show()
print(x)
