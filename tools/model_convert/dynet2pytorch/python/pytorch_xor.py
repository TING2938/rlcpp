import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, hidden) -> None:
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, input):
        x = torch.tanh(self.fc1(input))
        x = self.fc2(x)
        return x


ITERATIONS = 2000
T = 1.
F = -1.

model = Model(8)
mseLoss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

model.train()
for iter in range(ITERATIONS):
    mloss = 0.0
    for mi in range(4):
        x1 = mi % 2
        x2 = (mi // 2) % 2
        x = [T if x1 else F, T if x2 else F]
        y = T if x1 != x2 else F
        x = Variable(torch.Tensor([x]))
        y = Variable(torch.Tensor([y]))
        optimizer.zero_grad()
        pred = model(x)
        loss = mseLoss(y, pred)
        loss.backward()
        optimizer.step()
        mloss += loss.data.numpy()
    mloss /= 4.
    print("loss: %0.9f" % mloss)

torch.save(model.state_dict(), "xor.pth")

eval_model = Model(8)
eval_model.load_state_dict(torch.load("xor.pth"))
x = torch.tensor([T, F])
print("TF", eval_model(x).data.numpy())
x = torch.tensor([F, F])
print("FF", eval_model(x).data.numpy())
x = torch.tensor([T, T])
print("TT", eval_model(x).data.numpy())
x = torch.tensor([F, T])
print("FT", eval_model(x).data.numpy())
