# %%
import dynet as dy
import torch
import torch.nn as nn


def torch_load_from_dynet(torch_state_dict, dynet_parameters_list):
    for p_torch, p_dynet in zip(torch_state_dict.values(), dynet_parameters_list):
        p_torch.copy_(torch.from_numpy(p_dynet.as_array()))


def dynet_load_from_torch(dynet_parameters_list, torch_state_dict):
    for p_dynet, p_torch in zip(dynet_parameters_list, torch_state_dict.values()):
        p_dynet.set_value(p_torch)


class Torch_Model(nn.Module):
    """
    torch model define
    """

    def __init__(self, hidden) -> None:
        super(Torch_Model, self).__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, input):
        x = torch.tanh(self.fc1(input))
        x = self.fc2(x)
        return x


class Dynet_Model:
    """
    dynet model define
    """

    def __init__(self, hidden) -> None:
        self.m = dy.Model()
        self.pW1 = self.m.add_parameters((hidden, 2))
        self.pb1 = self.m.add_parameters(hidden)
        self.pW2 = self.m.add_parameters((1, hidden))
        self.pb2 = self.m.add_parameters(1)

    def __call__(self, input):
        h1 = dy.tanh(self.pW1 * input + self.pb1)
        y_pred = self.pW2 * h1 + self.pb2
        return y_pred


T = 1.
F = -1.
HIDDEN_SIZE = 8

# %% 1. torch load from torch model
torch_model = Torch_Model(HIDDEN_SIZE)
torch_model.load_state_dict(torch.load("xor.pth"))
print("\n\n1. torch load from torch model")
x = torch.tensor([T, F])
print("TF", torch_model(x).data.numpy())
x = torch.tensor([F, F])
print("FF", torch_model(x).data.numpy())
x = torch.tensor([T, T])
print("TT", torch_model(x).data.numpy())
x = torch.tensor([F, T])
print("FT", torch_model(x).data.numpy())

# %% 2. dynet load from dynet model
dynet_model = Dynet_Model(HIDDEN_SIZE)
x = dy.vecInput(2)
y_pred = dynet_model(x)
dynet_model.m.populate("xor.pymodel")
print("\n\n2. dynet load from dynet model")
x.set([T, F])
print("TF", y_pred.scalar_value())
x.set([F, F])
print("FF", y_pred.scalar_value())
x.set([T, T])
print("TT", y_pred.scalar_value())
x.set([F, T])
print("FT", y_pred.scalar_value())

# %% 3. dynet load from torch model
dynet_model2 = Dynet_Model(HIDDEN_SIZE)
x = dy.vecInput(2)
y_pred = dynet_model2(x)
dynet_load_from_torch(dynet_model2.m.parameters_list(),
                      torch_model.state_dict())
print("\n\n3. dynet load from torch model")
x.set([T, F])
print("TF", y_pred.scalar_value())
x.set([F, F])
print("FF", y_pred.scalar_value())
x.set([T, T])
print("TT", y_pred.scalar_value())
x.set([F, T])
print("FT", y_pred.scalar_value())
dynet_model2.m.save("dynet_model2.model")

# %% 4. torch load from dynet model
torch_model2 = Torch_Model(HIDDEN_SIZE)
torch_load_from_dynet(torch_model2.state_dict(),
                      dynet_model.m.parameters_list())
print("\n\n4. torch load from dynet model")
x = torch.tensor([T, F])
print("TF", torch_model2(x).data.numpy())
x = torch.tensor([F, F])
print("FF", torch_model2(x).data.numpy())
x = torch.tensor([T, T])
print("TT", torch_model2(x).data.numpy())
x = torch.tensor([F, T])
print("FT", torch_model2(x).data.numpy())
torch.save(torch_model2.state_dict(), "torch_model2.pth")
