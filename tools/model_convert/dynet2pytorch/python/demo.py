from __future__ import print_function
import torch
import torch.nn as nn

torch.manual_seed(1)


def load_from_dynet(state_dict, fnm, key=""):
    sd = iter(state_dict.values())
    cur = 0
    with open(fnm) as fid:
        for line in fid:
            head = line.split()
            line = fid.readline()
            if key == head[1][:head[1].rfind('/')]:
                params = [float(num) for num in line.split()]
                sd_cur = next(sd)
                if cur % 2 == 0:
                    row, _ = sd_cur.shape
                    for i, v in enumerate(params):
                        sd_cur[i % row, i // row] = v
                else:
                    for i, v in enumerate(params):
                        sd_cur[i] = v
                cur += 1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()

load_from_dynet(model.state_dict(),
                "/home/yeting/work/project/rlcpp/build/test/dynet/xor.model")


def test():
    model.eval()
    x_values = torch.Tensor([-1, -1])
    y_pred = model(x_values)
    print("[-1, -1]: ", y_pred)

    x_values[0] = -1
    x_values[1] = 1
    print("[-1, 1]: ", model(x_values))

    x_values[0] = 1
    x_values[1] = -1
    print("[1, -1]: ", model(x_values))

    x_values[0] = 1
    x_values[1] = 1
    print("[1, 1]: ", model(x_values))


test()
