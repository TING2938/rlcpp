import numpy as np


class RandomReply:
    def __init__(self, fnm):
        with open(fnm, "r") as fid:
            self._parse_header(fid)
            data = np.loadtxt(fid)
            self.state = data[:, 0:self.state_length]
            self.next_state = data[:, self.state_length:2*self.state_length]
            self.action = data[:, 2*self.state_length]
            self.reward = data[:, 2*self.state_length+1]
            self.done = data[:, 2*self.state_length+2]

    def _parse_header(self, fid):
        line = fid.readline().split()
        self.total_size = int(line[3])
        line = fid.readline().split()
        self.state_type = int(line[2])
        self.state_length = int(line[4])
        line = fid.readline().split()
        self.action_type = int(line[2])
        self.action_length = int(line[4])
        fid.readline()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fnm = "/home/yeting/work/project/AIFan/aifan_data/aifan_data_2013.03.11_05.39/AIFan_DQN_MemoryBuffer_2_3.dat"

    memory = RandomReply(fnm)

    print("finish reading memory data")

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(memory.reward)
    plt.show()
