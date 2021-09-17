from state import State
from action import Action
from reward import Reward

class ECN_Envirenment:
    def __init__(self, envModel):
        self.envModel = envModel
        self.state_space = State(k=3)
        self.action_space = Action(alpha=20)
        self.rewardAgent = Reward(omega=(0.7, 0.3))
        self.max_episode = 1000
    
    def step(self, action: int):
        next_state, RTT, outputAvgSpeed = self.envModel(self.action_space.to_KminKmaxPmax(action))
        reward = self.rewardAgent.getReward(RTT, outputAvgSpeed)
        done = True if reward < 1000.0 else False 
        return next_state, reward, done

    def reset(self):
        return self.state_space.reset()

    def render(self):
        pass 

    def close(self):
        pass 

