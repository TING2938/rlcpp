from time import sleep
import gym
import numpy as np


class QLearning_Agent:
    def __init__(self, obs_n, act_n, lr=0.01, gamma=0.9, e_greedy=0.1) -> None:
        self.obs_n = obs_n
        self.act_n = act_n
        self.lr = lr
        self.gamma = gamma
        self.e_greedy = e_greedy
        self.Q = np.zeros((obs_n, act_n))

    def sample(self, obs, greedy=True):
        e_greedy = self.e_greedy if greedy else 0
        if np.random.random() < 1 - e_greedy:
            Q_list = self.Q[obs, :]
            return Q_list.argmax()
        else:
            return np.random.randint(self.act_n)

    def learn(self, obs, action, reward, next_obs, done):
        predict_Q = self.Q[obs, action]
        target_Q = reward
        if not done:
            target_Q += self.gamma * self.Q[next_obs, :].max()
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)


def run_episode(env: gym.Env, agent: QLearning_Agent, bRender: bool = False):
    total_steps = 0
    total_reward = 0.0
    obs = env.reset()
    while True:
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.learn(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward
        total_steps += 1
        if bRender:
            env.render()
        if done:
            break
    return total_reward, total_steps


def test_episode(env: gym.Env, agent: QLearning_Agent):
    total_reward = 0.0
    obs = env.reset()
    while True:
        action = agent.sample(obs, True)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        total_reward += reward
        sleep(1)
        env.render()
        if done:
            print(f"test reward = {total_reward:.1f}")
            break


def main():
    env = gym.make("CliffWalking-v0")

    agent = QLearning_Agent(env.observation_space.n,
                            env.action_space.n, lr=0.1, gamma=0.9, e_greedy=0.1)

    bRender = False
    for episode in range(10000):
        total_reward, total_steps = run_episode(env, agent, bRender)
        print(
            f"Episode {episode}: steps = {total_steps}, reward = {total_reward:.1f}")
        if episode % 20 == 0:
            bRender = True
        else:
            bRender = False

    print(agent.Q)
    test_episode(env, agent)


main()
