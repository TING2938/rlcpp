
import gym
import numpy as np
import tensorflow as tf
from gym.core import Env
from tensorflow.keras import losses, optimizers

from agent import MyAgent

LEARN_FREQ = 5  # 训练频率
MEMORY_SIZE = 20000  # replay memory 大小
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存的数据
BATCH_SIZE = 32   # 每次给agent learn 的数据量
LEARNING_RATE = 0.001  # 学习率
GAMMA = 0.99  # reward 的衰减因子，一般取0.9-0.99之间
# 训练一个episode


def run_episode(env: Env, agent: MyAgent):
    # print("run episode")
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        next_obs, reward, done, _ = env.step(action)
        reward_list.append(reward)
        obs = next_obs
        if done:
            break
    return obs_list, action_list, reward_list

# 评估agent，跑5个episode
def evaluate(env: Env, agent: MyAgent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list)-2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i+1]
    return np.array(reward_list)

def main():
    # CartPole-v0: expected reward > 180
    # MountainCar-v0 : expected reward > -120
    env = gym.make('CartPole-v0')
    action_dim = env.action_space.n  # CartPole-v0: 2
    obs_dim = env.observation_space.shape[0]  # CartPole-v0: (4,)

    agent = MyAgent(obs_n=obs_dim,
                    act_n=action_dim,
                    lr=LEARNING_RATE)
    
    # start train
    for episode in range(1000):
        obs_list, action_list, reward_list = run_episode(env, agent)
        if episode % 10 == 0:
            print(f"Episode: {episode}, Reward Sum: {sum(reward_list)}")

        obs_list = np.array(obs_list)
        action_list = np.array(action_list)
        reward_list = calc_reward_to_go(reward_list, gamma=1)
        agent.learn(obs_list, action_list, reward_list)

        if (episode + 1) % 100 == 0:
            total_reward = evaluate(env, agent, render=True)
            print(f"Test reward: {total_reward}") 
    env.close()

if __name__ == "__main__":
    main()
