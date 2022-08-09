from time import sleep
import gym
import numpy as np
from agent import DQN_Agent


def train(env: gym.Env, agent: DQN_Agent):
    rewards = []
    ma_rewards = []  # 滑动平均reward
    steps = []

    tot_episode = 200
    for i_episode in range(tot_episode):
        ep_reward = 0
        ep_step = 0
        obs = env.reset()
        while True:
            ep_step += 1
            action = agent.sample(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.store((obs, action, reward, next_obs, done))
            obs = next_obs
            for _ in range(10):
                agent.learn()
            ep_reward += reward
            if done:
                break
        if (i_episode + 1) % 4 == 0:
            agent.Q_target.load_state_dict(agent.Q.state_dict())
        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_episode + 1) % 1 == 0:
            print(
                f"Episode: {i_episode+1}/{tot_episode}, Reward: {ep_reward:.2f}, Step: {ep_step}")
    return rewards, ma_rewards, steps


def test_episode(env: gym.Env, agent: DQN_Agent):
    total_reward = 0.0
    obs = env.reset()
    while True:
        action = agent.sample(obs, True)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        total_reward += reward
        # sleep(1)
        env.render()
        if done:
            print(f"test reward = {total_reward:.1f}")
            break


def main():
    env = gym.make("CartPole-v1")

    agent = DQN_Agent(env.observation_space.shape[0],
                      env.action_space.n, batch_size=64, lr=0.0001, gamma=0.95, e_greedy=0.1)

    train(env, agent)

    for _ in range(50):
        test_episode(env, agent)


main()
