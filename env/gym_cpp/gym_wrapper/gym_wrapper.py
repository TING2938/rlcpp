import gym 

class Gym_wrapper:
    def make(self, game_name: str):
        self.env = gym.make(game_name)
        self.max_episode_steps=self.env.spec.max_episode_steps

        obs_space_t = self.env.observation_space 
        act_space_t = self.env.action_space
        
        if len(obs_space_t.shape) == 0:
            self.bDiscrete_obs = True
            self.obs_n=obs_space_t.n
        else:
            self.bDiscrete_obs = False
            self.obs_shape = list(obs_space_t.shape)
            self.obs_low=list(obs_space_t.low)
            self.obs_high=list(obs_space_t.high)
        if len(act_space_t.shape) == 0:
            self.bDiscrete_act = True
            self.act_n=act_space_t.n
        else:
            self.bDiscrete_act = False
            self.act_shape=list(act_space_t.shape)
            self.act_low=list(act_space_t.low)
            self.act_high=list(act_space_t.high)

    def reset(self) -> list:
        obs = self.env.reset()
        if self.bDiscrete_obs:
            obs = [float(obs)]
        return list(obs)
        
    def step(self, act: list):
        if self.bDiscrete_act:
            act = int(act[0])
        next_obs, reward, done, _ = self.env.step(act)
        if self.bDiscrete_obs:
            next_obs = [float(next_obs)]
        return list(next_obs), reward, done

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

