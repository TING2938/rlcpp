from concurrent import futures
import os
import grpc
import logging
import gym 
import argparse

from gymEnv_pb2_grpc import GymServiceServicer, add_GymServiceServicer_to_server
from gymEnv_pb2 import Observation, StepResult, Msg, EnvSpace, Space

class EnvServer(GymServiceServicer):
    def make(self, request, context):
        self.env = gym.make(request.msg)
        logging.info(f"make gym env: {request.msg}")

        obs_space_t = self.env.observation_space 
        act_space_t = self.env.action_space
        if len(obs_space_t.shape) == 0:
            self.bDiscrete_obs = True
            obs_space = Space(bDiscrete=True, n=obs_space_t.n)
        else:
            self.bDiscrete_obs = False
            obs_space = Space(bDiscrete=False, shape=obs_space_t.shape, low=obs_space_t.low, high=obs_space_t.high)
        if len(act_space_t.shape) == 0:
            self.bDiscrete_act = True
            action_space = Space(bDiscrete=True, n=act_space_t.n)
        else:
            self.bDiscrete_act = False
            action_space = Space(bDiscrete=False, shape=act_space_t.shape, low=act_space_t.low, high=act_space_t.high)
        logging.info(f"    max episode steps: {self.env.spec.max_episode_steps}")
        return EnvSpace(obs_space=obs_space, action_space=action_space, max_episode_steps=self.env.spec.max_episode_steps)

    def reset(self, request, context):
        obs = self.env.reset()
        if self.bDiscrete_obs:
            obs = [obs]
        return Observation(obs=obs)
        
    def step(self, request, context):
        if self.bDiscrete_act:
            action = int(request.action[0])
        else:
            action = request.action
        next_obs, reward, done, _ = self.env.step(action)
        if self.bDiscrete_obs:
            next_obs = [next_obs]
        return StepResult(next_obs=Observation(obs=next_obs), reward=reward, done=done)

    def render(self, request, context):
        self.env.render()
        return Msg(msg="")

    def close(self, reuest, context):
        self.env.close()
        logging.info(f"    close gym env: {self.env}")
        return Msg(msg="")

def serve(addr):
    # use thread pool to deal with the tasks of server
    server = grpc.server(thread_pool=futures.ThreadPoolExecutor(max_workers=1))

    # add tasks function to rpc server
    add_GymServiceServicer_to_server(servicer=EnvServer(), server=server)

    server.add_insecure_port(addr)
    server.start()
    server.wait_for_termination()
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S %p")

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--addr", default="[::]:50248", type=str, help="address for this listening")
    args = parser.parse_args()

    serve(addr=args.addr)



