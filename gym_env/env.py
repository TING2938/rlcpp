from concurrent import futures
import grpc
import logging
import gym 
import argparse

from gymEnv_pb2_grpc import GymServiceServicer, add_GymServiceServicer_to_server
from gymEnv_pb2 import Action, Observation, StepResult, Msg, EnvSpace, Space

class EnvServer(GymServiceServicer):
    def make(self, request, context):
        self.env = gym.make(request.msg)
        self.bbox = self.env.observation_space 
        obs_space = Space()
        action_space = Space(bDiscrete=True)
        
        return EnvSpace(obs_space=obs_space, action_space=action_space)


    def reset(self, request, context):
        obs = self.env.reset()
        return Observation(obs=obs)

    def step(self, request, context):
        next_obs, reward, done, _ = self.env.step(request.action)
        return StepResult(next_obs=next_obs, reward=reward, done=done)

    def render(self, request, context):
        self.env.render()
        return Msg("")

    def close(self, reuest, context):
        self.env.close()
        return Msg("")

def serve(addr):
    # use thread pool to deal with the tasks of server
    server = grpc.server(thread_pool=futures.ThreadPoolExecutor(max_workers=8))

    # add tasks function to rpc server
    add_GymServiceServicer_to_server(servicer=EnvServer(), server=server)

    server.add_insecure_port(addr)
    server.start()
    server.wait_for_termination()
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S %p")

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--addr", default="[::]:50053", type=str, help="address for this listening")
    args = parser.parse_args()

    serve(addr=args.addr)



