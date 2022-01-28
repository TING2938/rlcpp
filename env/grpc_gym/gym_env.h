#ifndef __GYM_ENV_H__
#define __GYM_ENV_H__

#include <string>
#include <vector>
#include <cstdio>
#include <memory>
#include <grpcpp/grpcpp.h>
#include "proto_out/gymEnv.grpc.pb.h"

#include "env/env.h"
using std::string;

namespace rlcpp
{
    class Gym_gRPC : public Env
    {
    public:
        Gym_gRPC(string addr)
        {
            this->stub_ = gymEnv::GymService::NewStub(grpc::CreateChannel(addr, grpc::InsecureChannelCredentials()));
            this->emptyMsg.set_msg("empty");
        }

        void make(const string &gameName) override
        {
            gymEnv::Msg name;
            name.set_msg(gameName);
            grpc::ClientContext ctx;
            auto status = this->stub_->make(&ctx, name, &this->envSpace);
            if (!status.ok()) {
                std::cout << status.error_code() << ": " << status.error_message() << std::endl;
                std::exit(-1);
            }
            this->max_episode_steps = this->envSpace.max_episode_steps();
        }

        Space action_space() const override
        {
            auto space = this->envSpace.action_space();
            Space ret;
            if (space.bdiscrete())
            {
                ret.bDiscrete = true;
                ret.n = space.n();
            }
            else
            {
                ret.bDiscrete = false;
                ret.shape = { ALL(space.shape()) };
                ret.high = { ALL(space.high()) };
                ret.low = { ALL(space.low()) };
            }
            return ret;
        }

        Space obs_space() const override
        {
            auto space = this->envSpace.obs_space();
            Space ret;
            if (space.bdiscrete())
            {
                ret.bDiscrete = true;
                ret.n = space.n();
            }
            else
            {
                ret.bDiscrete = false;
                ret.shape = { ALL(space.shape()) };
                ret.high = { ALL(space.high()) };
                ret.low = { ALL(space.low()) };
            }
            return ret;
        }

        void step(const Action &action, State *next_obs, Real *reward, bool *done) override
        {
            grpc::ClientContext ctx;
            gymEnv::Action act;

            #if RLCPP_ACTION_TYPE == 0
                act.add_action(action); 
            #elif RLCPP_ACTION_TYPE == 1
                *act.mutable_action() = { ALL(action) }; 
            #endif

            this->stub_->step(&ctx, act, &this->stepResult);
            std::copy( ALL(this->stepResult.next_obs().obs()), next_obs->begin());
            *reward = this->stepResult.reward();
            *done = this->stepResult.done();
        }

        void reset(State *obs) override
        {
            grpc::ClientContext ctx;
            gymEnv::Observation tmp;
            this->stub_->reset(&ctx, this->emptyMsg, &tmp);
            std::copy( ALL(tmp.obs()), obs->begin());
        }

        void close() override
        {
            grpc::ClientContext ctx;
            this->stub_->close(&ctx, this->emptyMsg, &this->emptyMsg);
        }

        void render() override
        {
            grpc::ClientContext ctx;
            this->stub_->render(&ctx, this->emptyMsg, &this->emptyMsg);
        }

    private:
        std::unique_ptr<gymEnv::GymService::Stub> stub_;
        gymEnv::EnvSpace envSpace;
        gymEnv::Msg emptyMsg;
        gymEnv::StepResult stepResult;
    }; // !class Gym_Env

} // !namepsace rlcpp

#endif // !__GYM_ENV_H__
