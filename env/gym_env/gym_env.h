#ifndef __GYM_ENV_H__
#define __GYM_ENV_H__

#include <string>
#include <vector>
#include <cstdio>
#include <memory>
#include <grpcpp/grpcpp.h>
#include "proto_out/gymEnv.grpc.pb.h"

#include "../env.h"
using std::string;

namespace rlcpp
{
    class Gym_Env : public Env
    {
    public:
        Gym_Env(string addr)
        {
            this->stub_ = gymEnv::GymService::NewStub(grpc::CreateChannel(addr, grpc::InsecureChannelCredentials()));
            this->emptyMsg.set_msg("empty");
        }

        void make(const string &gameName) override
        {
            gymEnv::Msg name;
            name.set_msg(gameName);
            grpc::ClientContext ctx;
            this->stub_->make(&ctx, name, &this->envSpace);
        }

        Space action_space() const override
        {
            auto space = this->envSpace.action_space();
            // TODO: add support for Box type
            assert(space.bdiscrete() == true);
            Space ret;
            ret.bDiscrete = true;
            ret.n = space.n();
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
                ret.shape = {space.shape().begin(), space.shape().end()};
                ret.high = {space.high().begin(), space.high().end()};
                ret.low = {space.low().begin(), space.low().end()};
            }
            return ret;
        }

        void step(const Action &action, State *next_obs, double *reward, bool *done) override
        {
            grpc::ClientContext ctx;
            gymEnv::Action act;
            act.set_action(action);
            this->stub_->step(&ctx, act, &this->stepResult);
            this->stepResult.next_obs().obs();
            std::copy(this->stepResult.next_obs().obs().begin(), this->stepResult.next_obs().obs().end(), next_obs->begin());
            *reward = this->stepResult.reward();
            *done = this->stepResult.done();
        }

        void reset(State *obs) override
        {
            grpc::ClientContext ctx;
            gymEnv::Observation tmp;
            this->stub_->reset(&ctx, this->emptyMsg, &tmp);
            std::copy(tmp.obs().begin(), tmp.obs().end(), obs->begin());
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

        bool bDiscrete_action_space() override
        {
            return this->envSpace.action_space().bdiscrete();
        }

        bool bDiscrete_obs_space() override
        {
            return this->envSpace.obs_space().bdiscrete();
        }

    private:
        std::unique_ptr<gymEnv::GymService::Stub> stub_;
        gymEnv::EnvSpace envSpace;
        gymEnv::Msg emptyMsg;
        gymEnv::StepResult stepResult;
    }; // !class Gym_Env

} // !namepsace rlcpp

#endif // !__GYM_ENV_H__
