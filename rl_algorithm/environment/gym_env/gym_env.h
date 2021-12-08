#ifndef __GYM_ENV_H__
#define __GYM_ENV_H__

#include <string>
#include <vector>
#include <cstdio>
#include <memory>
#include <grpcpp/grpcpp.h>
#include "proto_out/gymEnv.grpc.pb.h"

#include "../../env.h"

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
        this->stub_->make(&this->ctx, name, &this->envSpace);
    }

    Int action_space() const override
    {
        auto space = this->envSpace.action_space();
        // TODO: add support for Box type 
        assert(space.bdiscrete() == true);
        return space.n();
    }

    Veci obs_space() const override
    {
        auto space = this->envSpace.obs_space();
        if (space.bdiscrete()) 
        {
            return {0, space.n()};
        } else 
        {
            return {space.shape().begin(), space.shape().end()};
        }
    }

    void step(const Action& action, State* next_obs, double* reward, bool* done) override
    {
        gymEnv::Action act;
        act.set_action(action);
        this->stub_->step(&this->ctx, act, &this->stepResult);
        this->stepResult.next_obs().obs();
        std::copy(this->stepResult.next_obs().obs().begin(), this->stepResult.next_obs().obs().end(), next_obs->begin());
        *reward = this->stepResult.reward();
        *done = this->stepResult.done();
    }

    void reset(State* obs) override
    {
        gymEnv::Observation tmp;
        this->stub_->reset(&this->ctx, this->emptyMsg, &tmp);
        std::copy(tmp.obs().begin(), tmp.obs().end(), obs->begin());
    }

    void close() override
    {
        this->stub_->close(&this->ctx, this->emptyMsg, &this->emptyMsg);
    }

    void render() override
    {
        this->stub_->render(&this->ctx, this->emptyMsg, &this->emptyMsg);
    }

private:
    std::unique_ptr<gymEnv::GymService::Stub> stub_;
    grpc::ClientContext ctx;
    gymEnv::EnvSpace envSpace;
    gymEnv::Msg emptyMsg;
    gymEnv::StepResult stepResult;
};

#endif // !__GYM_ENV_H__
