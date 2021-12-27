// copy from https://gitlab.com/TING2938/gym_cpp.git

#include <Python.h>

#include "env/env.h"

namespace rlcpp
{
    class Gym_cpp : public Env 
    {
    public:
        Gym_cpp()
        {
            Py_Initialize();
        }

        ~Gym_cpp()
        {
            Py_Finalize();
        }

        void make(const string &game_name) override
        {
            auto gym_wapper = PyImport_ImportModule("gym_wrapper");
            
            this->env_ = PyObject_CallObject(PyObject_GetAttrString(gym_wapper, "Gym_wrapper"), NULL);
            PyObject_CallMethod(this->env_, "make", "s", game_name.c_str());
            this->env_step = PyObject_GetAttrString(this->env_, "step");
            this->env_reset = PyObject_GetAttrString(this->env_, "reset");
            this->env_render = PyObject_GetAttrString(this->env_, "render");
            
            this->max_episode_steps = PyLong_AsLong(PyObject_GetAttrString(this->env_, "max_episode_steps"));
            
            this->action_space_.bDiscrete = PyObject_IsTrue(PyObject_GetAttrString(this->env_, "bDiscrete_act"));
            if (this->action_space_.bDiscrete)
            {
                this->action_space_.n = PyLong_AsLong(PyObject_GetAttrString(this->env_, "act_n"));
            } else {
                this->getList_int(&this->action_space_.shape, PyObject_GetAttrString(this->env_, "act_shape"));
                this->getList_float(&this->action_space_.high, PyObject_GetAttrString(this->env_, "act_high"));
                this->getList_float(&this->action_space_.low, PyObject_GetAttrString(this->env_, "act_low"));
            }

            this->obs_space_.bDiscrete = PyObject_IsTrue(PyObject_GetAttrString(this->env_, "bDiscrete_obs")); 
            if (this->obs_space_.bDiscrete)
            {
                this->obs_space_.n = PyLong_AsLong(PyObject_GetAttrString(this->env_, "obs_n"));
            } else {
                this->getList_int(&this->obs_space_.shape, PyObject_GetAttrString(this->env_, "obs_shape"));
                this->getList_float(&this->obs_space_.high, PyObject_GetAttrString(this->env_, "obs_high"));
                this->getList_float(&this->obs_space_.low, PyObject_GetAttrString(this->env_, "obs_low"));
            }
        }

        Space action_space() const override
        {
            return this->action_space_;
        }

        Space obs_space() const override
        {
            return this->obs_space_;
        }

        void reset(State *obs) override
        {
            auto pobs = PyObject_CallObject(this->env_reset, NULL); 
            this->getList_float(obs, pobs);
        }

        void close() override
        {
            PyObject_CallMethod(this->env_, "close", NULL);
        }

        void step(const Action &action, State *next_obs, Float *reward, bool *done) override
        {
            PyObject* ret;
            auto plist = PyList_New(action.size());
            for (ssize_t i = 0; i < action.size(); i++)
            {
                PyList_SetItem(plist, i, PyFloat_FromDouble(action[i]));
            }
            auto argList = PyTuple_New(1);
            PyTuple_SetItem(argList, 0, plist);
            ret = PyObject_CallObject(this->env_step, argList);
            Py_DecRef(argList);

            this->getList_float(next_obs, PyTuple_GetItem(ret, 0));
            *reward = PyFloat_AsDouble(PyTuple_GetItem(ret, 1));
            *done = PyObject_IsTrue(PyTuple_GetItem(ret, 2));
        }

        void render() override
        {
            PyObject_CallObject(this->env_render, NULL);
        }

    private:
        void getList_float(Vecf* vec, PyObject* plist)
        {
            auto len = PyList_Size(plist);
            vec->resize(len);
            for (ssize_t i = 0; i < len; i++)
            {
                (*vec)[i] = PyFloat_AsDouble(PyList_GetItem(plist, i));
            }
        }

        void getList_int(Veci* vec, PyObject* plist)
        {
            auto len = PyList_Size(plist);
            vec->resize(len);
            for (ssize_t i = 0; i < len; i++)
            {
                (*vec)[i] = PyLong_AsLong(PyList_GetItem(plist, i));
            }
        }
    private:
        Space action_space_;
        Space obs_space_;
        
        PyObject *env_ = NULL;
        PyObject *env_reset = NULL;
        PyObject *env_step = NULL;
        PyObject *env_render = NULL;
    }; // !class PyGym
} // !namespace rlcpp
