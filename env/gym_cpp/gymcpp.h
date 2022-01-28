// copy from https://gitlab.com/TING2938/gym_cpp.git

#define PY_SSIZE_T_CLEAN
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
            Py_DecRef(this->env_render);
            Py_DecRef(this->env_reset);
            Py_DecRef(this->env_step);
            Py_DecRef(this->env_);
            Py_Finalize();
        }

        void make(const string &game_name) override
        {
            auto gym_wapper = PyImport_ImportModule("gym_wrapper");

            auto Gym_wrapper = PyObject_GetAttrString(gym_wapper, "Gym_wrapper");
            this->env_ = PyObject_CallObject(Gym_wrapper, NULL);
            PyObject_CallMethod(this->env_, "make", "s", game_name.c_str());
            this->env_step = PyObject_GetAttrString(this->env_, "step");
            this->env_reset = PyObject_GetAttrString(this->env_, "reset");
            this->env_render = PyObject_GetAttrString(this->env_, "render");

            auto p_max_episode_steps = PyObject_GetAttrString(this->env_, "max_episode_steps");
            this->max_episode_steps = PyLong_AsLong(p_max_episode_steps);

            auto p_bDiscrete_act = PyObject_GetAttrString(this->env_, "bDiscrete_act");
            this->action_space_.bDiscrete = PyObject_IsTrue(p_bDiscrete_act);
            if (this->action_space_.bDiscrete)
            {
                auto p_act_n = PyObject_GetAttrString(this->env_, "act_n");
                this->action_space_.n = PyLong_AsLong(p_act_n);
                Py_DecRef(p_act_n);
            }
            else
            {
                auto p_act_shape = PyObject_GetAttrString(this->env_, "act_shape");
                auto p_act_high = PyObject_GetAttrString(this->env_, "act_high");
                auto p_act_low = PyObject_GetAttrString(this->env_, "act_low");
                this->getList(&this->action_space_.shape, p_act_shape);
                this->getList(&this->action_space_.high, p_act_high);
                this->getList(&this->action_space_.low, p_act_low);
                Py_DecRef(p_act_shape);
                Py_DecRef(p_act_high);
                Py_DecRef(p_act_low);
            }

            auto p_bDiscrete_obs = PyObject_GetAttrString(this->env_, "bDiscrete_obs");
            this->obs_space_.bDiscrete = PyObject_IsTrue(p_bDiscrete_obs);
            if (this->obs_space_.bDiscrete)
            {
                auto p_obs_n = PyObject_GetAttrString(this->env_, "obs_n");
                this->obs_space_.n = PyLong_AsLong(p_obs_n);
                Py_DecRef(p_obs_n);
            }
            else
            {
                auto p_obs_shape = PyObject_GetAttrString(this->env_, "obs_shape");
                auto p_obs_high = PyObject_GetAttrString(this->env_, "obs_high");
                auto p_obs_low = PyObject_GetAttrString(this->env_, "obs_low");
                this->getList(&this->obs_space_.shape, p_obs_shape);
                this->getList(&this->obs_space_.high, p_obs_high);
                this->getList(&this->obs_space_.low, p_obs_low);
                Py_DecRef(p_obs_shape);
                Py_DecRef(p_obs_high);
                Py_DecRef(p_obs_low);
            }
            Py_DecRef(p_bDiscrete_obs);
            Py_DecRef(p_bDiscrete_act);
            Py_DecRef(p_max_episode_steps);
            Py_DecRef(Gym_wrapper);
            Py_DecRef(gym_wapper);
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

            this->getList(obs, pobs);
            Py_DecRef(pobs);
        }

        void close() override
        {
            PyObject_CallMethod(this->env_, "close", NULL);
        }

        void step(const Action &action, State *next_obs, Real *reward, bool *done) override
        {
            PyObject *ret;

            #if RLCPP_ACTION_TYPE == 0
            auto plist = PyList_New(1);
            PyList_SET_ITEM(plist, 0, PyLong_FromLong(action));
            #elif RLCPP_ACTION_TYPE == 1
            auto plist = PyList_New(action.size());
            for (ssize_t i = 0; i < action.size(); i++)
            {
                PyList_SET_ITEM(plist, i, PyFloat_FromDouble(action[i]));
            }
            #endif 

            auto argList = PyTuple_New(1);
            PyTuple_SetItem(argList, 0, plist);
            ret = PyObject_CallObject(this->env_step, argList);

            this->getList(next_obs, PyTuple_GetItem(ret, 0));
            *reward = PyFloat_AsDouble(PyTuple_GetItem(ret, 1));
            *done = PyObject_IsTrue(PyTuple_GetItem(ret, 2));
            Py_DecRef(argList);
            Py_DecRef(plist);
            Py_DecRef(ret);
        }

        void render() override
        {
            auto ret = PyObject_CallObject(this->env_render, NULL);
            Py_DecRef(ret);
        }

    private:
        void getList(Vecf *vec, PyObject *plist)
        {
            auto len = PyList_Size(plist);
            vec->resize(len);
            for (ssize_t i = 0; i < len; i++)
            {
                (*vec)[i] = PyFloat_AsDouble(PyList_GetItem(plist, i));
            }
        }

        void getList(Veci *vec, PyObject *plist)
        {
            auto len = PyList_Size(plist);
            vec->resize(len);
            for (ssize_t i = 0; i < len; i++)
            {
                (*vec)[i] = PyLong_AsLong(PyList_GetItem(plist, i));
            }
        }

        void getList(Int* val, PyObject* plist)
        {
            *val = PyLong_AsLong(PyList_GetItem(plist, 0));
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
