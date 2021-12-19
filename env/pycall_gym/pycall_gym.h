
#include "env/env.h"
#include <Python.h>
#include "numpy/arrayobject.h"


namespace rlcpp
{
    class Pycall_Gym : public Env
    {
    public:
        Pycall_Gym()
        {
            Py_Initialize();
            this->init_numpy();
        }

        ~Pycall_Gym()
        {
            Py_Finalize();
        }

        void make(const string &game_name) override
        {
            auto gym = PyImport_ImportModule("gym");

            this->env_ = PyObject_CallMethod(gym, "make", "s", game_name.c_str());
            this->env_step = PyObject_GetAttrString(this->env_, "step");
            this->env_reset = PyObject_GetAttrString(this->env_, "reset");
            this->env_render = PyObject_GetAttrString(this->env_, "render");
            
            this->max_episode_steps = PyLong_AsSize_t(PyObject_GetAttrString(this->env_, "_max_episode_steps"));
            this->Parse_Shape(PyObject_GetAttrString(this->env_, "action_space"), &this->action_space_);
            this->Parse_Shape(PyObject_GetAttrString(this->env_, "observation_space"), &this->obs_space_);
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
            if (this->obs_space_.bDiscrete)
            {
                obs->front() = PyLong_AsSize_t(pobs);
            } else {
                auto npyobs = (PyArrayObject*)pobs; 
                for (ssize_t i = 0; i < obs->size(); i++)
                {
                    (*obs)[i] = *(float*)(npyobs->data + i * npyobs->strides[0]);
                }
            }
        }

        void close() override
        {
            PyObject_CallMethod(this->env_, "close", NULL);
        }

        void step(const Action &action, State *next_obs, Float *reward, bool *done) override
        {
            PyObject* ret;
            if (this->action_space_.bDiscrete) {
                ret = PyObject_CallFunction(this->env_step, "i", action.front());
            }
            else {
                auto plist = PyList_New(action.size());
                for (ssize_t i = 0; i < action.size(); i++)
                {
                    PyList_SET_ITEM(plist, i, PyFloat_FromDouble(action[i]));
                }
                auto argList = PyTuple_New(1);
                PyTuple_SetItem(argList, 0, plist);
                ret = PyObject_CallObject(this->env_step, argList);
            }
            *reward = PyFloat_AsDouble(PyTuple_GetItem(ret, 1));
            *done = PyObject_IsTrue(PyTuple_GetItem(ret, 2));

            if (this->obs_space_.bDiscrete)
            {
                next_obs->front() = PyLong_AsSize_t(PyTuple_GetItem(ret, 0));
            } else {
                auto npy_next_obs = (PyArrayObject*)PyTuple_GetItem(ret, 0);
                for (ssize_t i = 0; i < next_obs->size(); i++)
                {
                    (*next_obs)[i] = *(float*)(npy_next_obs->data + i * npy_next_obs->strides[0]);
                }
            }
        }

        void render() override
        {
            PyObject_CallObject(this->env_render, NULL);
        }
    private: 
        void Parse_Shape(PyObject* pspace, Space* space)
        {
            auto pspace_shape = PyObject_GetAttrString(pspace, "shape");
            auto shape_size = PyTuple_Size(pspace_shape);
            if (shape_size <= 0)
            {
                space->bDiscrete = true;
                space->n = PyFloat_AsDouble(PyObject_GetAttrString(pspace, "n"));
            } else {
                space->bDiscrete = false;
                space->shape.resize(shape_size);
                for (ssize_t i = 0; i < shape_size; i++)
                {
                    space->shape[i] = PyLong_AsSize_t(PyTuple_GetItem(pspace_shape, i));
                }
                auto plow = (PyArrayObject*)PyObject_GetAttrString(pspace, "low");
                auto phigh = (PyArrayObject*)PyObject_GetAttrString(pspace, "high");
                auto low_size = plow->dimensions[0];
                space->low.resize(low_size);
                space->high.resize(low_size);
                for (ssize_t i = 0; i < low_size; i++)
                {
                    space->low[i] = *(float*)(plow->data + i * plow->strides[0]);
                    space->high[i] = *(float*)(phigh->data + i * phigh->strides[0]);
                }
            }
        }

        int init_numpy() {
            import_array();
        }

    public:
        size_t max_episode_steps;
    private:
        Space action_space_;
        Space obs_space_;
        
        PyObject *env_ = NULL;
        PyObject *env_reset = NULL;
        PyObject *env_step = NULL;
        PyObject *env_render = NULL;
    }; // !class PyGym
} // !namespace rlcpp
