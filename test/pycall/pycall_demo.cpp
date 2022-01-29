#include <Python.h>
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char* argv[])
{
    Py_Initialize();  //使用python之前，要调用Py_Initialize();这个函数进行初始化
    if (!Py_IsInitialized()) {
        printf("初始化失败！");
        return 0;
    }

    PyObject* pModule_time = NULL;  //声明变量
    PyObject* pFunc_time   = NULL;  // 声明变量
    PyObject* pFunc_ctime  = NULL;  // 声明变量

    pModule_time = PyImport_ImportModule("time");  //这里是要调用的文件名hash_hmac.py
    if (pModule_time == NULL) {
        cout << "没找到" << endl;
    }

    pFunc_ctime = PyObject_GetAttrString(pModule_time, "ctime");  //这里是要调用的函数名
    pFunc_time  = PyObject_GetAttrString(pModule_time, "time");   //这里是要调用的函数名

    PyObject* pret0 = PyObject_CallObject(pFunc_time, NULL);
    PyObject* pret1 = PyObject_CallFunctionObjArgs(pFunc_ctime, pret0, NULL);
    PyObject* pret2 = PyObject_CallMethod(pModule_time, "time", NULL);

    double res0;
    PyArg_Parse(pret0, "d", &res0);  //转换返回类型

    char* res1;
    PyArg_Parse(pret1, "s", &res1);

    cout << "res0:" << res0 << endl;  //输出结果
    cout << "res1:" << res1 << endl;  //输出结果

    Py_Finalize();  //调用Py_Finalize，这个根Py_Initialize相对应的。

    return 0;
}
