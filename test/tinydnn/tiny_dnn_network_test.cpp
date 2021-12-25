#include <iostream>
#include <string>
#include "network/tiny_dnn/tiny_dnn_network.h"

using fc = tiny_dnn::fully_connected_layer;
using relu = tiny_dnn::relu_layer;

int main()
{
    std::cout << "hello world" << std::endl;

    rlcpp::TinyDNN_Network<tiny_dnn::mse> network;
    network.nn << fc(1, 100) << relu()
               << fc(100, 10) << relu()
               << fc(10, 1);

    rlcpp::TinyDNN_Network<tiny_dnn::mse> other;
    other.nn << fc(1, 100) << relu()
             << fc(100, 10) << relu()
             << fc(10, 1);

    bool bb = other.nn.has_same_weights(network.nn, 0.0001);
    std::cout << "same: " << bb << '\n'
              << network.nn.layer_size() << '\n'
              << network.nn.in_data_size() << '\n'
              << network.nn.out_data_size() << '\n';

    other.update_weights_from(&network);
    bb = other.nn.has_same_weights(network.nn, 0.0001);
    std::cout << "other same: " << bb << '\n'
              << other.nn.layer_size() << '\n'
              << other.nn.in_data_size() << '\n'
              << other.nn.out_data_size() << '\n';

    // create input and desired output on a period
    std::vector<tiny_dnn::vec_t> X;
    std::vector<tiny_dnn::vec_t> sinusX;
    for (float x = -3.1416f; x < 3.1416f; x += 0.2f)
    {
        tiny_dnn::vec_t vx = {x};
        tiny_dnn::vec_t vsinx = {sinf(x)};

        X.push_back(vx);
        sinusX.push_back(vsinx);
    }

    // set learning parameters
    size_t batch_size = 32; // 16 samples for each network weight update
    int epochs = 5000;      // 2000 presentation of all samples
    network.minibatch_size = batch_size;
    network.nepochs = epochs;

    network.learn(X, sinusX);
    bb = other.nn.has_same_weights(network.nn, 0.0001);
    std::cout << "other same: " << bb << '\n';

    other.update_weights_from(&network);
    bb = other.nn.has_same_weights(network.nn, 0.0001);
    std::cout << "update, then other same: " << bb << '\n';

    // compare prediction and desired 
    std::vector<tiny_dnn::vec_t> xv;
    tiny_dnn::vec_t yv;
    for (float x = -3.1416f; x < 3.1416f; x += 0.2f)
    {
        xv.push_back({x});
        yv.push_back(sinf(x));
    }
    std::vector<tiny_dnn::vec_t> pred(xv.size(), tiny_dnn::vec_t(1, 0));
    other.predict(xv, &pred);
    for (int i = 0; i < xv.size(); i++)
    {
        printf("yv: %.5f   pred: %.5f\n", yv[i], pred[i][0]);
    }
}
