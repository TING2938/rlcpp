#include "load_dynet_model.hpp"

struct Net : public torch::nn::Module
{
    Net() : fc1(2, 8), fc2(8, 1)
    {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::tanh(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};

void test(Net& model)
{
    model.eval();

    auto x_values = torch::tensor({-1.f, -1.f});
    std::cout << "x_values: \n" << x_values << std::endl;
    auto y_pred = model.forward(x_values);
    std::cout << "[-1, -1]: " << y_pred << "\n\n";

    x_values[0] = -1;
    x_values[1] = 1;
    std::cout << "x_values: \n" << x_values << std::endl;
    std::cout << "[-1, 1]: " << model.forward(x_values) << "\n\n";

    x_values[0] = 1;
    x_values[1] = -1;
    std::cout << "x_values: \n" << x_values << std::endl;
    std::cout << "[1, -1]: " << model.forward(x_values) << "\n\n";

    x_values[0] = 1;
    x_values[1] = 1;
    std::cout << "x_values: \n" << x_values << std::endl;
    std::cout << "[1, 1]: " << model.forward(x_values) << "\n\n";
}

int main(int argc, char** argv)
{
    torch::manual_seed(1);

    Net model;

    rlcpp::load_from_dynet(model, "/home/yeting/work/project/rlcpp/build/test/dynet/xor.model");

    test(model);
}
