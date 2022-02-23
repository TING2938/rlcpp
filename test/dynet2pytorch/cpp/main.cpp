
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

void load_from_dynet(torch::nn::Module& model, std::string fnm, std::string key = "")
{
    auto parameters = model.parameters();
    std::string line, head;
    std::stringstream ss;
    int cur = 0;
    float value;
    std::ifstream fid(fnm);
    if (fid) {
        while (std::getline(fid, line)) {
            ss.str(line);
            ss >> head >> head;
            ss.clear();
            std::getline(fid, line);
            if (key == head.substr(0, head.rfind('/'))) {
                auto param_cur = parameters[cur].data();
                if (cur % 2 == 0) {
                    int row = param_cur.size(0);
                    ss.str(line);
                    int i = 0;
                    while (ss >> value) {
                        param_cur[i % row][i / row] = value;
                        i++;
                    }
                    ss.clear();
                } else {
                    ss.str(line);
                    int i = 0;
                    while (ss >> value) {
                        param_cur[i] = value;
                        i++;
                    }
                    ss.clear();
                }
                cur++;
            }
        }
    }
    fid.close();
}

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

    load_from_dynet(model, "/home/yeting/work/project/rlcpp/build/test/dynet/xor.model");

    test(model);
}
