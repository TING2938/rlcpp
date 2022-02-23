#include "load_dynet_model.hpp"
#include "test/dynet/data-io.h"

using namespace std;

struct Net : public torch::nn::Module
{
    Net() : fc1(784, 512), fc2(512, 512), fc3(512, 10)
    {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }

    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
};

void test(Net& model, vector<vector<float>>& mnist_dev, vector<unsigned>& labels)
{
    model.eval();

    double dpos = 0;
    for (unsigned i = 0; i < mnist_dev.size(); ++i) {
        // build graph for this instance
        auto x      = torch::from_blob(mnist_dev[i].data(), 784);
        auto pred_y = model.forward(x);

        // Increment count of positive classification
        if (pred_y.argmax().item().toLong() == labels[i])
            dpos++;
    }

    // Print informations
    cerr << "E = " << (dpos / (double)mnist_dev.size()) << '\n';
}

int main(int argc, char** argv)
{
    torch::manual_seed(1);

    Net model;

    rlcpp::load_from_dynet(model,
                           "/home/yeting/work/project/rlcpp/build/"
                           "encdec_mlp_784-512-relu-0.2_512-512-relu-0.2_512-10-softmax_14523.params");


    // Load Dataset ----------------------------------------------------------------------------------
    // Load data
    string dev_file        = "data/t10k-images.idx3-ubyte";
    string dev_labels_file = "data/t10k-labels.idx1-ubyte";

    vector<vector<float>> mnist_dev;
    vector<unsigned> mnist_dev_labels;

    read_mnist(dev_file, mnist_dev);
    read_mnist_labels(dev_labels_file, mnist_dev_labels);


    // test the model
    test(model, mnist_dev, mnist_dev_labels);
}
