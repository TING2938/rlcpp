#include <iostream>
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"
#include "dynet/training.h"
#include "tools/vector_tools.h"

using namespace std;
using namespace dynet;
using namespace rlcpp::opt;

// [low, up]
dynet::real getRand(dynet::real low, dynet::real up)
{
    return dynet::real(rand()) / dynet::real(RAND_MAX) * (up - low) + low;
}

int main(int argc, char** argv)
{
    // srand(45667);

    // prepare data
    int ntrain = 500;
    int ntest  = 100;

    dynet::real p1 = 32.32;
    dynet::real p2 = 25.66;

    std::vector<dynet::real> train_x(ntrain), train_y(ntrain), test_x(ntest), test_y(ntest);
    for (int i = 0; i < ntrain; i++) {
        train_x[i] = i * getRand(-1, 1);
        train_y[i] = p1 * train_x[i] + p2;
    }
    for (int i = 0; i < ntest; i++) {
        test_x[i] = 5 * i + 3;
        test_y[i] = p1 * test_x[i] + p2;
    }

    dynet::real ymean = rlcpp::mean(train_y);
    dynet::real ystd  = rlcpp::stddev(train_y);
    train_y -= ymean;
    train_y /= ystd;
    train_x -= ymean;
    train_x /= ystd;

    dynet::initialize(argc, argv);

    const unsigned ITERATIONS = 200;

    // ParameterCollection (all the model parameters).
    ParameterCollection m;
    SimpleSGDTrainer trainer(m);


    Parameter p_a = m.add_parameters({1});
    Parameter p_b = m.add_parameters({1});
    if (argc == 2) {
        // Load the model and parameters from file if given.
        TextFileLoader loader(argv[1]);
        loader.populate(m);
    }

    // Static declaration of the computation graph.
    ComputationGraph cg;

    auto dist = dynet::random_normal(cg, {2}, 15, 1);
    std::cout << cg.forward(dist) << std::endl;

    Expression a = parameter(cg, p_a);
    Expression b = parameter(cg, p_b);

    // Set x_values to change the inputs to the network.
    dynet::real x_value;
    Expression x = input(cg, &x_value);
    dynet::real y_value;  // Set y_value to change the target output.
    Expression y = input(cg, &y_value);

    Expression y_pred    = a * x + b;
    Expression loss_expr = squared_distance(y_pred, y);

    // Show the computation graph, just for fun.
    // cg.print_graphviz();

    // Train the parameters.
    for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
        double loss = 0;
        for (unsigned mi = 0; mi < ntrain; ++mi) {
            x_value = train_x[mi];
            y_value = train_y[mi];
            loss += as_scalar(cg.forward(loss_expr));
            cg.backward(loss_expr);
            trainer.update();
        }
        loss /= ntrain;
        // cerr << "E = " << loss << endl;
    }

    auto aaa = as_scalar(a.value());
    auto bbb = as_scalar(b.value()) * ystd - (aaa - 1) * ymean;
    printf("a=%f, b=%f\n", aaa, bbb);
}
