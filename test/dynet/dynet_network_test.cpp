
#include "network/dynet_network/dynet_network.h"

using dynet::Expression;

void getTrainData(std::vector<float>& vx, std::vector<float>& vy)
{
    vx.clear();
    vy.clear();
    for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {
        vx.push_back(x);
        vy.push_back(sinf(x));
    }
}

int main(int argc, char** argv)
{
    dynet::initialize(argc, argv);

    std::vector<dynet::Layer> layers = {
        dynet::Layer(/* input_dim */ 1, /* output_dim */ 100, /* activation */ dynet::RELU, /* dropout_rate */ 0.0),
        dynet::Layer(/* input_dim */ 100, /* output_dim */ 100, /* activation */ dynet::RELU, /* dropout_rate */ 0.0),
        dynet::Layer(/* input_dim */ 100, /* output_dim */ 1, /* activation */ dynet::LINEAR, /* dropout_rate */ 0.0)};

    rlcpp::Dynet_Network network;
    network.build_model(layers);
    dynet::SimpleSGDTrainer trainer(network.model);

    rlcpp::Dynet_Network other;
    other.build_model(layers);

    other.update_weights_from(&network);

    std::vector<float> X;
    std::vector<float> sinusX;
    getTrainData(X, sinusX);

    int batch_size       = 16;
    unsigned num_batches = X.size() / batch_size;

    std::vector<Expression> cur_batch;
    std::vector<Expression> cur_y_batch;

    std::vector<unsigned> order(num_batches);
    for (unsigned i = 0; i < num_batches; ++i)
        order[i] = i;

    for (int eposide = 0; eposide < 10000; eposide++) {
        std::random_shuffle(order.begin(), order.end());

        double loss        = 0;
        double num_samples = 0;

        network.nn.enable_dropout();

        for (int si = 0; si < num_batches; si++) {
            dynet::ComputationGraph cg;

            int id         = order[si] * batch_size;
            unsigned bsize = std::min<unsigned>(X.size() - id, batch_size);

            cur_batch   = std::vector<Expression>(bsize);
            cur_y_batch = std::vector<Expression>(bsize);
            for (unsigned idx = 0; idx < bsize; idx++) {
                cur_batch[idx]   = dynet::input(cg, &X[id + idx]);
                cur_y_batch[idx] = dynet::input(cg, &sinusX[id + idx]);
            }
            Expression x_batch  = dynet::reshape(dynet::concatenate_cols(cur_batch), dynet::Dim({1}, bsize));
            Expression y_batch  = dynet::reshape(dynet::concatenate_cols(cur_y_batch), dynet::Dim({1}, bsize));
            Expression y_pred   = network.nn.run(x_batch, cg);
            Expression losses   = dynet::squared_distance(y_pred, y_batch);
            Expression sum_loss = dynet::sum_batches(losses) / bsize;

            loss += dynet::as_scalar(cg.forward(sum_loss));
            cg.backward(sum_loss);
            trainer.update();
            num_samples += bsize;

            if (si == num_batches - 1 || (si + 1) % (num_batches / 2) == 0) {
                trainer.status();
                std::cerr << "E = " << (loss / num_samples) << " \n";
                loss        = 0;
                num_samples = 0;
            }
        }
    }

    std::vector<float> px, py;
    for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {
        px.push_back(x);
        py.push_back(sinf(x));
    }

    auto py_pred        = network.predict(px);
    auto other_py_pred1 = other.predict(px);
    other.update_weights_from(&network);
    auto other_py_pred2 = other.predict(px);

    for (int i = 0; i < px.size(); i++) {
        printf("py: %.5f   py_pred: %.5f  other_py_pred1: %.5f, other_py_pred2: %.5f\n", py[i], py_pred[i],
               other_py_pred1[i], other_py_pred2[i]);
    }
}
