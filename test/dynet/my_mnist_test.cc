/**
 * Train a multilayer perceptron to classify mnist digits
 *
 * This provide an example of usage of the mlp.h model
 */
#include "network/dynet_network/dynet_network.h"
#include "data-io.h"

using namespace std;
using namespace dynet;

int main(int argc, char **argv)
{
    dynet::initialize(argc, argv);

    // Load Dataset ----------------------------------------------------------------------------------
    // Load data
    string train_file = "data/train-images.idx3-ubyte";
    string dev_file = "data/t10k-images.idx3-ubyte";
    string train_labels_file = "data/train-labels.idx1-ubyte";
    string dev_labels_file = "data/t10k-labels.idx1-ubyte";
    unsigned BATCH_SIZE = 32;
    unsigned NUM_EPOCHS = 20;

    vector<vector<float>> mnist_train, mnist_dev;

    read_mnist(train_file, mnist_train);
    read_mnist(dev_file, mnist_dev);

    // Load labels
    vector<unsigned> mnist_train_labels, mnist_dev_labels;

    read_mnist_labels(train_labels_file, mnist_train_labels);
    read_mnist_labels(dev_labels_file, mnist_dev_labels);

    // Build model -----------------------------------------------------------------------------------
    rlcpp::Dynet_Network network;

    vector<Layer> layers = {Layer(/* input_dim */ 784, /* output_dim */ 512, /* activation */ RELU, /* dropout_rate */ 0.2),
                            Layer(/* input_dim */ 512, /* output_dim */ 512, /* activation */ RELU, /* dropout_rate */ 0.2),
                            Layer(/* input_dim */ 512, /* output_dim */ 10, /* activation */ LINEAR, /* dropout_rate */ 0.0)};
    network.build_model(layers);

    // Use Adam optimizer
    AdamTrainer trainer(network.model);
    trainer.clip_threshold *= BATCH_SIZE;

    // Create model

    // Initialize variables for training -------------------------------------------------------------
    // Number of batches in training set
    unsigned num_batches = mnist_train.size() / BATCH_SIZE - 1;

    // Random indexing
    unsigned si;
    vector<unsigned> order(num_batches);
    std::iota(order.begin(), order.end(), 0);

    // Run for the given number of epochs
    for (unsigned epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        // Reshuffle the dataset
        cerr << "**SHUFFLE\n";
        random_shuffle(order.begin(), order.end());
        // Initialize loss and number of samples processed (to average loss)
        double loss = 0;
        double num_samples = 0;

        // Activate dropout
        network.nn.enable_dropout();

        for (si = 0; si < num_batches; ++si)
        {
            // build graph for this instance
            ComputationGraph cg;
            // Compute batch start id and size
            int id = order[si] * BATCH_SIZE;
            unsigned bsize = std::min((unsigned)mnist_train.size() - id, BATCH_SIZE);
            // Get input batch

            vector<float> batch_x(784 * bsize);
            vector<unsigned> batch_y(bsize);

            for (unsigned idx = 0; idx < bsize; ++idx)
            {
                std::copy(mnist_train[id+idx].begin(), mnist_train[id+idx].end(), batch_x.begin() + idx * 784);
                batch_y[idx] =  mnist_train_labels[id+idx];
            }
            Expression batch_x_expr = input(cg, Dim({784}, bsize), batch_x);
            // Get negative log likelihood on batch
            Expression loss_expr = network.nn.get_nll(batch_x_expr, batch_y, cg);
            // Get scalar error for monitoring
            loss += as_scalar(cg.forward(loss_expr));
            // Increment number of samples processed
            num_samples += bsize;
            // Compute gradient with backward pass
            cg.backward(loss_expr);
            // Update parameters
            trainer.update();
            // Print progress every tenth of the dataset
            if ((si + 1) % (num_batches / 10) == 0 || si == num_batches - 1)
            {
                // Print informations
                trainer.status();
                cerr << " E = " << (loss / num_samples) << " \n";
                // Reinitialize loss
                loss = 0;
                num_samples = 0;
            }
        }

        // Disable dropout for dev testing
        network.nn.disable_dropout();

        // Show score on dev data
        if (si == num_batches)
        {
            double dpos = 0;
            for (unsigned i = 0; i < mnist_dev.size(); ++i)
            {
                // build graph for this instance
                ComputationGraph cg;
                // Get input expression
                Expression x = input(cg, {784}, mnist_dev[i]);
                // Get negative log likelihood on batch
                unsigned predicted_idx = network.nn.predict(x, cg);
                // Increment count of positive classification
                if (predicted_idx == mnist_dev_labels[i])
                    dpos++;
            }
            // Print informations
            cerr << "\n***DEV [epoch=" << (epoch)
                 << "] E = " << (dpos / (double)mnist_dev.size()) << ' ';
        }
    }
}
