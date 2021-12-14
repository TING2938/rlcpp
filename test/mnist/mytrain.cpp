#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::layers;

int main()
{
  network<sequential> net;

  // add layers
  net << conv(32, 32, 5, 1, 6) << activation::tanh()  // in:32x32x1, 5x5conv, 6fmaps
      << ave_pool(28, 28, 6, 2) << activation::tanh() // in:28x28x6, 2x2pooling
      << fc(14 * 14 * 6, 120) << activation::tanh()   // in:14x14x6, out:120
      << fc(120, 10);                     // in:120,     out:10

  assert(net.in_data_size() == 32 * 32);
  assert(net.out_data_size() == 10);

  // load MNIST dataset
  std::vector<label_t> train_labels;
  std::vector<vec_t> train_images;

  parse_mnist_labels("train-labels.idx1-ubyte", &train_labels);
  parse_mnist_images("train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);

  // declare optimization algorithm
  adagrad optimizer;

  // train (50-epoch, 30-minibatch)
  net.train<mse, adagrad>(optimizer, train_images, train_labels, 30, 50);

  // save
  net.save("net");

  // load
  // network<sequential> net2;

  // net2.load("net");
}
