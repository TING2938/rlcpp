
    ./train_mnist \
    --train data/train-images.idx3-ubyte \
    --train_labels data/train-labels.idx1-ubyte \
    --dev data/t10k-images.idx3-ubyte \
    --dev_labels data/t10k-labels.idx1-ubyte \
    --batch_size 64 \
    --num_epochs 20
