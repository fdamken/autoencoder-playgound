#!/usr/bin/env bash

for bottleneck in 2 3 5 10 20 200; do
    python mnist-autoencoder.py $bottleneck --cuda
done

python mnist-conv-autoencoder.py --cuda

for bottleneck in 2 3 5 10 20 200; do
    python mnist-variational-autoencoder.py $bottleneck --cuda
done
