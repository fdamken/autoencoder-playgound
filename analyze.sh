#!/usr/bin/env bash

set -o errexit
set -o nounset

export PYTHONPATH=.

for bottleneck in 2 3 5 10 20 200; do
    echo "Analyzing AE with bottleneck $bottleneck."
    python src/mnist_analyze.py ae -s -b $bottleneck
done

echo "Analyzing CAE."
python src/mnist_analyze.py cae -s

for bottleneck in 2 3 5 10 20 200; do
    echo "Analyzing VAE with bottleneck $bottleneck."
    python src/mnist_analyze.py vae -s -b $bottleneck
done
