#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=Part1_Task2/MNIST_LeCun_solver.prototxt $@
