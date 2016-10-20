#!/bin/sh

# set wd
cd ../

# init cnn submodule
echo "init cnn submodule" >/dev/stderr
git submodule init
git submodule update

# build
mkdir -p build
cd build
# get eigen
echo "clone eigen3" >>/dev/stderr
git clone https://github.com/RLovelett/eigen.git eigen

# cmake
cmake -DEIGEN3_INCLUDE_DIR=build/eigen/ .. 

#make
make -j2
