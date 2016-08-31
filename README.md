# Sequence Labeling by Neural Network

The repository is for sequence labeling through neural network (deep learning) methods .

## RUN FLOW SAMPLE

directory [run\_flow\_example](run_flow_example) contains naive example to bulid project and run samples under linux .

## Build

### dependency

we are using [CNN library](https://github.com/clab/cnn) fork [CNN-Oneplus](https://github.com/oneplus/cnn) as the basic neural framework. After clone the repository, we should use 

```shell
git submodule init
git submodule update
```

to clone down the `cnn` module.

cnn needs `boost` and `eigen3`. `cmake` is also needed.

### Under MSVC

**boost-1.57.0, boost-1.58.0** are supported, and **boost-1.60.0** leads to some compiling errors.

1. get [eigen3](https://bitbucket.org/eigen/eigen/)
2. open `git bash` or `cmd`, change directory to the repository root
3. change directory to the *cnn*, and change the branch : `git checkout cnn-msvc` , return the root directory.
5. make a directory to build, `mkdir build`
6. `cd build`
7. using the command to make : `cmake .. -DEIGEN3_INCLUDE_DIR=/eigen/path -DBOOST_ROOT=/boost/path -DBoost_USE_STATIC_LIBS=On` , **Boost_USE_STATIC_LIBS=On** is needed for Windows.
8. open the VS solution under *build* folder

### Under Linux

you can just use `run_flow_example`

### CNN version

on Windows, use branch `cnn-msvc` 

on Linux, using branch `remotes/origin/bugfix/memalloc`.  Or `remotes/origin/master` and `7b8adbc` commit (but a memory-pool bug exists).

## Plan 


it is now based on [CNN library](https://github.com/clab/cnn)

steps :

1. postagging based on example `tag-bilstm.cc` of CNN [done]

2. chinese segmentation(using sequence labeling method) , ner [done]

3. more various structures based on CNN [doing]

4. (almost)from scratch ?? -> NO , need more time to think about it !

## WIKI

[wiki](https://github.com/memeda/sequence-labeling-by-nn/wiki) pages for more detail infomation.