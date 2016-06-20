# Sequence Labeling by Neural Network

The repository is for sequence labeling through neural network (deep learning) methods .

## RUN FLOW SAMPLE

directory [run\_flow\_example](run_flow_example) contains naive example to bulid project and run samples under linux .

## Build

### Under MSVC

1. open git bash
4. git checkout msvc
5. `mkdir build`
6. `cd build`
7. `cmake .. -DEIGEN3_INCLUDE_DIR=/eigen/path -DBOOST_ROOT=/boost/path -DBoost_USE_STATIC_LIBS=On`
8. open the VS solution under build folder

### Under Linux

1. get eigen3

2. `cmake .. -DEIGEN3_INCLUDE_DIR=/eigen/path` # no `-DBoost_USE_STATIC_LIBS=On` !

BOOST is assumed has already exists in global environment.

### CNN version

on Windows , use branch `cnn-msvc` 

on Linux , using branch `remotes/origin/bugfix/memalloc`.  Or `remotes/origin/master` and `7b8adbc` commit (but a memory-pool bug exists).

## Plan 


it is now based on [CNN library](https://github.com/clab/cnn)

steps :

1. postagging based on example `tag-bilstm.cc` of CNN [done]

2. chinese segmentation(using sequence labeling method) , ner [done]

3. more various structures based on CNN [doing]

4. (almost)from scratch ?? -> NO , need more time to think about it !
