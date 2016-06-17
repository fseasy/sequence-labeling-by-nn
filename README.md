# Sequence Labeling by Neural Network

The repository is for sequence labeling through neural network (deep learning) methods .

## Build

### Under MSVC

1. open git bash
4. git checkout msvc
5. `mkdir build`
6. `cd build`
7. `cmake .. -DEIGEN3_INCLUDE_DIR=/eigen/path -DBOOST_ROOT=/boost/path -DBoost_USE_STATIC_LIBS=On`
8. open the vs solution under build folder




### Under Linux

1. get eigen3

2. `cmake .. -DEIGEN3_INCLUDE_DIR=/eigen/path -DBoost_USE_STATIC_LIBS=On`(assume boost has already exists in global environment)

## Plan 


it is now based on [CNN library](https://github.com/clab/cnn)

steps :

1. postagging based on example `tag-bilstm.cc` of CNN [done]

2. chinese segmentation(using sequence labeling method) , ner [done]

3. more various structures based on CNN [doing]

4. (almost)from scratch ?? -> NO , need more time to think about it !
