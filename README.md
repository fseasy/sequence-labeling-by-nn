#Sequence Labeling by Neural Network

The repository is for sequence labeling through neural negwork (deep learning) methods .

# Build

## Under MSVC

1. open git bash
4. git checkout msvc
5. `mkdir build`
6. `cd build`
7. `cmake .. -DEIGEN3_INCLUDE_DIR=/eigen/path -DBOOST_ROOT=/boost/path -DBoost_USE_STATIC_LIBS=On`
8. open the vs solution under build folder



it is now based on [CNN library](https://github.com/clab/cnn)

steps :

1. postagging based on example `tag-bilstm.cc` of CNN [doing]

2. chinese segmentation(using sequence labeling method) , ner , srl

3. more various structures based on CNN

4. (almost)from scratch ?


