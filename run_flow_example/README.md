# runing flow example

```shell
sh buildflow.sh # build program 
sh runflow.sh # run sample , only postagger input1_feature2output model is demonstrated
```

To ensure build flow work successfully, **you should promise your machine has `boost` library** , and `1.57`,`1.58` has been validated , and 1.60 will not work because of some libiraries has been became hpp instead of lib . 

## currently model

The code has be restructured about 2 times and there also exists so many duplicated code for different model and task .

For now , almost every task has the same model structure , including 2 input types and 3 output types . there are described as following :

1. input : single input(input1 , input2d) , double channel 

    single input means only use training data tokens as input , for CWS and POSTAG , there only word(character for CWS) , so alse may be named input1 . But for NER , both word and postag are the input , so also called input2d ;

    double channel means not only training data tokens , but also word embeddings from unlabelled data . Word embedding is trained using `Word2vec` , using `skip-gram` choice .

2. output : classification , pretag , crf
    
    classification means only use current input sequence infomation to predict tag , 
    pretag means add "previous tag infomation" ,
    crf means do viterbi decoding .

So it may be have at least 6 models (some task may havn't classification output model ).

What's more , for POSTAG input1+classification ,  we have added handcraft feature , and RNN , GRU to replace LSTM , MLP and so on . the model becomes more and more . it becomes difficult to demonstrate all models .

And for compile speed , we has comment some model building , if you need build it , just uncomment the `add_subdirectory(XXX)` at CMakeLists.txt under every task root directory , such as `postagger/CMakeLists.txt`
