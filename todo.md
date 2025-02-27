## TODO List
1. ~~Compression npy -> bl2~~
2. ~~Multiprocessing~~
3. ~~Write a program to load sessions and build attributes~~
4. Write testing code to confirm the results are the same
5. Write testing code to confirm that manually loading lfp files work, it should be the same as loading lfp files from the allensdk
6. ~~Write a function to select probe for each session~~
7. Optimize the code to use one cpu thread per thread
8. Try combining training and attrs (?) -> but this means attrs will be running on GPUs and distributed GPU processing still needs to be tested for speed



## TODO list for training LSTM
1. add an eval set
2. add LR scheduler
3. add early stopping
4. and gaussian noise to training data


## Sanity checks to do
1. check Xtrain test etc
2. check attribution score calc results (with the same attributions)

## Side quest
1. Modify the function the add validation sets


## TODO
1. seqlen 20-2000
2. 