## TODO List
1. ~~Compression npy -> bl2~~
2. ~~Multiprocessing~~
3. ~~Write a program to load sessions and build attributes~~
4. Write testing code to confirm the results are the same
5. Write testing code to confirm that manually loading lfp files work, it should be the same as loading lfp files from the allensdk
6. Write a function to select probe for each session
7. Optimize the code to use one cpu thread per thread
6. Try combining training and attrs (?) -> but this means attrs will be running on GPUs and distributed GPU processing still needs to be tested for speed



## TODO list for training LSTM
1. add an eval set
2. add LR handler
3. add early stopping
4. and gaussian noise to training data