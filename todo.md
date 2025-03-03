## TODO List
1. ~~Compression npy -> bl2~~
2. ~~Multiprocessing~~
3. ~~Write a program to load sessions and build attributes~~
4. ~~Write testing code to confirm the results are the same~~
5. ~~Write testing code to confirm that manually loading lfp files work, it should be the same as loading lfp files from the allensdk~~
6. ~~Write a function to select probe for each session~~
7. ~~Optimize the code to use one cpu thread per thread~~
8. ~~Try combining training and attrs (?) -> but this means attrs will be running on GPUs and distributed GPU ~~
1. ~~check Xtrain test etc~~
2. ~~check attribution score calc results (with the same attributions)~~

## TODO list for training LSTM
1. ~~add an val set~~
2. ~~add LR scheduler~~
3. ~~add early stopping~~
4. and gaussian noise to training data

## TODO
1. ~~keep trying with training~~
2. ~~write a script to generate test sets using passive trials~~
3. ~~save those trials~~
4. ~~test on the current model and save the results~~
5. ~~get the code back and test on the original model~~
6. ~~save the code and the results~~
7. ~~compare the results~~
8. seqlen 20-2000 and compare how they differ in channel 5, broadband
9. use the original code to try a new session