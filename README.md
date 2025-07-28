# spikes2LFP project

this project explores the relationship between spiking activity across visual and non-visual areas and multi-layer LFP in the primary visual cortex (VISp/V1) of the mouse brain.

## Data
The data used for this project is from the Allen Institute Brain Observatory, although with light modifications the code can work on any similar dataset.

## Models
The code currently supports the following models to predict LFP from spiking activity:
+ Linear regression model
+ GRU model
+ LSTM model
+ Transformer model

* All models include an optional CNN layer

## Model attribution
We use integrated gradients to estimate the contribution of each feature (neuron) to the prediction

## Usage
* To follow a step-by-step application of the pipeline, use start_notebook.ipynb
* To run the model and save the results use start.py

## dependencies
+ allensdk
+ pytorch
+ sklearn
+ pandas
+ scipy
+ matplotlib 
+ seaborn
