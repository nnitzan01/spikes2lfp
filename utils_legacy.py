import torch
from sklearn.model_selection import train_test_split
def generate_training_data(lfp, spikes_obj, seqlength, test_size=0.2, make_val=False, val_size=0.5):
    num_trials = int(lfp.shape[0] / seqlength)
    X = spikes_obj.spkMat[:num_trials * seqlength, :]
    X = X.reshape(num_trials, seqlength, X.shape[1])
    y = lfp[:num_trials * seqlength, :,:]
    y = y.reshape(num_trials, seqlength, y.shape[1], y.shape[2])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train = torch.Tensor(X_train).float()
    y_train = torch.Tensor(y_train).float()

    X_train = torch.reshape(X_train, (X_train.shape[0]*X_train.shape[1], X_train.shape[2]))
    y_train = torch.reshape(y_train, (y_train.shape[0]*y_train.shape[1], y_train.shape[2], y_train.shape[3]))
    
    if make_val:
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size, random_state=42)
        X_test = torch.Tensor(X_test).float()
        y_test = torch.Tensor(y_test).float()    
        X_val = torch.Tensor(X_val).float()
        y_val = torch.Tensor(y_val).float()
        X_test = torch.reshape(X_test, (X_test.shape[0]*X_test.shape[1], X_test.shape[2]))
        y_test = torch.reshape(y_test, (y_test.shape[0]*y_test.shape[1], y_test.shape[2], y_test.shape[3]))
        X_val = torch.reshape(X_val, (X_val.shape[0]*X_val.shape[1], X_val.shape[2]))
        y_val = torch.reshape(y_val, (y_val.shape[0]*y_val.shape[1], y_val.shape[2], y_val.shape[3]))
        return X_train, y_train, X_test, y_test, X_val, y_val
    else:
        X_test = torch.Tensor(X_test).float()
        y_test = torch.Tensor(y_test).float()
        X_test = torch.reshape(X_test, (X_test.shape[0]*X_test.shape[1], X_test.shape[2]))
        y_test = torch.reshape(y_test, (y_test.shape[0]*y_test.shape[1], y_test.shape[2], y_test.shape[3]))
        return X_train, y_train, X_test, y_test