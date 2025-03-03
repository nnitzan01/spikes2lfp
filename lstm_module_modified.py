import sys 
import numpy as np 
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

class LSTMnet(nn.Module):
    def __init__(self, input_size, num_hidden, num_layers, print=False):
        super().__init__()
        self.print = print
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, num_hidden, num_layers, batch_first=True)
        self.ln = nn.LayerNorm(num_hidden)
        self.out = nn.Linear(num_hidden, 1)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = self.ln(out)
        out = self.out(out)
        return out
    
    
class process_model:
    def __init__(self, model, seqlength, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.seqlength = seqlength
        self.trained = False
            
    
    def train(self, x_train, y_train, x_val, y_val, numepochs, batch_size, lr=1e-4, weight_decay=5e-4):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numepochs)
        losses_train = np.zeros(numepochs)
        losses_val = np.zeros(numepochs)
        batch_size = batch_size
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        patience = 5
        patience_count = 0
        loss_train_best = np.inf
        for epochi in range(numepochs):
            self.model.train()
            losses_train_epoch = []
            for (x_t, y_t) in train_loader:
                yHat = self.model(x_t.to(self.device))
                loss = self.criterion(torch.squeeze(y_t.to(self.device)), torch.squeeze(yHat))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses_train_epoch.append(loss.item())
            losses_train[epochi] = np.mean(losses_train_epoch)

            self.model.eval()
            losses_val_list = []
            with torch.no_grad():
                for (x_v, y_v) in val_loader:
                    yHat = self.model(x_v.to(self.device))
                    loss = self.criterion(torch.squeeze(y_v.to(self.device)), torch.squeeze(yHat))
                    losses_val_list.append(loss.item())
            losses_val[epochi] = np.mean(losses_val_list)
            if losses_val[epochi] < loss_train_best:
                patience_count = 0
                loss_train_best = losses_val[epochi]
            else:
                patience_count += 1
                if patience_count > patience:
                    break

        self.trained = True
        return losses_train, losses_val

    def evaluate(self, X):
        self.model.eval()
        input_size = X.shape[2]
        y_hat = np.zeros([X.shape[0], X.shape[1]])
        with torch.no_grad():
            for i in range(X.shape[0]):
                x = X[i, :, :].view(1, self.seqlength, input_size).to(self.device)
                yy = self.model(x)
                y_hat[i] = np.squeeze(yy.cpu().numpy())
        return y_hat
    
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        # print a message if model wasn't trained
        if not hasattr(self, 'trained'):
            sys.stdout.write('Model was not yet trained.\n')
        return self.model