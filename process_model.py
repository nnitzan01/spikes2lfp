import torch
import torch.nn as nn
import sys 
import numpy as np 

class process_model:
    def __init__(self, model, seqlength, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.seqlength = seqlength
        self.trained = False
            
    def train(self, x_train, y_train, numepochs, lr=.001, weight_decay=0.001):
        self.model.train()
        lossfun = self.criterion 
        N = x_train.shape[0] - np.mod(x_train.shape[0], self.seqlength)
        input_size = x_train.shape[1]
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        losses = np.zeros(numepochs)
        
        for epochi in range(numepochs):
            seglosses = []
            for timei in range(0, N , self.seqlength):
                # grab a snippet of data
                X = x_train[timei:timei+self.seqlength,:].view(self.seqlength,1,input_size).to(self.device)
                y = y_train[timei:timei+self.seqlength].view(self.seqlength,1).to(self.device)
                # forward pass and loss
                yHat = self.model(X)
                loss = lossfun(torch.squeeze(y), torch.squeeze(yHat))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                seglosses.append(loss.item())
                
            # average losses from this epoch
            losses[epochi] = np.mean(seglosses)
            msg = f'Finished epoch {epochi+1}/{numepochs}'
            sys.stdout.write('\r' + msg)
        # flag model as "trained"
        self.trained = True
        return losses

    def evaluate(self,x_data):
        self.model.eval()
        N = x_data.shape[0] - np.mod(x_data.shape[0], self.seqlength)
        input_size = x_data.shape[1]
        
        y_hat = np.zeros(x_data.shape[0])
        with torch.no_grad():
            for timei in range(0, N, self.seqlength):
                X = x_data[timei:timei+self.seqlength, :].view(self.seqlength, 1, input_size).to(self.device)
                yy = self.model(X)
                y_hat[timei:timei+self.seqlength] = np.squeeze(yy.cpu().numpy())
        return y_hat
    
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        # print a message if model wasn't trained
        if not hasattr(self, 'trained'):
            sys.stdout.write('Model was not yet trained.\n')
        return self.model