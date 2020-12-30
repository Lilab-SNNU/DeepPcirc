import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NetCNN(nn.Module):
    def __init__(self):
        super(NetCNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 128, 12)
        self.conv1.add_module("batchnorm",nn.BatchNorm1d(128))
        self.conv1.add_module("relu", nn.ReLU())

        self.conv2 = nn.Conv1d(128,64,8)
        self.conv2.add_module("batchnorm2", nn.BatchNorm1d(64))
        self.conv2.add_module("relu2", nn.ReLU())
        self.conv3 = nn.Conv1d(64,32,6)
        self.conv3.add_module("batchnorm3", nn.BatchNorm1d(32))
        self.conv3.add_module("relu3", nn.ReLU())
        self.pool = nn.MaxPool1d(4,4)
        self.dp = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = F.relu(x)
        x = self.pool(x)
        x = self.dp(x)
        return x

class NetRNN(nn.Module):
    def __init__(self, wordvec_len ,HIDDEN_NUM, LAYER_NUM, DROPOUT, cell):
        super(NetRNN,self).__init__()
        self.rnn = torch.nn.Sequential()

        if cell == 'LSTM':
            self.rnn.add_module("lstm", nn.LSTM(input_size=wordvec_len, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
                               bidirectional=True, dropout=DROPOUT))
        else:
            self.rnn.add_module("gru", nn.GRU(input_size=wordvec_len, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
                                                    bidirectional=True, dropout=DROPOUT))


    def forward(self, x):

        x = x.permute(2,0,1)
        out, _ = self.rnn(x) 
        out = torch.mean(out, 0)
        return out

class ronghe_model(nn.Module):
    def __init__(self, wordvec_len, HIDDEN_NUM, LAYER_NUM, DROPOUT, cell):
        super(ronghe_model, self).__init__()
        self.wordvec_len = wordvec_len
        self.HIDDEN_NUM = HIDDEN_NUM
        self.LAYER_NUM = LAYER_NUM
        self.DROPOUT = DROPOUT
        self.cell = cell
        self.cnn = NetCNN()
        self.rnn = NetRNN(wordvec_len ,HIDDEN_NUM, LAYER_NUM, DROPOUT, cell)
        self.fc = nn.Linear(HIDDEN_NUM*2,10)
        self.fc2 = nn.Linear(10, 2)
        self.dp = nn.Dropout(0.5)


    def forward(self, x):

        x = self.cnn(x)
        x = self.rnn(x)
        x = self.fc(x)
        x = self.dp(F.elu(x))
        x = self.fc2(x)
        return x

def train(model, loss, optimizer, x_val, y_val):

    x = x_val.to(device)
    y = y_val.to(device)
    model.train()
    optimizer.zero_grad()
    fx = model(x)
    output = loss(fx, y)
    pred_prob = F.log_softmax(fx, dim=1)
    output.backward()
    optimizer.step()

    return output.item(), pred_prob, list(np.array(y_val)), list(fx.data.cpu().detach().numpy().argmax(axis=1))

def predict(model, x_val):
    model.eval()

    output = model.forward(x_val)
    return output

def save_checkpoint(state,is_best,model_path):
    if is_best:
        print('=> Saving a new best from epoch %d"' % state['epoch'])
        torch.save(state, model_path + '/' + 'checkpointbi_NCP_ANF.pt')

    else:
        print("=> Validation Performance did not improve")


def ytest_ypred_to_file(y_test, y_pred, out_fn):
    with open(out_fn,'w') as f:
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred[i])+'\n')
