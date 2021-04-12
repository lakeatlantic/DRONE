import torch.nn as nn
import torch.nn.functional as F
import torch as T
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn.init import xavier_uniform_


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

    # def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, constrain_out=False, norm_in=True, discrete_action=True):
    #     super(MLPNetwork, self).__init__()
    #     self.hidden = nn.Linear(input_dim, hidden_dim)   # hidden layer
    #     self.predict = nn.Linear(hidden_dim, out_dim)   # output layer

    # def forward(self, x):
    #     x = F.relu(self.hidden(x))      # activation function for hidden layer
    #     x = self.predict(x)             # linear output
    #     return x

    # def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, constrain_out=False, norm_in=True, discrete_action=True):
    #     super(MLPNetwork, self).__init__()
    #     # input to first hidden layer
    #     self.hidden1 = Linear(input_dim, 16)
    #     xavier_uniform_(self.hidden1.weight)
    #     self.act1 = Sigmoid()
    #     # second hidden layer
    #     self.hidden2 = Linear(16, 8)
    #     xavier_uniform_(self.hidden2.weight)
    #     self.act2 = Sigmoid()
    #     # third hidden layer and output
    #     self.hidden3 = Linear(8, 1)
    #     xavier_uniform_(self.hidden3.weight)
 
    # # forward propagate input
    # def forward(self, X):
    #     # input to first hidden layer
    #     X = self.hidden1(X)
    #     X = self.act1(X)
    #      # second hidden layer
    #     X = self.hidden2(X)
    #     X = self.act2(X)
    #     # third hidden layer and output
    #     X = self.hidden3(X)
    #     return X