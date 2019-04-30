import torch 

from torch.nn.parameter import Parameter 
from torch.nn.modules.module import Module 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import math
import time 
from typing import List

def accuracy(output: 'torch.Tensor', labels: 'torch.Tensor') -> float:
    """Calculates accuracy given output and correct labels

    Args:
        output: output of model
        labels: correct labels
    
    Returns:
        accuracy of model
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    return correct / len(labels)

class GCNLayer(Module):
    """Single covolutional layer in a GCN"""
    def __init__(self, 
                 in_features: 'torch.Tensor', 
                 out_features: 'torch.Tensor', 
                 bias: bool = True) -> None:
        """Constructor

        Args:
            in_features: layer input features
            out_features: layer output features
            bias: if True, adds bias to layer parameters. Default is True.
        """
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize Parameters
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, 
                input_features: 'torch.Tensor', 
                adj: 'torch.Tensor') -> 'torch.Tensor':
        """Defines forward pass through layer

        Note: PyTorch automatically defines backward pass based \
            on this forward pass.

        Args:
            input_features: input to layer
            adj: Adjacency matrix 

        Returns:
            Output features for layer
        """
        support = torch.mm(input_features, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCNModel(Module):
    def __init__(self, 
                 nfeat: int,
                 hidden: List[int], 
                 nclass: int, 
                 dropout: float) -> None:
        """Constructor

        Args:
            nfeat: Number of input features
            hidden: Number of features for each hidden layer
            nclass: Number of classes
            dropout: Dropout rate
        """
        super(GCNModel, self).__init__()

        layer_sizes = [nfeat] + hidden + [nclass]
        self.nlayers = len(layer_sizes) - 1
        for i in range(len(layer_sizes) - 1):
            self.add_module(str(i), GCNLayer(layer_sizes[i], layer_sizes[i+1]))
        self.dropout = dropout

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def forward(self, 
                x: 'torch.Tensor', 
                adj: 'torch.Tensor') -> 'torch.Tensor':
        """Defines forward pass through model

        Note: PyTorch automatically defines backward pass based \
            on this forward pass.

        Args:
            x: model input features
            adj: Adjacency matrix 

        Returns:
            Predicted labels for nodes in graph
        """
        for i in range(self.nlayers):
            layer = self[i]
            """
            if i == 0:
                x = F.relu(layer(x, adj))
                x = F.dropout(x, self.dropout, training=self.training)
            elif i < self.nlayers - 1:
                x = F.relu(layer(x, adj))
            else:
                x = layer(x, adj)
                x = F.dropout(x, self.dropout, training=self.training)
            """
            if i < self.nlayers - 1:
                x = F.relu(layer(x, adj))
                x = F.dropout(x, self.dropout, training=self.training)  

            else: # No ReLU/Dropout for last layer
                x = layer(x, adj)
            
            
        return F.log_softmax(x, dim=1)

class GCN:
    def __init__(self, 
                 adj: 'torch.Tensor',
                 features: 'torch.Tensor',
                 labels: 'torch.Tensor',
                 index_train: 'torch.Tensor',
                 index_val: 'torch.Tensor',
                 index_test: 'torch.Tensor') -> None:
        """Constructor

        Args:
            adj: Adjacency matrix
            features: Input features
            labels: Labels
            index_train: Indeces for training nodes in adjacency matrix
            index_val: Indeces for validation nodes in adjacency matrix
            index_test: Indeces for testing nodes in adjacency matrix
        """
        self.adj = adj
        self.index_train = index_train
        self.index_val = index_val
        self.index_test = index_test
        self.features = features 
        self.labels = labels 

    def train(self, 
              hidden: List[int],
              dropout: float,
              learning_rate: float, 
              weight_decay: float,
              epochs: int, 
              verbose: bool = False) -> None:
        """Constructor

        Args:
            hidden: Number of features for each hidden layer.
            dropout: Dropout rate
            learning_rate: Learning Rate 
            weight_decay: Weight decay
            epochs: Number of training epochs
            verbose: Prints accuracy/timing messages if True. \ 
                Defaults to False.
        """
        if epochs <= 0:
            raise ValueError(f'epochs cannot be less than 1.')

        self.model = GCNModel(nfeat=self.features.shape[1],
                    hidden=hidden,
                    nclass=self.labels.max().item() + 1,
                    dropout=dropout)
        self.optimizer = Adam(self.model.parameters(), 
                              lr=learning_rate, 
                              weight_decay=weight_decay)

        labels_train = self.labels[self.index_train]
        labels_val = self.labels[self.index_val]
        labels_test = self.labels[self.index_test]

        if verbose:
            print('Beginning Training')

        output = None
        self.model.train()      # set model in training mode
        begin = time.time()
        for epoch in range(epochs):
            self.optimizer.zero_grad()  # clear gradients
        
            output = self.model(self.features, self.adj)
            output_train = output[self.index_train]
            loss_train = F.nll_loss(output_train, labels_train)
            acc_train = accuracy(output_train, labels_train)
            loss_train.backward()
            self.optimizer.step()

            output_val = output[self.index_val]
            loss_val = F.nll_loss(output_val, labels_val)
            acc_val = accuracy(output_val, labels_val)

            if verbose:
                print(f'Epoch {epoch} \n' +
                    f'\tTraining Loss: {loss_train.item():.3f}\n' +
                    f'\tTraining Accuracy: {acc_train.item()*100:.2f}\n' +
                    f'\tValidation Loss: {loss_val.item():.3f}\n' +
                    f'\tValidation Accuracy: {acc_val.item()*100:.2f}')
        
        if verbose:
            print(f'Training Time: {time.time() - begin:.3f}')
            print('Beginning Testing')

        begin = time.time()
        self.model.eval() # set model in evaluation mode
        output_test = self.model(self.features, self.adj)[self.index_test]
        loss_test = F.nll_loss(output_test, labels_test)
        acc_test = accuracy(output_test, labels_test)

        if verbose:
            print('------------------------------------')
            print(f'Testing Time: {time.time() - begin}')
            print(f'Testing Loss: {loss_test}')
            print(f'Testing Accuracy: {acc_test*100:.2f}%')
            print('------------------------------------')
