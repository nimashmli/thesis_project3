import torch
import torch.nn as nn 

class model(nn.Module) : 
    def __init__(self,layer_neuron ):
        super().__init__()
        a = layer_neuron[0]
        self.layers = nn.ModuleList()
        for b in layer_neuron[1:] : 
            self.layers.append(nn.Linear(a , b))
            a = b 
        self.relu = nn.ReLU()
    def forward(self , x ) : 
        x = x.view(x.size(0), -1) 
        for layer in self.layers[:-1] : 
            x = layer(x)
            x = self.relu(x)
        x = self.layers[-1](x)
        return x



