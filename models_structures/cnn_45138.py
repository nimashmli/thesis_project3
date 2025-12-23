#this is a simulation of article : CNNandLSTM-BasedEmotionCharting Using
#                                  Physiological Signals
import torch
import torch.nn as nn 

class BatchNormalization(nn.Module):
    def __init__(self, num_channel):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_channel, requires_grad=True))
        self.beta  = nn.Parameter(torch.zeros(num_channel, requires_grad=True))

    def forward(self, x):
        # x: (batch, C, H, W)
        mean = x.mean(dim=(0,2,3), keepdim=True)
        var  = x.var(dim=(0,2,3), keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        return x_norm * self.alpha.view(1,-1,1,1) + self.beta.view(1,-1,1,1)
#########################################################################################
class layer(nn.Module): 
    def __init__(self, in_chanenl , out_channel , kernel , padding , stride ) -> None:
        super().__init__() 
        self.conv = nn.Conv2d(in_channels=in_chanenl, 
                              out_channels=out_channel, 
                              kernel_size=kernel , 
                              stride=stride , 
                              padding=padding
                              )
        self.batch_norm = BatchNormalization(out_channel) 
        self.maxpool = nn.MaxPool2d(kernel_size=2 , stride=2)
        self.relu = nn.ReLU()

    def forward(self , x) :
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return  x 
##########################################################################################3

class model(nn.Module): 
    def __init__(self , time_len , num_output): 
        super().__init__()
        self.layer1 = layer(1 , 4 , 3,1 , 1)
        self.layer2 = layer(4 , 8 , 3 , 1 , 1 )
        #self.layer3 = layer(16 , 32 , 3 , 1 , 1)
        self.dropout = nn.Dropout(p = 0.2)
        #out : (batch , 8 , 16 , 10)
        self.fully_connected  = nn.Linear(8*(time_len//4)*20 , num_output)
        self.mapping_idx = [4,13,19,21,29,31,37,39,47,49,55,57,67,76]
    def x_mapping(self , x )  :
        #this function maps the input x (batch , time_len , 14) into (batch , 1 , time_len , 81)
        B , T , C = x.shape
        mapped = torch.zeros(B , T , 81 , device=x.device)
        for ch in range(C) :
            mapped[ :  , : , self.mapping_idx[ch]] = x[:  , : , ch]
        return mapped.unsqueeze(1) #(batch , 1 , time_len  , 81 )
    def forward(self , x ) :
        x = self.x_mapping(x)
        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        batch = x.shape[0]
        x  = x.view(batch , -1)
        x = self.dropout(x)
        x = self.fully_connected(x)
        return x 
