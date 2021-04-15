import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class ConvBlock(nn.Module):
    def __init__(self,in_channels, f, kernel, dilation):
        super(ConvBlock, self).__init__()
        
        self.in_channels = in_channels
        self.f = f
        
        self.kernel = kernel
        self.pool = 2
        self.dilation = 1
        
        self.conv_stride = 1
        self.pool_stride = 2
        self.conv_padding = int((self.kernel-1)/2)
        self.pool_padding = 0

        self.conv = nn.Conv1d(in_channels = in_channels, out_channels = self.f, kernel_size = self.kernel, dilation = self.dilation)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.f)
        self.maxpool = nn.MaxPool1d(kernel_size = self.pool, stride = self.pool_stride)

    def forward(self,x):
        #print("Input ", x.size())
        x = F.pad(x, (self.conv_padding, self.conv_padding))
        x = self.conv(x)
        #print("After conv ",x.size())
        x = self.act(x)
        x = self.bn(x)
        x = self.maxpool(x)
        #print("After pool ",x.size())
        return x
    
    def getConvDim(self, dim_in):
        return int(((dim_in + 2*self.conv_padding - self.dilation*(self.kernel-1)-1)/self.conv_stride) +1)
    def getPoolDim(self, dim_in):
        return int(((dim_in + 2*self.pool_padding - self.dilation*(self.pool-1)-1)/self.pool_stride) +1)

class ConvNet(nn.Module):
    def __init__(self, filters, dim_in, kernel = 3, dilation = True):
        super(ConvNet, self).__init__()
        self.filters = filters
        self.dim_in = dim_in
        self.kernel = kernel

        self.model = nn.Sequential()
        dim_out = dim_in
        in_channels = 1
        for i,f in enumerate(self.filters):
            c = ConvBlock(in_channels, f, self.kernel, dilation = i+1)
            self.model.add_module("conv_block_{}".format(i+1), c)
            in_channels = f
            dim_out = c.getConvDim(dim_out)
            #print(dim_out)
            dim_out = c.getPoolDim(dim_out)
            #print(dim_out)

        self.model.add_module("flatten_1",nn.Flatten())
        self.model.add_module("linear_1", nn.Linear(f*dim_out, 16))
        self.model.add_module("relu_out1",nn.ReLU())
        self.model.add_module("batch_norm1d",nn.BatchNorm1d(16))
        self.model.add_module("dropout_1",nn.Dropout(0.2))
        self.model.add_module("linear_2",nn.Linear(16,4))
        self.model.add_module("relu_out2",nn.ReLU())
        self.model.add_module("linear_3",nn.Linear(4,1))


    def forward(self,x):
        return self.model(x)