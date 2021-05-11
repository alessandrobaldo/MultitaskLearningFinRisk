import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from collections import OrderedDict

class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()

class Identity(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()
	def forward(self, x):
		return x

class ResBlock(nn.Module):
	def __init__(self, i, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
		super(ResBlock, self).__init__()
		self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
										   stride=stride, padding=padding, dilation=dilation))
		self.chomp1 = Chomp1d(padding)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
										   stride=stride, padding=padding, dilation=dilation))
		self.chomp2 = Chomp1d(padding)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)

		self.net = nn.Sequential()
		self.net.add_module("conv1_block_{}".format(i+1),self.conv1)
		self.net.add_module("chomp1_block_{}".format(i+1),self.chomp1)
		self.net.add_module("relu1_block_{}".format(i+1),self.relu1)
		self.net.add_module("drop1_block_{}".format(i+1),self.dropout1)

		self.net.add_module("conv2_block_{}".format(i+1),self.conv2)
		self.net.add_module("chomp2_block_{}".format(i+1),self.chomp2)
		self.net.add_module("relu2_block_{}".format(i+1),self.relu2)
		self.net.add_module("drop2_block_{}".format(i+1),self.dropout2)
		
		self.skip = nn.Sequential()
		
		if n_inputs != n_outputs:
			self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
			self.skip.add_module("downsample_block_{}".format(i+1), self.downsample)
		else:
			self.downsample = None
			self.identity = Identity()
			self.skip.add_module("identity_block_{}".format(i+1), self.identity)
		
		self.relu = nn.ReLU()
		
		self.block = nn.Sequential(OrderedDict([
			('block_{}'.format(i+1), self.net),
			('skip_conn_{}'.format(i+1), self.skip),
			('final_relu_{}'.format(i+1), self.relu)
		]))
		self.init_weights()

	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.01)
		self.conv2.weight.data.normal_(0, 0.01)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.net(x)
		res = self.skip(x)
		return self.relu(out + res)


class ResNet(nn.Module):
	def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
		super(ResNet, self).__init__()
		layers = []
		num_levels = len(num_channels)
		for i in range(num_levels):
			dilation_size = 2 ** i
			in_channels = num_inputs if i == 0 else num_channels[i-1]
			out_channels = num_channels[i]
			layers += [ResBlock(i, in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
									 padding=(kernel_size-1) * dilation_size, dropout=dropout)]

		self.network = nn.Sequential()
		for i,l in enumerate(layers):
			self.network.add_module("resblock_{}".format(i+1), l)
		
		f = nn.Flatten()
		self.network.add_module("flatten",f)

	def forward(self, x):
		return self.network(x)

class ResForkNet(nn.Module):
	def __init__(self, n_dependents, num_inputs, timesteps, num_channels, ll_sizes = [100, 64, 16], kernel_size=3, dropout=0.2):
		super(ResForkNet, self).__init__()

		self.fork_net = nn.Sequential()
		self.subnets = []
		self.timesteps = timesteps
		self.n_dependents = n_dependents
		
		for n in range(n_dependents):
			r = ResNet(num_inputs, num_channels, kernel_size=3, dropout=0.2)
			self.subnets += [r]
			self.fork_net.add_module("resnet_{}".format(n+1), r)

		self.feature_dim = self.getFeatureDim(r, num_inputs, timesteps)

		l1 = nn.Linear(self.feature_dim*self.n_dependents,ll_sizes[0])
		l2 = nn.Linear(ll_sizes[0],ll_sizes[1])
		l3 = nn.Linear(ll_sizes[1],ll_sizes[2])
		l4 = nn.Linear(ll_sizes[2],self.n_dependents)
		self.linears = [l1,l2,l3,l4]
		self.fork_net.add_module("linear1", l1)
		self.fork_net.add_module("linear2", l2)
		self.fork_net.add_module("linear3", l3)
		self.fork_net.add_module("linear4", l4)		

	def getFeatureDim(self, r, num_inputs, timesteps):
		bs = 1
		inp = Variable(torch.rand(bs, num_inputs, timesteps))
		output_feat = r(inp)
		n_size = output_feat.data.view(bs, -1).size(1)
		return n_size

	def forward(self,x):
		xs = [r(x[:,:,i*self.timesteps:(i+1)*self.timesteps]) for i,r in enumerate(self.subnets)]
		y = torch.cat(xs, axis = 1)
		for layer in self.linears:
			y = layer(y)
				
		return y
	
	
def negative_log_prior(params):
	regularization_term = 0
	for name, W in params:
		regularization_term += W.norm(2)
	return 0.5*regularization_term

def train(model, train_loader, test_loader, loss_module, optimizer, epochs):
	print("-"*15," START TRAINING ", "-"*15)
	losses = {"train":[], "val":[]}
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	model = model.to(device)
	model.train()
	
	for i in range(1, epochs+1):
		train_loss = 0
		for j,(batch,labels) in enumerate(train_loader,0):
			batch, labels,_ = batch.to(device), labels.to(device)
			preds = model(batch)
			loss = loss_module(preds, labels) + negative_log_prior(model.named_parameters())
			train_loss += loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			with torch.no_grad():
				val = iter(test_loader)
				val_X, val_Y,_ = next(val)
				val_X, val_Y = val_X.to(device), val_Y.to(device)
				val_preds = torch.squeeze(model(val_X))
				val_loss = loss_module(val_preds, val_Y) + negative_log_prior(model.named_parameters())
		
			print('',end='\r')
			print("Epochs:[{}/{}] {}>{} train_loss: {}, val_loss: {}".format(i,epochs,"-"*j,"-"*(len(train_loader)-j-1),train_loss/(j+1), val_loss), end='')
		losses["train"].append(train_loss.item()/(j+1))
		losses["val"].append(val_loss.item())
	return losses