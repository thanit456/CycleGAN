import torch 
from torch import nn
import torch.nn.functional as F

class DCDiscriminator(nn.Module):
	def __init__(self, conv_dim=64):
		super(DCGAN, self).__init__()
		self.conv1 = nn.Conv2D(3, conv_dim//2, kernel_size=(4,4), stride=2, padding=(2,2))
		self.bn1 = nn.BatchNorm2D(conv_dim//2)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2D(conv_dim//2, conv_dim, kernel_size=(4,4), stride=2, padding=(2,2))
		self.bn2 = nn.BatchNorm2D(conv_dim)
		self.conv3 = nn.Conv2D(conv_dim, conv_dim*2, kernel_size=(4,4), stride=2, padding=(2,2))
		self.bn3 = nn.BatchNorm2D(conv_dim*2)
		self.conv4 = nn.Conv2D(conv_dim*2, 1, kernel_size=(4, 4), stride=2, padding=(0, 0))
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x) 
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x) 
		x = self.relu(x)

		x = self.conv3(x)
		x = self.bn3(x) 
		x = self.relu(x)

		x = self.conv4(x)
		x = F.sigmoid(x)
		return x 

class DCGenerator(nn.Module):
	def __init__(self, noise_size, conv_dim):
		self.deconv1 = nn.ConvTranspose2d(100, conv_dim*2, kernel_size=(4,4), stride=2, padding=(0, 0))
		self.bn1 = nn.BatchNorm2d(conv_dim*2) 
		self.deconv2 = nn.ConvTranspose2d(conv_dim*2, conv_dim, kernel_size=(4,4), stride=2, padding=(2,2))
		self.bn2 = nn.BatchNorm2d(conv_dim)
		self.deconv3 = nn.ConvTranspose2d(conv_dim, conv_dim//2, kernel_size=(4,4), stride=2, padding=(2,2))
		self.bn3 = nn.BatchNorm2d(conv_dim//2)
		self.deconv4 = nn.ConvTranspose2d(conv_dim, 3, kernel_size=(4,4), stride=2, padding=(2,2))
		self.relu = nn.ReLU()
	def forward(self, x):
		x = self.deconv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.deconv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		x = self.deconv3(x)
		x = self.bn3(x)
		x = self.relu(x)
	
		x = self.deconv4(x)
		x = F.tanh(x)
		return x

if __name__ == '__main__':
	x = torch.rand(64, 64, 3)
	discriminator = DCDiscriminator()
	print(discrimator(x).shape)

	generator = DCGenerator()
	print(generator(x).shape)
