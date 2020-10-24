import torch 
from torch import nn
import torch.nn.functional as F

from .dc_gan import DCDiscriminator, DCGenerator

#training loop 
def training_loop(data_loader, noise_loader):
	discriminator = DCDiscriminator()
	generator = DCGenerator()
	discriminator_optimizer = torch.optim.Adam()
	generator_optimizer = torch.optim.Adam()

	# draw training examples 
	training_batch = data_loader.grab()
	
	# draw noise samples 
	noise_batch = noise_loader.grab()
	# generate fake images from noise samples
	fake_batch = generator(noise_batch)

	# WARNING : I'm unsure that the generator's weights have to be updated or not?
	# compute discriminator loss
	discriminator_loss = (torch.sum(torch.pow(discriminator(training_batch) - 1), 2)) / (len(training_batch)*2) + \ 
		torch.sum(torch.pow(discriminator(fake_batch), 2)) / (len(fake_batch)*2)

	discriminator_loss.backward()
	discriminator_optimizer.step()

	new_noise_batch = noise_loader.grab()
	
	# compute generator loss 
	generator_loss = torch.sum(torch.pow(discriminator(generator(new_noise_batch)), 2)) / len(new_noise_batch)
	
	generator_loss.backward()
	generator_optmizer.step()
