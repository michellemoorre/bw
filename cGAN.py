import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer

class CGAN(pl.LightningModule):

	def __init__(self, generator, discriminator, datamodule):
		super().__init__()
        self.save_hyperparameters()

		self.generator = generator
		self.discriminator = discriminator
		self.datamodule = datamodule
		self.trainloader = datamodule.train_dataloader()
		self.sample = lambda: next(iter(self.trainloader))
		self.val = next(iter(datamodule.val_dataloader()))
		self.loss = nn.BCEWithLogitsLoss()

	def forward(self, z):
		return torch.cat([z, self.generator(z)], dim = 1)

	def generator_step(self, x):
		z = x['L']
		self.generated_imgs = self(z)
		d_output = self.discriminator(self.generated_imgs)
		g_loss = self.loss(d_output, torch.ones(d_output.shape))
		g_loss_l1 = nn.L1Loss()(self.generated_imgs[:, 1:, ...], x['ab'])

		return g_loss + g_loss_l1

	def discriminator_step(self, x):
		x_ = torch.cat([x['L'], x['ab']], dim = 1)
		d_output_real = self.discriminator(x_)
		d_loss_real = self.loss(d_output_real, torch.ones(d_output_real.shape))

		d_output_fake = self.discriminator(self.generated_imgs)
		d_loss_fake = self.loss(d_output_fake, torch.zeros(d_output_fake.shape))
   
		return d_loss_real + d_loss_fake

	def training_step(self, batch, batch_idx, optimizer_idx):
		if optimizer_idx == 0:
			loss = self.generator_step(batch)

		if optimizer_idx == 1:
			loss = self.discriminator_step(batch)

		return loss

	def configure_optimizers(self):
		g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
		d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
		return [g_optimizer, d_optimizer], []

	def on_epoch_end(self):
		sample_imgs = self(self.val['L'])
		grid = torchvision.utils.make_grid(sample_imgs)
		self.logger.experiment.add_image("generated_images", grid, self.current_epoch)