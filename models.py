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


class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
			)

	def forward(self, x):
		return self.double_conv(x)

class UNetDownBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.down = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
			)

	def forward(self, x):
		out = self.down(x)
		return out

class UNetUpBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
		self.conv = DoubleConv(in_channels, out_channels)
		
	def forward(self, x, concat_):
		x = self.up(x)
		diffY = concat_.size()[2] - x.size()[2]
		diffX = concat_.size()[3] - x.size()[3]
		x = F.pad(x, [diffX // 2, diffX - diffX // 2,
					diffY // 2, diffY - diffY // 2])
		out = self.conv(torch.cat([concat_, x], dim=1))
		return out

class UNet(nn.Module):
	def __init__(self, in_channels = 1, mid_channels = 64, out_channels = 2, n_blocks = 3):
		super().__init__()

		self.in_conv = DoubleConv(in_channels, mid_channels)
		self.encoder = nn.ModuleList()
		self.decoder = nn.ModuleList()

		for i in range(n_blocks):
			self.encoder.append(UNetDownBlock(mid_channels * 2 ** i, mid_channels * 2 ** (i+1)))
		for i in range(n_blocks):	
			self.decoder.append(UNetUpBlock(mid_channels * 2 ** (i+1), mid_channels * 2 ** i))
		self.decoder = self.decoder[::-1]
		self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)


	def forward(self, x):
		in_ = self.in_conv(x)
		concat_ = [in_]
		for i, block in enumerate(self.encoder[:-1]):
			concat_.append(block(concat_[i]))

		out = self.encoder[-1](concat_[-1])
		for i, block in enumerate(self.decoder):
			out = block(out, concat_[-1-i])

		return self.out_conv(out)

class PatchBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel = 4, stride = 2, padding = 1, norm = True, act = True):
		super().__init__()

		layers = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias = ~norm)])
		if norm: layers.append(nn.BatchNorm2d(out_channels))
		if act: layers.append(nn.LeakyReLU(0.2, True))
		self.patchblck = nn.Sequential(*layers)

	def forward(self, x):
		return self.patchblck(x)

class PatchDiscriminator(nn.Module):
	def __init__(self, in_channels = 3, mid_channels = 64, n_blocks = 3):
		super().__init__()

		layers = nn.ModuleList([PatchBlock(in_channels, mid_channels, norm = False)])
		layers += nn.ModuleList([PatchBlock(mid_channels * 2 ** i, mid_channels * 2 ** (i + 1), stride = 1 if i == (n_blocks-1) else 2)   
							for i in range(n_blocks)])
		layers += nn.ModuleList([PatchBlock(mid_channels * 2 ** n_blocks, 1, stride = 1, norm=False, act=False)])

		self.model = nn.Sequential(*layers)


	def forward(self, x):
		return self.model(x)



