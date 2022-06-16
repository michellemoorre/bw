import cv2
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, Dataset

class CustomDataset(Dataset):    
	def __init__(self, x_train):
		# x_train: DataFrame
		self.data = x_train
		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize((256, 256))
		])

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		''' Load image, transform, and return image '''
		image = cv2.imread(self.data[idx])
		
		image_Lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
		image_Lab = self.transform(image_Lab)

		return {'L': image_Lab[0, ...].unsqueeze(0), 'ab': image_Lab[1:, ...]}

class CustomDataModule(pl.LightningDataModule):
	def __init__(self, dataset, batch_size = 32):
		super().__init__()
		self.batch_size = batch_size
		self.dataset = dataset
		self.train, self.val = random_split(dataset, [len(dataset) - 32, 32])

	def train_dataloader(self):
		return DataLoader(self.train, batch_size = self.batch_size, shuffle = True)	

	def val_dataloader(self):
		return DataLoader(self.val, batch_size = self.batch_size)




			