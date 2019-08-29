import numpy as np
import os
from os.path import join
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch 
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from os import listdir
import pandas as pd

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

transform = transforms.Compose([
	transforms.Resize((224, 224)),
	#transforms.Resize((299, 299)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])


class DATASET(Dataset):

	def __init__(self, domain_name, label_path, mode=True, path='../dataset_public/'):
		self.path = path
		self.domain_path = join(path, domain_name)
		self.label_path = join(path, domain_name, label_path)
		self.mode = mode
		self.csv_data = pd.read_csv(self.label_path).values
		self.images_path = self.csv_data[:, 0] 
		self.label_data = self.csv_data[:, 1]

		if mode == False:
			self.index = []
			#sample = np.random.randint(0, 345, 20)
			sample = (9, 13, 18 ,19 ,26, 40, 61, 116, 322, 327, 337, 342)
			for i, s in enumerate(sample):
				img_index = [x for x in range(len(self.label_data)) if self.label_data[x] == s]
				self.index.append(img_index[:-1])

			#self.index = np.array(self.index).reshape(-1)
			self.index = np.concatenate(self.index, axis=0)

			self.images_path = self.images_path[self.index]
			self.csv_data = self.csv_data[self.index]
			self.label_data = self.label_data[self.index]

	def __len__(self):
		return len(self.csv_data)

	def __getitem__(self, idx):
		image = Image.open(join(self.path, self.images_path[idx]))
		
		image = transform(image).to(device)

		label = int(self.label_data[idx])
		label = torch.LongTensor([label]).to(device)
		
		return image, label

if __name__ == '__main__':
	x = DATASET('infograph', 'infograph_test.csv')
	x = DataLoader(x, batch_size=50)
