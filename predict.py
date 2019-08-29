import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Function
from model import feature_extractor, predictor
from utils.LoadData import DATASET
import sys
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
from PIL import Image

BATCH_SIZE = 500

mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])

transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')



if __name__ == '__main__':
	
	argument = sys.argv[1:]
	image_path = argument[0]
	target_domain = argument[1]



	if target_domain == 'real':
		source_domain = ['infograph', 'quickdraw', 'sketch']
	elif target_domain == 'infograph':
		source_domain = ['quickdraw', 'real', 'sketch']
	elif target_domain == 'sketch':
		source_domain = ['infograph', 'quickdraw', 'real']
	elif target_domain == 'quickdraw':
		source_domain = ['infograph', 'real', 'sketch']

	source_clf = {}

	images = os.listdir(image_path)

	extractor = feature_extractor().to(device)
	extractor.load_state_dict(torch.load('./models/'+'extractor_'+target_domain+'.pth'))
	extractor.eval()

	pred_list = []
	for source in source_domain:
		print(source)
		source_clf[source] = {}
		source_clf[source]['c1'] = predictor().to(device)
		source_clf[source]['c2'] = predictor().to(device)
		source_clf[source]['c1'].load_state_dict(torch.load('./models/'+source+'_c1_'+target_domain+'.pth'))
		source_clf[source]['c2'].load_state_dict(torch.load('./models/'+source+'_c2_'+target_domain+'.pth'))
		source_clf[source]['c1'].eval()
		source_clf[source]['c2'].eval()
		

	img_batch = []
	for index, img_name in enumerate(images):
		
		#img = transform(Image.open('./dataset_public/test/'+img_name))
		img = transform(Image.open(image_path + img_name))
		#img = img.unsqueeze(0).to(device)
		img_batch.append(img)
		if len(img_batch) != BATCH_SIZE and index+1 != len(images):
			continue

		img_batch = torch.stack(img_batch).to(device)
		
		print('[%d]/[%d]' % (index+1, len(images)))
		with torch.no_grad():
			feature = extractor(img_batch)
			p = 0
			for source in source_domain:
				if isinstance(p, int):
					p = F.softmax(source_clf[source]['c1'](feature), dim=1)
					p += F.softmax(source_clf[source]['c2'](feature), dim=1)	
				else:
					p += F.softmax(source_clf[source]['c1'](feature), dim=1)
					p += F.softmax(source_clf[source]['c2'](feature), dim=1)
				
		img_batch = []	

		output = np.argmax(p.cpu().detach().numpy(), axis=1)
		pred_list.append(output.reshape(-1))
	
	pred_list = np.concatenate([i for i in pred_list], axis=0)

	np.savetxt('./pred_'+target_domain+'.csv', pred_list, fmt='%d')



	images = os.listdir(image_path)

	pred_csv = pd.read_csv('./pred_'+target_domain+'.csv')
	col = pred_csv.columns[0]

	pred_csv = pd.read_csv('./pred_'+target_domain+'.csv').values
	pred_csv = np.concatenate([np.array([col]).reshape(1, 1), pred_csv])

	x = []
	for i, j in zip(images, pred_csv):
		x.append(['test/'+i, j[0]])


	x = np.array(x)
	csv = pd.DataFrame({'image_name': x[:, 0], 'label':x[:, 1]})

	csv.to_csv('./pred_'+target_domain+'.csv', index = False)
	print('--------done--------')

# python predict.py D:\DLCV\final\dataset_public\test\ real