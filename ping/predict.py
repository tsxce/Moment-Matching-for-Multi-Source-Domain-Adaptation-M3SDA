import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
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

images = os.listdir('./dataset_public/test')


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
	source_domain = argument[:-1]
	target_domain = argument[-1]
	source_clf = {}


	extractor = feature_extractor().to(device)
	extractor.load_state_dict(torch.load('./model/2real/extractor_16.pth'))
	extractor.eval()

	pred_list = []
	for source in source_domain:
		print(source)
		source_clf[source] = {}
		source_clf[source]['c1'] = predictor().to(device)
		source_clf[source]['c2'] = predictor().to(device)
		source_clf[source]['c1'].load_state_dict(torch.load('./model/2real/'+source+'_c1_16.pth'))
		source_clf[source]['c2'].load_state_dict(torch.load('./model/2real/'+source+'_c2_16.pth'))
		source_clf[source]['c1'].eval()
		source_clf[source]['c2'].eval()
		

	img_batch = []
	for index, img_name in enumerate(images):
		
		img = transform(Image.open('./dataset_public/test/'+img_name))
		#img = img.unsqueeze(0).to(device)
		img_batch.append(img)
		if len(img_batch) != BATCH_SIZE and index+1 != len(images):
			continue

		img_batch = torch.stack(img_batch).to(device)
		total = 0.44 + 0.71 + 0.54
		print('[%d]/[%d]' % (index+1, len(images)))
		with torch.no_grad():
			feature = extractor(img_batch)
			p = 0
			for source in source_domain:
				if isinstance(p, int):
					if source == 'quickdraw':
						p = F.softmax(source_clf[source]['c1'](feature), dim=1) * 0.26 / total
						p += F.softmax(source_clf[source]['c2'](feature), dim=1) * 0.26 / total
					elif source == 'sketch':
						p = F.softmax(source_clf[source]['c1'](feature), dim=1) * 0.41 / total
						p += F.softmax(source_clf[source]['c2'](feature), dim=1) * 0.41 / total
					else:
						p = F.softmax(source_clf[source]['c1'](feature), dim=1) * 0.31 / total
						p += F.softmax(source_clf[source]['c2'](feature), dim=1) * 0.31 / total
					

				else:
					if source == 'quickdraw':
						p += F.softmax(source_clf[source]['c1'](feature), dim=1) * 0.26 / total
						p += F.softmax(source_clf[source]['c2'](feature), dim=1) * 0.26 / total
					elif source =='sketch':
						p += F.softmax(source_clf[source]['c1'](feature), dim=1) * 0.41 / total
						p += F.softmax(source_clf[source]['c2'](feature), dim=1) * 0.41 / total
					else:
						p += F.softmax(source_clf[source]['c1'](feature), dim=1) * 0.31 / total
						p += F.softmax(source_clf[source]['c2'](feature), dim=1) * 0.31 / total
					
		img_batch = []	

		output = np.argmax(p.cpu().detach().numpy(), axis=1)
		pred_list.append(output.reshape(-1))
	
	pred_list = np.concatenate([i for i in pred_list], axis=0)

	np.savetxt('pred.csv', pred_list, fmt='%d')



	images = os.listdir('./dataset_public/test')

	pred_csv = pd.read_csv('./pred.csv', )
	col = pred_csv.columns[0]

	pred_csv = pd.read_csv('./pred.csv', ).values
	pred_csv = np.concatenate([np.array([col]).reshape(1, 1), pred_csv])

	x = []
	for i, j in zip(images, pred_csv):
		x.append(['test/'+i, j[0]])


	x = np.array(x)
	csv = pd.DataFrame({'image_name': x[:, 0], 'label':x[:, 1]})

	csv.to_csv('./pred.csv', index = False)
