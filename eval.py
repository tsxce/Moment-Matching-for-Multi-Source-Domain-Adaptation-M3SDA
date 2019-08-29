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

'''
dataset : infograph, quickdraw, real, sketch
'''

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
#print('cuda = ', cuda)
BATCH_SIZE = 500


if __name__ == '__main__':
	
	argument = sys.argv[1:]
	source_domain = argument[:-1]
	target_domain = argument[-1]

	
	N = len(source_domain)
	# dataloader
	source_dataloader_list = []
	source_clf = {}

	extractor = feature_extractor().to(device)
	extractor.load_state_dict(torch.load('./models/'+'extractor_'+target_domain+'.pth'))
	for source in source_domain:
		print(source)
		# c1 : for target
		# c2 : for source
		source_clf[source] = {}
		source_clf[source]['c1'] = predictor().to(device)
		source_clf[source]['c2'] = predictor().to(device)
		source_clf[source]['c1'].load_state_dict(torch.load('./models/'+source+'_c1_'+target_domain+'.pth'))
		source_clf[source]['c2'].load_state_dict(torch.load('./models/'+source+'_c2_'+target_domain+'.pth'))

	target_dataset = DATASET(target_domain, target_domain+'_train.csv')
	target_dataloader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True)

	extractor.eval()
	for source in source_domain:
		source_clf[source]['c1'] = source_clf[source]['c1'].eval()
		source_clf[source]['c2'] = source_clf[source]['c2'].eval()
	
	source_ac = {}
	fianl_ac = 0
	for source in source_domain:
		source_ac[source] = defaultdict(int)
	with torch.no_grad():
		for index, batch in enumerate(target_dataloader):

			print('[%d]/[%d]' % (index, len(target_dataloader)))
			x, y = batch
			x = x.to(device)
			y = y.to(device)
			y = y.view(-1)

			feature = extractor(x)
			final_pred = 1
			for source in source_domain:
				pred1 = source_clf[source]['c1'](feature)
				pred2 = source_clf[source]['c2'](feature)
				source_ac[source]['c1'] += np.sum(np.argmax(pred1.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
				source_ac[source]['c2'] += np.sum(np.argmax(pred2.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
				if isinstance(final_pred, int):
					if source == 'quickdraw':
						final_pred = (F.softmax(source_clf[source]['c1'](feature), dim=1) +  F.softmax(source_clf[source]['c2'](feature), dim=1) )/2 
					
					elif source == 'sketch':
						final_pred = (F.softmax(source_clf[source]['c1'](feature), dim=1) +  F.softmax(source_clf[source]['c2'](feature), dim=1) )/2 					
					else:
						final_pred = (F.softmax(source_clf[source]['c1'](feature), dim=1) +  F.softmax(source_clf[source]['c2'](feature), dim=1) )/2 
				
				else:
					if source == 'quickdraw':
						final_pred += (F.softmax(source_clf[source]['c1'](feature), dim=1) +  F.softmax(source_clf[source]['c2'](feature), dim=1) )/2 
							 
					elif source =='sketch':
						ffinal_pred += (F.softmax(source_clf[source]['c1'](feature), dim=1) +  F.softmax(source_clf[source]['c2'](feature), dim=1) )/2 
					
					else:
						final_pred += (F.softmax(source_clf[source]['c1'](feature), dim=1) +  F.softmax(source_clf[source]['c2'](feature), dim=1) )/2 						
					
			fianl_ac += np.sum(np.argmax(final_pred.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
		
	for source in source_domain:
		print('Current Source : ', source)
		print('Accuray for c1 : [%.4f]' % (source_ac[source]['c1']/BATCH_SIZE/len(target_dataloader)))
		print('Accuray for c2 : [%.4f]' % (source_ac[source]['c2']/BATCH_SIZE/len(target_dataloader)))

	print('Combine Ac : [%.4f]' % (fianl_ac/BATCH_SIZE/len(target_dataloader)))

# python eval.py sketch infograph quickdraw real