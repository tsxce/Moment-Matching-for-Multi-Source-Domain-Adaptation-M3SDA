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
from model import predictor,feature_extractor
from utils.LoadData import DATASET
import sys
import utils.grad_reverse as my_function

import argparse
import os
import random
import shutil
import time
import warnings

class ToRGB(object):

	def __init__(self):
		pass
		
	def __call__(self, sample):

		sample = sample.convert('RGB')
		return sample


###		basic setting 	 ###
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
download = True
BATCH_SIZE = 256
EP = 30
###		-------------	 ###

mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])

rgb_transform = transforms.Compose([
	transforms.Resize((32, 32)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])

gray2rgb_transform = transforms.Compose([
	ToRGB(),
	transforms.Resize((32, 32)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])


def train(encoder, cls_model_1, cls_model_2, optimizer_encoder, optimizer_clf_1, optimizer_clf_2, ep, train_loader, test_loader, src_name, tar_name):
	loss_fn_cls = nn.CrossEntropyLoss()
	loss_l1 = nn.L1Loss()
	ac_list, loss_list = [], []
	max_= 0

	end = time.time()

	for i in range(ep):

		cls_model_1.train()
		cls_model_2.train()
		encoder.train()

		print(i,'/', ep)
		for index, (src_batch, tar_batch) in enumerate(zip(train_loader, test_loader)):
			
			# step 1
			x, y = src_batch
			x = x.to(device)
			y = y.to(device)
			y = y.view(-1)

			feature = encoder(x)
			pred1 = cls_model_1(feature)
			pred2 = cls_model_2(feature)

			loss1 = loss_fn_cls(pred1, y) 
			loss2 = loss_fn_cls(pred2, y)
			loss = loss1 + loss2
		
			optimizer_encoder.zero_grad()
			optimizer_clf_1.zero_grad()
			optimizer_clf_2.zero_grad()
			loss.backward()
			optimizer_encoder.step()
			optimizer_clf_1.step()
			optimizer_clf_2.step()


			optimizer_encoder.zero_grad()
			optimizer_clf_1.zero_grad()
			optimizer_clf_2.zero_grad()
			# step 2
			tar_x, _ = tar_batch
			tar_x = tar_x.to(device)

			s_feature = encoder(x)
			pred_s_1 = cls_model_1(s_feature)
			pred_s_2 = cls_model_2(s_feature)

			t_feature = encoder(tar_x)
			pred_tar_1 = cls_model_1(t_feature)
			pred_tar_2 = cls_model_2(t_feature)

			src_loss = loss_fn_cls(pred_s_1, y) + loss_fn_cls(pred_s_2, y)
			#discrepency_loss = torch.mean(torch.abs(F.softmax(pred_tar_1, dim=1) - F.softmax(pred_tar_2, dim=1)))
			discrepency_loss = loss_l1(F.softmax(pred_tar_1, dim=1), F.softmax(pred_tar_2, dim=1))

			loss = src_loss - discrepency_loss

			optimizer_clf_1.zero_grad()
			optimizer_clf_2.zero_grad()

			loss.backward()

			optimizer_clf_1.step()
			optimizer_clf_2.step()

			optimizer_encoder.zero_grad()
			optimizer_clf_1.zero_grad()
			optimizer_clf_2.zero_grad()

			# step 3
			for i in range(3):
				t_feature = encoder(tar_x)

				pred_tar_1 = cls_model_1(t_feature)
				pred_tar_2 = cls_model_2(t_feature)

				#discrepency_loss = torch.mean(abs(F.softmax(pred_tar_1, dim=1) - F.softmax(pred_tar_2, dim=1)))
				discrepency_loss = loss_l1(F.softmax(pred_tar_1, dim=1), F.softmax(pred_tar_2, dim=1))

				discrepency_loss.backward()
				optimizer_encoder.step()

				optimizer_encoder.zero_grad()
				optimizer_clf_1.zero_grad()
				optimizer_clf_2.zero_grad()



			if index % 10 == 0:
				print('[%d]/[%d]' % (index, min([len(train_loader), len(test_loader)])))

		if i % 10 == 0:
			torch.save(cls_model_1.state_dict(), './model/epoch'+i+'_'+src_name+'2'+tar_name+'_1.pth')
			torch.save(cls_model_2.state_dict(), './model/epoch'+i+'_'+src_name+'2'+tar_name+'_2.pth')
			torch.save(encoder.state_dict(), './model/epoch'+i+'_'+src_name+'2'+tar_name+'.pth')
			

		cls_model_1.eval()
		cls_model_2.eval()
		encoder.eval()

		ac_1 = 0
		ac_2 = 0
		ac_3 = 0

		eval_loader = DATASET(tar_name, tar_name+'_train.csv', mode = False)
		eval_loader = torch.utils.data.DataLoader(
			dataset = eval_loader,
			batch_size = BATCH_SIZE,
			shuffle = True,
		)


		total_loss=0
		with torch.no_grad():
			for batch in eval_loader:
				
				x, y = batch
				x = x.to(device)
				y = y.to(device)
				y = y.view(-1)

				feature = encoder(x)
				
				pred_c1 = cls_model_1(feature)
				pred_c2 = cls_model_2(feature)
				pred_combine = pred_c1 + pred_c2

				ac_1 += np.sum(np.argmax(pred_c1.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
				ac_2 += np.sum(np.argmax(pred_c2.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
				ac_3 += np.sum(np.argmax(pred_combine.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())

				total_loss += loss.item()
		
		print('Accuracy : [%.3f], Avg Loss : [%.4f]' % ((ac_1 / len(eval_loader) / BATCH_SIZE), (total_loss / len(eval_loader))) ) 
		print('Accuracy : [%.3f], Avg Loss : [%.4f]' % ((ac_2 / len(eval_loader) / BATCH_SIZE), (total_loss / len(eval_loader))) ) 
		print('Accuracy : [%.3f], Avg Loss : [%.4f]' % ((ac_3 / len(eval_loader) / BATCH_SIZE), (total_loss / len(eval_loader))) ) 
		
		ac = max([ac_1, ac_2, ac_3])

		ac_list.append(ac/len(eval_loader)/BATCH_SIZE)
		loss_list.append(total_loss / len(eval_loader) / BATCH_SIZE)
		if (ac / len(eval_loader) / BATCH_SIZE) > max_:
			max_ = (ac / len(eval_loader) / BATCH_SIZE)
			torch.save(cls_model_1.state_dict(), './model/mcd_'+src_name+'2'+tar_name+'_1.pth')
			torch.save(cls_model_2.state_dict(), './model/mcd_'+src_name+'2'+tar_name+'_2.pth')
			torch.save(encoder.state_dict(), './model/mcd_'+src_name+'2'+tar_name+'.pth')

	return ac_list, loss_list

def weights_init_uniform(m):
	classname = m.__class__.__name__
	# for every Linear layer in a model..
	if classname.find('Linear') != -1:
		# apply a uniform distribution to the weights and a bias=0
		m.weight.data.uniform_(0.0, 1.0)
		m.bias.data.fill_(0)



def main(src, tar):

	G = feature_extractor().to(device)

	cls_c1 = predictor().to(device)
	cls_c2 = predictor().to(device)

	cls_c1.apply(weights_init_uniform)
	cls_c2.apply(weights_init_uniform)

	###		 dataloader  	 ###
	if src == 'sketch':
		src_train_set = DATASET('sketch', 'sketch_train.csv' )

	elif src == 'infograph':
		src_train_set = DATASET('infograpth', 'infograph_train.csv')

	elif src == 'real':
		src_train_set = DATASET('real', 'real_train.csv')

	elif src == 'quickdraw':
		src_train_set = DATASET('quickdraw', 'quickdraw_train.csv')


	if tar == 'sketch':
		tar_train_set = DATASET('sketch', 'sketch_train.csv' )

	elif tar == 'infograph':
		tar_train_set = DATASET('infograpth', 'infograph_train.csv')

	elif tar == 'real':
		tar_train_set = DATASET('real', 'real_train.csv')

	elif tar == 'quickdraw':
		tar_train_set = DATASET('quickdraw', 'quickdraw_train.csv')



	src_train_loader = torch.utils.data.DataLoader(
		dataset = src_train_set,
		batch_size = BATCH_SIZE,
		shuffle = True,
		)

	tar_train_loader = torch.utils.data.DataLoader(
		dataset = tar_train_set,
		batch_size = BATCH_SIZE,
		shuffle = True,
		)

	optimizer_encoder = optim.Adam(G.parameters() , lr=2e-4, weight_decay=0.0005)
	optimizer_clf_1 = optim.Adam(cls_c1.parameters(), lr=2e-4, weight_decay=0.0005)
	optimizer_clf_2 = optim.Adam(cls_c2.parameters(), lr=2e-4, weight_decay=0.0005)

	# train
	ac_list, loss_list = train(G, cls_c1, cls_c2, optimizer_encoder, optimizer_clf_1, optimizer_clf_2, EP, src_train_loader, tar_train_loader, src, tar)
	ac_list = np.array(ac_list).flatten()
	
	# plot tsne
	loss_list = np.array(loss_list).flatten()
	#epoch = [i for i in range(EP)]
	#my_function.tsne_plot(G, src_train_loader, tar_train_loader, src, tar, BATCH_SIZE, 'mcd', mode=False)

	### plot learning curve  ###
	plt.figure()
	plt.plot(epoch, ac_list)
	plt.xlabel('EPOCH')
	plt.ylabel('Accuracy')
	plt.title('domian_adapt : ' + src + ' to ' + tar)
	plt.savefig('./learning_curve/domian_adapt_' + src + '_to_' + tar + '_accuracy.jpg')

	plt.figure()
	plt.plot(epoch, loss_list)
	plt.xlabel('EPOCH')
	plt.ylabel('Loss')
	plt.title('domian_adapt : ' + src + ' to ' + tar)
	plt.savefig('./learning_curve/domian_adapt_' + src + '_to_' + tar + '_loss.jpg')


if __name__ == '__main__':

	source, target = sys.argv[1:]
	main(source, target)


