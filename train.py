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
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
'''
dataset : infograph, quickdraw, real, sketch
'''

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
#print('cuda = ', cuda)
BATCH_SIZE = 128
EP = 15


if __name__ == '__main__':
	
	argument = sys.argv[1:]
	source_domain = argument[:-1]
	target_domain = argument[-1]

	os.makedirs('./model/'+target_domain, exist_ok=True)

	N = len(source_domain)
	# dataloader
	source_dataloader_list = []
	source_clf = {}
	source_loss = {}
	extractor = feature_extractor().to(device)
	#extractor.load_state_dict(torch.load('./model/2'+target_domain+'/extractor_5.pth'))
	extractor_optim = optim.Adam(extractor.parameters(), lr=3e-4)
	min_ = float('inf')


	for source in source_domain:
		print(source)
		dataset = DATASET(source, source+'_train.csv')
		dataset = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True)
		if len(dataset) < min_:
			min_ = len(dataset)

		source_dataloader_list.append(dataset)

		# c1 : for target
		# c2 : for source
		source_clf[source] = {}
		source_clf[source]['c1'] = predictor().to(device)
		source_clf[source]['c2'] = predictor().to(device)
		#source_clf[source]['c1'].load_state_dict(torch.load('./model/2'+target_domain+'/'+source+'_c1_5.pth'))
		#source_clf[source]['c2'].load_state_dict(torch.load('./model/2'+target_domain+'/'+source+'_c2_5.pth'))
		source_clf[source]['optim'] = optim.Adam(list(source_clf[source]['c1'].parameters()) + list(source_clf[source]['c2'].parameters()), lr=3e-4)

	target_dataset = DATASET(target_domain, target_domain+'_train.csv')
	target_dataloader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True)
	if len(target_dataloader) < min_:
		min_ = len(target_dataloader)

	loss_extractor = nn.CrossEntropyLoss()
	loss_l1 = nn.L1Loss()
	loss_l2 = nn.MSELoss()

	for source in source_domain:
		source_loss[source] = {}
		for i in range(1, 3):
			source_loss[source][str(i)] = {}
			source_loss[source][str(i)]['loss'] = []
			source_loss[source][str(i)]['ac'] = []
			
	mcd_loss_plot = []
	dis_loss_plot = []
	# train
	for ep in range(EP):
		print('\n')
		print('epoch ', ep)

		extractor.train()

		source_ac = {}
		for source in source_domain:
			source_clf[source]['c1'] = source_clf[source]['c1'].train()
			source_clf[source]['c2'] = source_clf[source]['c2'].train()
			source_ac[source] = defaultdict(int)

		record = {}
		for source in source_domain:
			record[source] = {}
			for i in range(1, 3):
				record[source][str(i)] = 0
		mcd_loss = 0
		dis_loss = 0


		weights = [0, 0, 0]
		for batch_index, (src_batch, tar_batch) in enumerate(zip(zip(*source_dataloader_list), target_dataloader)):
			

			src_len = len(src_batch)
			loss_cls = 0
			# train extractor and source clssifier
			for index, batch in enumerate(src_batch):
				x, y = batch
				x = x.to(device)
				y = y.to(device)
				y = y.view(-1)

				feature = extractor(x)
				pred1 = source_clf[source_domain[index]]['c1'](feature)
				pred2 = source_clf[source_domain[index]]['c2'](feature)

				source_ac[source_domain[index]]['c1'] += torch.sum(torch.max(pred1, dim=1)[1] == y).item()
				source_ac[source_domain[index]]['c2'] += torch.sum(torch.max(pred2, dim=1)[1] == y).item()
				loss1 = loss_extractor(pred1, y)
				loss2 = loss_extractor(pred2, y)
				loss_cls += loss1 + loss2 
				record[source_domain[index]]['1'] += loss1.item()
				record[source_domain[index]]['2'] += loss2.item()
				

			if batch_index % 10 == 0:
				for source in source_domain:
						print(source)
						print('c1 : [%.8f]' % (source_ac[source]['c1']/(batch_index+1)/BATCH_SIZE))
						print('c2 : [%.8f]' % (source_ac[source]['c2']/(batch_index+1)/BATCH_SIZE))
						weights[index] = max([source_ac[source]['c1'], source_ac[source]['c2']])
						#print('\n')

				
			m1_loss = 0
			m2_loss = 0
			for k in range(1, 3):
				for i_index, batch in enumerate(src_batch):
					x, y = batch
					x = x.to(device)
					y = y.to(device)
					y = y.view(-1)

					tar_x, _ = tar_batch
					tar_x = tar_x.to(device)

					src_feature = extractor(x)
					tar_feature = extractor(tar_x)

					e_src = torch.mean(src_feature**k, dim=0)
					e_tar = torch.mean(tar_feature**k, dim=0)
					m1_dist = e_src.dist(e_tar)
					m1_loss += m1_dist
					for j_index, other_batch in enumerate(src_batch[i_index+1:]):
						other_x, other_y = other_batch
						other_x = other_x.to(device)
						other_y = other_y.to(device)
						other_y = other_y.view(-1)
						other_feature = extractor(other_x)

						e_other = torch.mean(other_feature**k, dim=0)
						m2_dist = e_src.dist(e_other)
						m2_loss += m2_dist


			loss_m =  (EP-ep)/EP * (m1_loss/N + m2_loss/N/(N-1)*2) * 0.8
			mcd_loss += loss_m.item()

			loss = loss_cls + loss_m

			if batch_index % 10 == 0:
				print('[%d]/[%d]' % (batch_index, min_))
				print('class loss : [%.5f]' % (loss_cls))
				print('msd loss : [%.5f]' % (loss_m))

			
			extractor_optim.zero_grad()
			for source in source_domain:
				source_clf[source]['optim'].zero_grad()
			
			loss.backward()

			extractor_optim.step()
			
			for source in source_domain:	
				source_clf[source]['optim'].step()	
				source_clf[source]['optim'].zero_grad()
				
			extractor_optim.zero_grad()
			

			tar_x , _ = tar_batch
			tar_x = tar_x.to(device)
			tar_feature = extractor(tar_x)
			loss = 0
			d_loss = 0
			c_loss = 0
			for index, batch in enumerate(src_batch):
				x, y = batch
				x = x.to(device)
				y = y.to(device)
				y = y.view(-1)

				feature = extractor(x)

				pred1 = source_clf[source_domain[index]]['c1'](feature)
				pred2 = source_clf[source_domain[index]]['c2'](feature)

				c_loss += loss_extractor(pred1, y) + loss_extractor(pred2, y)

				pred_c1 = source_clf[source_domain[index]]['c1'](tar_feature)
				pred_c2 = source_clf[source_domain[index]]['c2'](tar_feature)
				combine1 = (F.softmax(pred_c1, dim=1) + F.softmax(pred_c2, dim=1))/2

				d_loss += loss_l1(pred_c1, pred_c2)


				for index_2, o_batch in enumerate(src_batch[index+1:]):
					
					pred_2_c1 = source_clf[source_domain[index_2+index]]['c1'](tar_feature)
					pred_2_c2 = source_clf[source_domain[index_2+index]]['c2'](tar_feature)
					combine2 = (F.softmax(pred_2_c1, dim=1) + F.softmax(pred_2_c2, dim=1))/2

					d_loss += loss_l1(combine1, combine2) * 0.1


				#discrepency_loss = torch.mean(torch.sum(abs(F.softmax(pred_c1, dim=1) - F.softmax(pred_c2, dim=1)), dim=1))
				#discrepency_loss = loss_l1(F.softmax(pred_c1, dim=1), F.softmax(pred_c2, dim=1))

				#loss += clf_loss - discrepency_loss
			loss = c_loss - d_loss 

			loss.backward()
			extractor_optim.zero_grad()
			for source in source_domain:
				source_clf[source]['optim'].zero_grad()

			for source in source_domain:
				source_clf[source]['optim'].step()

			for source in source_domain:
				source_clf[source]['optim'].zero_grad()
			

			all_dis = 0
			for i in range(3):
				discrepency_loss = 0
				tar_feature = extractor(tar_x)

				for index, _ in enumerate(src_batch):

					pred_c1 = source_clf[source_domain[index]]['c1'](tar_feature)
					pred_c2 = source_clf[source_domain[index]]['c2'](tar_feature)
					combine1 = (F.softmax(pred_c1, dim=1) + F.softmax(pred_c2, dim=1))/2

					discrepency_loss += loss_l1(pred_c1, pred_c2)

					for index2, _ in enumerate(src_batch[index+1:]):

						pred_2_c1 = source_clf[source_domain[index2+index]]['c1'](tar_feature)
						pred_2_c2 = source_clf[source_domain[index2+index]]['c2'](tar_feature)
						combine2 = (F.softmax(pred_2_c1, dim=1) + F.softmax(pred_2_c2, dim=1))/2

						discrepency_loss += loss_l1(combine1, combine2) * 0.1
					#discrepency_loss += torch.mean(torch.sum(abs(F.softmax(pred_c1, dim=1) - F.softmax(pred_c2, dim=1)), dim=1))
					#discrepency_loss += loss_l1(F.softmax(pred_c1, dim=1), F.softmax(pred_c2, dim=1)) 

				all_dis += discrepency_loss.item()

				extractor_optim.zero_grad()

				for source in source_domain:
					source_clf[source]['optim'].zero_grad()

				discrepency_loss.backward()

				extractor_optim.step()
				extractor_optim.zero_grad()
				
				for source in source_domain:
					source_clf[source]['optim'].zero_grad()

			dis_loss += all_dis

			if batch_index % 10 == 0:
				print('Discrepency Loss : [%.4f]' % (all_dis))
				

		###

		for source in source_domain:
			for i in range(1, 3):
				source_loss[source][str(i)]['loss'].append(record[source][str(i)]/min_/BATCH_SIZE)
				source_loss[source][str(i)]['ac'].append(source_ac[source]['c'+str(i)]/min_/BATCH_SIZE)
		dis_loss_plot.append(dis_loss/min_/BATCH_SIZE)
		mcd_loss_plot.append(mcd_loss/min_/BATCH_SIZE)

		###





		extractor.eval()
		for source in source_domain:
			source_clf[source]['c1'] = source_clf[source]['c1'].eval()
			source_clf[source]['c2'] = source_clf[source]['c2'].eval()
		
		source_ac = {}
		eval_loss = {}

		eval_loader = DATASET(target_domain, target_domain+'_train.csv')
		eval_loader = DataLoader(eval_loader, batch_size = BATCH_SIZE, shuffle=True)
		for source in source_domain:
			source_ac[source] = defaultdict(int)
			eval_loss[source] = defaultdict(int)
		fianl_ac = 0
		with torch.no_grad():
			for index, batch in enumerate(eval_loader):
				x, y = batch
				x = x.to(device)
				y = y.to(device)
				y = y.view(-1)

				feature = extractor(x)
				final_pred = 1
				for index_s, source in enumerate(source_domain):
					pred1 = source_clf[source]['c1'](feature)
					pred2 = source_clf[source]['c2'](feature)
					
					eval_loss[source]['c1'] += loss_extractor(pred1, y)
					eval_loss[source]['c2'] += loss_extractor(pred2, y)
					
					if isinstance(final_pred, int):
						final_pred =(F.softmax(pred1, dim=1) + F.softmax(pred2, dim=1))/2	
					else:
						final_pred +=( F.softmax(pred1, dim=1) + F.softmax(pred2, dim=1))/2	
					
					source_ac[source]['c1'] += np.sum(np.argmax(pred1.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
					source_ac[source]['c2'] += np.sum(np.argmax(pred2.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())

				fianl_ac += np.sum(np.argmax(final_pred.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
		for source in source_domain:
			print('Current Source : ', source)
			print('Accuray for c1 : [%.4f]' % (source_ac[source]['c1']/BATCH_SIZE/len(eval_loader)))
			print('Accuray for c2 : [%.4f]' % (source_ac[source]['c2']/BATCH_SIZE/len(eval_loader)))
			print('eval loss c1 : [%.4f]' % (eval_loss[source]['c1']))
			print('eval loss c2 : [%.4f]' % (eval_loss[source]['c2']))
			
		print('Combine Ac : [%.4f]' % (fianl_ac/BATCH_SIZE/len(eval_loader)))




		torch.save(extractor.state_dict(), './model/'+target_domain+'/extractor'+'_'+str(ep)+'.pth')
		for source in source_domain:
			torch.save(source_clf[source]['c1'].state_dict(), './model/'+target_domain+'/'+source+'_c1_'+str(ep)+'.pth')
			torch.save(source_clf[source]['c2'].state_dict(), './model/'+target_domain+'/'+source+'_c2_'+str(ep)+'.pth')
						

	###

	ep_list = [i for i in range(EP)]

	for source in source_domain:
		for i in range(1, 3):
			plt.plot(ep_list, source_loss[source][str(i)]['loss'], label='loss-c'+str(i))
			plt.plot(ep_list, source_loss[source][str(i)]['ac'], label='ac-c'+str(i))
			plt.title(source)
			plt.xlabel('EP')
		plt.legend()
		plt.savefig('./model/' + target_domain + '/' + source + '_loss_ac.jpg')
		plt.show()


	plt.plot(ep_list, dis_loss_plot, label='discrepency loss')
	plt.legend()
	plt.title('discrepency loss')
	plt.xlabel('EP')
	plt.ylabel('loss')
	plt.savefig('./model/' + target_domain + '/discrepency_loss.jpg')
	plt.show()


	plt.plot(ep_list, mcd_loss_plot, label='mc loss')
	plt.legend()
	plt.title('MC loss')
	plt.xlabel('EP')
	plt.ylabel('loss')
	plt.savefig('./model/' + target_domain + '/mc_loss.jpg')
	plt.show()




