import numpy as np
import scipy
import os


use_num = 1000
Classfier_size = 8194

f = np.load('data\\f.npy')
print(f.shape)
f = f[0:use_num]
print(f.shape)

rank = np.linalg.matrix_rank(f)
if rank != min(f.shape):
	print("f rank too small", rank)
	exit()



W = np.zeros((use_num,8194),dtype = float)
'''
for i in range(use_num):

	r0 = np.load('data\\save_model\\base_class_'+str(i)+'\\save.npy')
	W[i] = np.vstack([r0.item()['kernel'],r0.item()['bias']]).flat
	print(r0.item()['kernel'].shape, r0.item()['bias'].shape)
	print(np.vstack([r0.item()['kernel'],r0.item()['bias']]).shape)
	#print(W[i][8192],W[i][8193])
print(W)
'''

W_full = np.load('data\\base_classifier.npy').item()
#print(W_full)

for k,v in W_full.items():
	id = int(k.split('_')[2])
	W[id] = np.vstack([v.item()['kernel'],v.item()['bias']]).flat
	#print('id=', id)
	#print('v=',v.item()['kernel'])


train_data = np.load('train_4096_fc7.npy')
test_data = np.load('test_4096_fc7.npy')
train_label = np.load('trainlabel_350_fc7.npy')
test_label = np.load('testlabel_150_fc7.npy')
print('trls',train_label.shape)
print('tels',test_label.shape)

W_trained = np.zeros((50,8194))
W_num = np.zeros((50,))

feture_avg = np.zeros((50, 4096))
feture_avg_num = np.zeros((50,))
for i in range(train_label.size):
	print(i)
	train_data[i] = train_data[i] / np.linalg.norm(train_data[i])
	feture_avg[train_label[i]] = feture_avg[train_label[i]] + train_data[i]
	feture_avg_num[train_label[i]] = feture_avg_num[train_label[i]] + 1
	#ax = np.linalg.lstsq(np.transpose(f), np.transpose(train_data[i]))
	#W_trained[train_label[i]] = W_trained[train_label[i]] + ax[0].dot(W)
	#W_num[train_label[i]] = W_num[train_label[i]] + 1 

for i in range(50):
	feture_avg[i] = feture_avg[i] / feture_avg_num[i]
	feture_avg[i] = feture_avg[i] / np.linalg.norm(feture_avg[i])

np.save('data\\feature_avg.npy', feture_avg)

alf = 1
beta = 0
W_novel = np.zeros((50,8194))

W_novel_read = np.load('data\\novel_classifier(3).npy').item()
#print(W_full)

for k,v in W_novel_read.items():
	id = int(k.split('_')[2])
	W_novel[id] = np.vstack([v.item()['kernel'],v.item()['bias']]).flat

haoye_trained_classifier = np.load('data/W_new.npy')
print("haoye_trained_classifier", haoye_trained_classifier.shape)



def get_classifier_param_haoye(id):
	return np.hstack([haoye_trained_classifier[id][0:4096],haoye_trained_classifier[id][4097:8193], haoye_trained_classifier[id][4096], haoye_trained_classifier[id][8193]])

def get_classifier_param_linalg(fet):
	ax = np.linalg.lstsq(np.transpose(f), np.transpose(feture_avg[i]))
	return ax[0].dot(W) 

W_trained_w = np.zeros((50,4096,2))
W_trained_b = np.zeros((50,2))

if not os.path.exists('data\\W_trained_w.npy'):

	for i in range(50):
		print(i)
		#ax = np.linalg.lstsq(np.transpose(f), np.transpose(feture_avg[i]))
		#W_trained[i] = get_classifier_param_linalg(feture_avg[i]) 
		W_trained[i] = get_classifier_param_haoye(i) 
		#W_trained[i] = W_trained[i] / W_num[i]
		W_trained_w[i] = W_trained[i].reshape(4097,2)[0:4096]
		#print(W_trained_w[i].shape)
		W_trained_b[i] = W_trained[i].reshape(4097,2)[4096]
		#print(W_trained_b[i])

	np.save('data\\W_trained_w.npy', W_trained_w)
	np.save('data\\W_trained_b.npy', W_trained_b)

else:
	W_trained_w = np.load('data\\W_trained_w.npy')
	W_trained_b = np.load('data\\W_trained_b.npy')


W_novel_w = np.zeros((50,4096,2))
W_novel_b = np.zeros((50,2))

for i in range(50):
	W_novel_w[i] = W_novel[i].reshape(4097,2)[0:4096]
	W_novel_b[i] = W_novel[i].reshape(4097,2)[4096]
	W_trained_w[i] = W_trained_w[i] * alf + W_novel_w[i] * beta
	W_trained_b[i] = W_trained_b[i] * alf + W_novel_b[i] * beta


def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

def cal_label(feture, label_):
	max_ans = 0
	id_ans = -1
	for i in range(50):

		#print(i)
		before_softmax = feture.dot(W_trained_w[i]) + W_trained_b[i]
		after_softmax = softmax(before_softmax)
		#if i==label_:
			#print('label_ ',i,before_softmax[1], before_softmax[0])
		if before_softmax[1] - before_softmax[0] > max_ans:
			#print('update_max ', before_softmax[1] - before_softmax[0] , 'i ',i, before_softmax[1], before_softmax[0])
			max_ans = before_softmax[1] - before_softmax[0]
			id_ans = i
	if id_ans != label_:

		print('label_',label_, (feture.dot(W_trained_w[label_]) + W_trained_b[label_])[1], (feture.dot(W_trained_w[label_]) + W_trained_b[label_])[0])
		print('max_ans',id_ans, (feture.dot(W_trained_w[id_ans]) + W_trained_b[id_ans])[1], (feture.dot(W_trained_w[id_ans]) + W_trained_b[id_ans])[0])
	return id_ans


for i in range(50):
	cnt_i = 0
	cnt_n_i = 0
	for j in range(test_label.size):
		before_softmax = test_data[j].dot(W_trained_w[i]) + W_trained_b[i]
		if test_label[j] == i:
			if before_softmax[1] > before_softmax[0]:
				cnt_i = cnt_i + 1
		if test_label[j] != i:
			if before_softmax[1] < before_softmax[0]:
				cnt_n_i = cnt_n_i + 1
	print ('3 :',cnt_i,'147 :',cnt_n_i)


tot = 0
type_tot = 0
cnt_type_tot = [0,0,0,0]
for i in range(test_label.size):
	#test_data[i] = test_data[i] / np.linalg.norm(test_data[i])
	get_label = cal_label(test_data[i], test_label[i])
	print('get_label',get_label, 'test_label', test_label[i])

	if get_label == test_label[i]:
		tot = tot + 1
		type_tot = type_tot + 1

	if i % 3 == 2:
		print('type_tot', type_tot)
		type_tot = 0
print(tot)

