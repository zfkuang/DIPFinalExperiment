import numpy as np
import scipy
import os


use_num = 1000
Classfier_size = 8194

f = np.load('data/f.npy')
print('f.shape', f.shape)
rank = np.linalg.matrix_rank(f)
if rank != min(f.shape):
	print("f rank too small", rank)
	exit()


base_fc8 = np.transpose(np.load('data/base_fc8.npy'))
novel_fc8 = np.transpose(np.load('data/novel_fc8.npy'))
print('base_fc8.shape', base_fc8.shape)
print('novel_fc8.shape', novel_fc8.shape)


train_data = np.load('data/trainingData.npy')
test_data = np.load('data/testingData.npy')
train_label = np.array([i for i in range(50) for j in range(10)])
test_label = np.load('data/testingLabel.npy')

print('train_data.shape', train_data.shape)
print('train_label.shape', train_label.shape)
print('test_data.shape', test_data.shape)
print('test_label.shape', test_label.shape)


W_trained = np.zeros((50,4096))
W_num = np.zeros((50,))

feature_avg = np.zeros((50, 4096))
feature_avg_num = np.zeros((50,))
for i in range(train_label.size):
	print(i)
	#train_data[i] = train_data[i] / np.linalg.norm(train_data[i])
	feature_avg[train_label[i]] = feature_avg[train_label[i]] + train_data[i]
	feature_avg_num[train_label[i]] = feature_avg_num[train_label[i]] + 1

for i in range(50):
	feature_avg[i] = feature_avg[i] / feature_avg_num[i]
	#feature_avg[i] = feature_avg[i] / np.linalg.norm(feature_avg[i])

np.save('data/feature_avg.npy', feature_avg)




mc_tained_w = np.zeros((50,4096))
if os.path.exists('data/W_new.npy'):
	mc_Hao_w = np.load('data/W_new.npy')
	# print ("mc_hao_w.shape = ", mc_Hao_w.shape)
	# assert(0)
	#mc_Hao_w.npy (50 * 4096)


def get_classifier_param_linalg(feature):
	ax = np.linalg.lstsq(np.transpose(f), np.transpose(feature))
	return ax[0].dot(base_fc8) 

if not os.path.exists('data/mc_tained_w.npy'):
	for i in range(50):
		print(i)
		# mc_tained_w[i] = get_classifier_param_linalg(feature_avg[i])
		mc_tained_w[i] = mc_Hao_w[i] # Hao God please use this 
	np.save('data/mc_tained_w.npy', mc_tained_w)

else:
	mc_tained_w = np.load('data/mc_tained_w.npy')


alf = 1
beta = 0.3

mc_tained_w = mc_tained_w * alf + novel_fc8 * beta



def cal_label(feature, label_):
	max_ans = 0
	id_ans = -1
	for i in range(50):
		y = feature.dot(mc_tained_w[i]) 
		if y > max_ans:
			max_ans = y
			id_ans = i
	if id_ans != label_:
		# print('error')
		# print(max_ans)
		pass
	return id_ans




tot = 0
right = []

for i in range(test_label.size):
	print('i',i)
	#test_data[i] = test_data[i] / np.linalg.norm(test_data[i])
	get_label = cal_label(test_data[i], test_label[i])
	print('get_label',get_label, 'test_label', test_label[i])
	if get_label + 1 == test_label[i]:
		tot = tot + 1

print('right number:',tot, 'in', test_label.size, 'acc', 1.0 * tot / test_label.size)

