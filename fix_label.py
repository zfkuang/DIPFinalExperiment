import numpy as np

#pic_num = 63996

lab = np.load('data\\label.npy')
print((lab == 0).any())
print((lab == 1000).any())

print(lab.shape)
fc7 = np.load('data\\fc7.npy')
print(fc7.shape)

f = open('data\\correct.txt', 'r')
ct = [0]
for i in range(1000):
	ct.append(int(f.readline().split()[1]))

lab1 = np.zeros(lab.size, dtype = int)
for i in range(lab.size):
	lab1[i] = ct[lab[i]+1] - 1


cnt = np.zeros(1000, dtype = int)
f = np.zeros((1000,4096), dtype = float)
for i in range(lab.size):
	#print(i)
	fc7[i] = fc7[i] / np.linalg.norm(fc7[i])
	f[lab1[i]] = f[lab1[i]] + fc7[i]
	cnt[lab1[i]] = cnt[lab1[i]] + 1
	#print(np.sum(np.square(fc7[i])))

for i in range(1000):
	f[i] = f[i] / cnt[i]
	f[i] = f[i] / np.linalg.norm(f[i])
	print(np.linalg.norm(f[i]))


np.save('data\\f.npy', f)



