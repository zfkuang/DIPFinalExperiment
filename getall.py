import os
import numpy as np

files = os.listdir('data/save_model')
alist = list(range(0,50))
res = {}
for filename in files:
    alist.remove(int(filename.split('_')[-1]))
    a = np.load('data/save_model/'+filename+'/save.npy')
    res[filename] = a

print(alist)

np.save('data/novel_classifier.npy', res)