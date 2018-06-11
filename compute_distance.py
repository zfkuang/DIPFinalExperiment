import numpy as np

def get_distance(a, b):
    a_norm = np.linalg.norm(a, axis=-1)
    b_norm = np.linalg.norm(b, axis=-1)
    norm_product = np.multiply(a_norm, b_norm)
    dot = np.mean(np.multiply(a,b), axis=-1)
    ones = np.ones(shape=dot.shape)
    return 10 * (ones - np.divide(dot, norm_product))

train = np.load('data/em_train.npy')
test = np.load('data/em_test.npy')

print(train.shape)
print(test.shape)

length = train.shape[0]

dist = np.zeros(shape=(length, length))

for i in range(length):
    for j in range(length):
        dist[i][j] = get_distance(train[i], train[j])
        print(i, j, dist[i][j])

np.save("data/dist_train.npy", dist)

