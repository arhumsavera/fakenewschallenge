from tqdm import tqdm
import _pickle as pkl

with open('../data/glove.6B.200d.txt', 'rb') as f:
    data = f.readlines()

ddict = {}
for i in tqdm(data):
    li = i.split()
    k, v = li[0], list(map(float, li[1:]))
    ddict[k] = v
#print(len(ddict), len(ddict['the']))

with open('../data/glove.6B.200d.pkl', 'wb') as f:
    pkl.dump(ddict, f)