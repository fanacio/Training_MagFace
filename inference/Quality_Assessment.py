
import numpy as np

with open('toy_imgs/feat.list', 'r') as f:
    lines = f.readlines()

img_2_feats = {}
img_2_mag = {}
for line in lines:
    parts = line.strip().split(' ')
    imgname = parts[0]
    feats = [float(e) for e in parts[1:]]
    mag = np.linalg.norm(feats)
    img_2_feats[imgname] = feats/mag
    img_2_mag[imgname] = mag

imgnames = list(img_2_mag.keys())
mags = [img_2_mag[imgname] for imgname in imgnames]
sort_idx = np.argsort(mags)

for idx_ in sort_idx:
    print(imgnames[idx_], float('{0:.2f}'.format(mags[idx_])))

#print([float('{0:.2f}'.format(mags[idx_])) for idx_ in sort_idx])


