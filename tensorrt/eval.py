import cv2
from glob import glob
import numpy as np
import time
import pandas as pd

from TRTEngine import ModelTRT


model = ModelTRT(None)
model.predict(np.random.rand(5,608,608))

inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
inst.setUseSpatialPropagation(False)

flow_img = None
prevgray = None

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

true = []
pred = []

for seq in sorted(glob('data/annotations/*.txt'))[66:]: 
    df_lab = pd.read_csv(seq, names=['img', 'x', 'y'])

    seq = seq.replace('annotations', 'sequences')[:-4]

    img_list = sorted(glob(f'{seq}/*'))
    
    for i, path in enumerate(img_list):
        loc = df_lab[df_lab.img==i+1]

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (608,608))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if prevgray is None:
            prevgray = gray.copy()

        if flow_img is not None:
            flow_img = inst.calc(prevgray, gray, warp_flow(flow_img,flow_img))
        else:
            flow_img = inst.calc(prevgray, gray, None)

        img = np.concatenate((img, np.reshape(flow_img*255., (*img.shape[:2], 2))), axis=2)

        prevgray = gray

        img = img * 1./255.

        out = model.predict(img.transpose((2,0,1)))

        pred.append(out.sum()/100)
        true.append(loc.shape[0])

assert np.array(true).shape==np.array(pred).shape
print(np.mean(np.abs(np.array(true)-np.array(pred))))

