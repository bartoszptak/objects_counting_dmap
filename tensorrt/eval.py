import cv2
import time
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

from TRTEngine import ModelTRT


def _warp_flow(img, flow):
    """Remap the input optical flow image.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    flow : np.ndarray
        Optical flow image.

    Returns
    -------
    np.ndarray
        Remapped optical flow image.
    """
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


@click.command()
@click.option('--model_path', help='Path to TensorRT model', default='./visdrone_unet++_resnet34_11_fp32.trt')
@click.option('--flow_type', help='DIS optical flow type, available: {medium, fast, ultrafast}', default='medium')
@click.option('--framework', help='Evaluation framework', default='trt')
@click.option('--dataset', help='Type of dataset for evaluation, available: {val, all}', default='val')
def evaluate(model_path, flow_type, framework, dataset):
    model = ModelTRT(None, model_path)

    print('Model warm up...')
    for _ in range(5):
        model.predict(np.random.rand(5,608,608))

    if flow_type == 'medium':
        inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    elif flow_type == 'fast':
        inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    elif flow_type == 'ultrafast':
        inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    else:
        print('Flow type unknown')
        inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    print('Set {} DIS flow type'.format(flow_type))
    inst.setUseSpatialPropagation(False)

    flow_img = None
    prevgray = None

    true = []
    pred = []

    if dataset == 'all':
        sequences = sorted(glob('./annotations/*.txt'))
    else:
        sequences = sorted(glob('./annotations/*.txt'))[66:]
    frames_num = 0
    inference_time = 0
    flow_time = 0
    process_time = 0

    for seq in tqdm(sequences):
        df_lab = pd.read_csv(seq, names=['img', 'x', 'y'])

        seq = seq.replace('annotations', 'sequences')[:-4]

        img_list = sorted(glob(f'{seq}/*'))
        frames_num += len(img_list)
        
        for i, path in enumerate(img_list):
            loc = df_lab[df_lab.img==i+1]

            process_start = time.time()

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (608,608))

            flow_start = time.time()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if prevgray is None:
                prevgray = gray.copy()

            if flow_img is not None:
               flow_img = inst.calc(prevgray, gray, _warp_flow(flow_img,flow_img))
            else:
               flow_img = inst.calc(prevgray, gray, None)

            img = np.concatenate((img, np.reshape(flow_img*255., (*img.shape[:2], 2))), axis=2)

            prevgray = gray

            img = img * 1./255
            flow_time += time.time() - flow_start

            inference_start = time.time()
            out = model.predict(img.transpose((2,0,1)))
            inference_time += time.time() - inference_start

            pred.append(out.sum()/100)

            process_time += time.time() - process_start

            true.append(loc.shape[0])

    mean_process_time = process_time / frames_num
    mean_inference_time = inference_time / frames_num
    mean_flow_time = flow_time / frames_num
    fps = frames_num / inference_time

    print("Frames: {}".format(frames_num))
    print("Mean process time: {:.4f}".format(mean_process_time))
    print("Mean flow time: {:.4f}".format(mean_flow_time))
    print("Mean inference time: {:.4f}".format(mean_inference_time))
    print("Inference FPS: {:.4f}".format(fps))

    assert np.array(true).shape==np.array(pred).shape
    mae = np.mean(np.abs(np.array(true)-np.array(pred)))
    rmse = np.sqrt(np.mean(np.power(np.array(true)-np.array(pred), 2)))
    print('MAE: {:.4f}'.format(mae))
    print('RMSE: {:.4f}'.format(rmse))


if __name__ == '__main__':
    evaluate()
