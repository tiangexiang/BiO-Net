import os
import sys
from PIL import Image
import numpy as np

def load_data(path, data_name):

    print('loading '+data_name+' data...')
    if data_name == 'monuseg':
        folders = os.listdir(path)
        imgs_list = []
        masks_list = []
        for folder in folders:
            items = os.listdir(os.path.join(path,'x'))
            for item in items:
                imgs_list.append(np.array(Image.open(os.path.join(path,'x',item)).convert('RGB')))
                masks_list.append(np.array(Image.open(os.path.join(path,'y',item)).convert('L')))
    elif data_name == 'tnbc':
        folders = os.listdir(path)

        mask_config = []
        image_config = []
        for folder in folders:
            a,b = list(folder.split('_'))
            
            items = os.listdir(os.path.join(path,folder))
            for item in items:
                entry = []
                fnum, idx = list(item.split('_'))
                fnum = int(fnum)
                idx = int(idx[:-4])
                entry.append(fnum)
                entry.append(idx)
                entry.append(os.path.join(path,folder,item))
                if a[:2] == 'GT':
                    mask_config.append(entry)
                else:
                    image_config.append(entry)

        image_config = sorted(image_config, key = lambda x : (x[0],x[1]))
        mask_config = sorted(mask_config, key = lambda x : (x[0],x[1]))

        imgs_list = []
        masks_list = []
        for i in range(len(image_config)):
            imgs_list.append(np.array(Image.open(image_config[i][-1]).convert('RGB')))
            masks_list.append(np.array(Image.open(mask_config[i][-1])))
    else:
        print('dataset name cannot be recognized!')
        return None, None

    imgs_np = np.asarray(imgs_list)
    masks_np = np.asarray(masks_list)

    x = np.asarray(imgs_np, dtype=np.float32)/255
    y = np.asarray(masks_np, dtype=np.float32)/255
    
    if len(y.shape) == 3:
        y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
        
    print("Successfully loaded data from "+path)
    print("data shape:", x.shape,y.shape)
    
    return x, y
