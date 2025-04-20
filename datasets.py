import os
import json
import numpy as np
from PIL import Image
import torch
import copy
import utils
import imutils
from torch.utils.data import Dataset
from torchvision import transforms


def get_metadata(dataset_name):
    if dataset_name == 'voc12':
        meta = {
            'num_classes': 20,
            'path_to_dataset': './metadata/voc12',
            'path_to_images': '/data/data/VOC2012/JPEGImages'
        }
    elif dataset_name == 'voc07':
        meta = {
            'num_classes': 20,
            'path_to_dataset': './metadata/voc07',
            'path_to_images': '/data/data/VOC2007/JPEGImages'
        }
    elif dataset_name == 'coco':
        meta = {
            'num_classes': 80,
            'path_to_dataset': './metadata/coco',
            'path_to_images': '/data/data/coco'
        }
    elif dataset_name == 'nuswide':
        meta = {
            'num_classes': 81,
            'path_to_dataset': './metadata/nuswide',
            'path_to_images': 'data/nuswide/Flickr'
        }
    else:
        raise NotImplementedError('Metadata dictionary not implemented.')
    return meta

def generate_split(num_ex, frac, rng):
    '''
    Computes indices for a randomized split of num_ex objects into two parts,
    so we return two index vectors: idx_1 and idx_2. Note that idx_1 has length
    (1.0 - frac)*num_ex and idx_2 has length frac*num_ex. Sorted index sets are 
    returned because this function is for splitting, not shuffling. 
    '''
    
    # compute size of each split:
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2
    
    # assign indices to splits:
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[-n_2:])
    
    return (idx_1, idx_2)

class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

class ImageDataset(Dataset):

    def __init__(self, dataset_name, image_ids, label_matrix,
                 resize_long=None, ex_resize_long=None, resize=None, ex_resize=None, colorjitter=False, img_normal=TorchvisionNormalize(), ex_normal=True, hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):

        meta = get_metadata(dataset_name)
        self.num_classes = meta['num_classes']
        self.path_to_images = meta['path_to_images']

        self.image_ids = image_ids
        self.label_matrix = label_matrix

        self.resize_long = resize_long
        self.ex_resize_long = ex_resize_long
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.ex_normal = ex_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch
        self.colorjitter = colorjitter
        
        self.resize = resize
        if resize is not None:
            self.resize = transforms.Resize(resize)
        
        self.ex_resize = ex_resize
        if ex_resize is not None:
            self.ex_resize = transforms.Resize(ex_resize)
        # if self.colorjitter:
        self.color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        # name = self.img_name_list[idx]

        image_path = os.path.join(self.path_to_images, self.image_ids[idx])

        # name_str = name # this contains '.jpg'
        with Image.open(image_path) as I_raw:
            img = I_raw.convert('RGB')
        ex_img = img.copy()
        label = torch.FloatTensor(np.copy(self.label_matrix[idx, :]))

        if self.resize:
            img = self.resize(img)
        if self.ex_resize:
            ex_img = self.ex_resize(ex_img)

        if self.resize_long is not None:
            img = imutils.random_resize_long(np.asarray(img), self.resize_long[0], self.resize_long[1])
        if self.ex_resize_long is not None:    
            ex_img = imutils.random_resize_long(np.asarray(ex_img), self.ex_resize_long[0], self.ex_resize_long[1])       

        if self.colorjitter:
        #     img = torchvision.transforms.ToPILImage()(img)
        #     img = self.colorjitter(img)
            # ex_img = torchvision.transforms.ToPILImage()(ex_img)
            ex_img = self.color_jitter(ex_img)
        
        if self.img_normal:
            img = self.img_normal(img)
        if self.ex_normal:
            ex_img = self.img_normal(ex_img)
            # ex_img = np.asarray(ex_img)/255.
        
        if self.hor_flip:
            img = imutils.random_lr_flip(img)
            ex_img = imutils.random_lr_flip(ex_img)
        
        if self.crop_size is not None:
            if self.crop_method == 'random':
                img = imutils.random_crop(img, self.crop_size, 0)
                ex_img = imutils.random_crop(ex_img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)
                ex_img = imutils.top_left_crop(ex_img, self.crop_size, 0)

        img = np.asarray(img).copy()
        ex_img = np.asarray(ex_img).copy()

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)
            ex_img = imutils.HWC_to_CHW(ex_img)

        # print(img.shape)
        # print(ex_img.shape)

        return {'id': self.image_ids[idx], 'image': img, 'ex_image':ex_img, 'label': label, 'idx':idx}

def get_data(P):
    '''
    Given a parameter dictionary P, initialize and return the specified dataset. 
    '''
    
    # select and return the right dataset:
    if P['dataset'] == 'coco':
        ds = multilabel(P).get_datasets()
    elif P['dataset'] == 'voc07':
        ds = multilabel(P).get_datasets()
    elif P['dataset'] == 'voc12':
        ds = multilabel(P).get_datasets()
    elif P['dataset'] == 'nuswide':
        ds = multilabel(P).get_datasets()
    else:
        raise ValueError('Unknown dataset.')
            
    return ds

def load_data(base_path, P):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
        data[phase]['images'] = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
    return data

class multilabel:

    def __init__(self, P):
        
        # get dataset metadata:
        meta = get_metadata(P['dataset'])
        self.base_path = meta['path_to_dataset']
        
        # load data:
        source_data = load_data(self.base_path, P)
        
        # generate indices to split official train set into train and val:
        # split_idx = {}
        # (split_idx['train'], split_idx['val']) = generate_split(
        #     len(source_data['train']['images']),
        #     P['val_frac'],
        #     np.random.RandomState(P['split_seed'])
        #     )
        
        # subsample split indices: # commenting this out makes the val set map be low?
        # ss_rng = np.random.RandomState(P['ss_seed'])
        # for phase in ['train', 'val']:
        #     num_initial = len(split_idx[phase])
        #     num_final = int(np.round(P['ss_frac_{}'.format(phase)] * num_initial))
        #     split_idx[phase] = split_idx[phase][np.sort(ss_rng.permutation(num_initial)[:num_final])]
        
        # define train set:
        self.train = ImageDataset(
            P['dataset'], # dataset name
            source_data['train']['images'], # image ids 
            source_data['train']['labels'], # label matrix
            resize=P['train_resize'], 
            ex_resize=P['train_resize'],
            colorjitter=P['colorjitter'],
            hor_flip=P['train_flip']
        )

        self.infer = ImageDataset(
            P['dataset'], # dataset name
            source_data['train']['images'], # image ids 
            source_data['train']['labels'], # label matrix
            resize=P['resize'],
            ex_resize=P['resize'],
            ex_normal=False
            # resize_long=P['resize_long']
        )    
    
        self.thres = ImageDataset(
            P['dataset'], # dataset name
            source_data['train']['images'], # image ids 
            source_data['train']['labels'], # label matrix
            resize=P['resize'],
            ex_resize=P['resize'],
            ex_normal=False
            # resize_long=P['resize_long']
        )   

        # define val set:
        self.val = ImageDataset(
            P['dataset'],
            source_data['val']['images'],
            source_data['val']['labels'],
            resize=P['train_resize']
        )
        
        # define test set:
        self.test = ImageDataset(
            P['dataset'],
            source_data['val']['images'],
            source_data['val']['labels'],
            resize=P['train_resize']
        )
        
        # define dict of dataset lengths: 
        self.lengths = {'train': len(self.train), 'thres': len(self.thres), 'infer': len(self.infer), 'val': len(self.val), 'test': len(self.test)}
    
    def get_datasets(self):
        return {'train': self.train, 'thres': self.thres, 'infer': self.infer, 'val': self.val, 'test': self.test}

def parse_categories(categories):
    category_list = []
    id_to_index = {}
    for i in range(len(categories)):
        category_list.append(categories[i]['name'])
        id_to_index[categories[i]['id']] = i
    return (category_list, id_to_index)

def get_category_list(P):
    if P['dataset'] == 'pascal':
        catName_to_catID = {
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3,
            'bottle': 4,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'chair': 8,
            'cow': 9,
            'diningtable': 10,
            'dog': 11,
            'horse': 12,
            'motorbike': 13,
            'person': 14,
            'pottedplant': 15,
            'sheep': 16,
            'sofa': 17,
            'train': 18,
            'tvmonitor': 19
        }
        return list(catName_to_catID.keys())
    
    elif P['dataset'] == 'coco':
        load_path = 'data/coco'
        meta = {}
        meta['category_id_to_index'] = {}
        meta['category_list'] = []

        with open(os.path.join(load_path, 'annotations', 'instances_train2014.json'), 'r') as f:
            D = json.load(f)

        (meta['category_list'], meta['category_id_to_index']) = parse_categories(D['categories'])
        return meta['category_list']

    elif P['dataset'] == 'nuswide':
        pass # TODO
    
    elif P['dataset'] == 'cub':
        pass # TODO