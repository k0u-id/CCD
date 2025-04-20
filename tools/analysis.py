import numpy as np
from tqdm import tqdm

data = np.load('results/20230711_174617_1epoch_test/datas.npy', allow_pickle=True).item()

img_name = data['name']
cls_pred = data['pred']
cls_gt   = data['true']
cls_clip = data['clip']
entropy  = data['entropy']

pascal_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                  "bus", "car", "cat", "chair", "cow",
                  "dining table", "dog", "horse", "motorbike", "person",
                  "potted plant", "sheep", "sofa", "train", "tv monitor"]

#### top 1 inferences ####

m_entropy = 0.0001*entropy.mean()

clip_true_count = 0
cls_true_count = 0
both_true_count = 0
false_count = 0
num_samples = 0 
highent_cls_num_count = 0
lowent_cls_num_count = 0
same_count = 0
dif_true_count = 0

a = cls_pred.copy()
a = (a == a.max(axis=1)[:,None]).astype(int)

for i in tqdm(range(len(img_name))):
    if entropy[i] > m_entropy:
        highent_cls_num_count += cls_gt[i, :].sum()
        continue
    else:
        lowent_cls_num_count += cls_gt[i, :].sum()
        num_samples += 1
        # if clip is true
        if (cls_clip[i, :] * cls_gt[i, :]).sum():            
            # if cls is true 
            if (a[i, :] * cls_gt[i, :]).sum():
                if (a[i, :] * cls_clip[i, :]).sum():
                    same_count += 1
                    both_true_count += 1
                else:
                    both_true_count += 1
                    dif_true_count += 1
            else:
                clip_true_count += 1
        else:
            if (a[i, :] * cls_gt[i, :]).sum():
                cls_true_count += 1
            # both false
            else:
                false_count += 1
                if (a[i, :] * cls_clip[i, :]).sum():
                    same_count += 1
                print(entropy[i])
                print('cls_num : %d' % cls_gt[i,:].sum())
                print('clip label : ' + pascal_classes[np.argmax(cls_clip[i, :])])    
                print('cls_label : ' + pascal_classes[np.argmax(a[i, :])])
                print('gt_label : ' + pascal_classes[np.argmax(cls_gt[i,:])])
                print(img_name[i])

print('whole sample number')
print(len(img_name))
print('mean entropy')
print(m_entropy)
print('samples below threshold')
print(num_samples)
print('both true')
print(both_true_count / num_samples)
print('clip true')
print(clip_true_count / num_samples)
print('cls true')
print(cls_true_count / num_samples)
print('same_pred')
print(same_count / num_samples)
print('dif_true')
print(dif_true_count / num_samples)

print('false')
print(false_count / num_samples)

print('low entropy average cls num')
print(lowent_cls_num_count / num_samples)

print('high entropy average cls num')
print(highent_cls_num_count / (len(img_name) - num_samples))
