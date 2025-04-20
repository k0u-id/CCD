import numpy as np
import torch
import datasets
import models
from instrumentation import compute_metrics
import os
import clip
import torch.nn.functional as F
import torch.nn as nn

pascal_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                  "bus", "car", "cat", "chair", "cow",
                  "dining table", "dog", "horse", "motorbike", "person, people, human",
                  "potted plant", "sheep", "sofa", "train", "tv monitor"]

templates = ['a clean origami of {}.'] #, 'a photo of many {}.']

def run_test(P, logger):
    dataset = datasets.get_data(P)
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset['test'],
            batch_size = P['bsize'],
            shuffle = False,
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = False,
            pin_memory = True
        )
    
    ###################################################### CLIP

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load('./pretrained/RN50x64.pt.pt', device=device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_weights = []
        for classname in pascal_classes:
        
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)
            
            text_features = clip_model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_features = text_features.mean(dim=0) # [512]
            text_features /= text_features.norm()
            
            text_weights.append(text_features) 
        
        text_weights = torch.stack(text_weights, dim=1).cuda() # [512, 20]
        text_weights = text_weights.permute(1, 0) # [20, 512]

    ###########################################################

    model = models.ImageClassifier(P)
    
    path = os.path.join(P['save_path'], 'bestmodel.pt') # change this line to test model
    model_state, _ = torch.load(path)
    model.load_state_dict(model_state)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Test phase

    phase = 'test'
    model.eval()
    y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
    y_true = np.zeros((len(dataset[phase]), P['num_classes']))
    y_clip = np.zeros((len(dataset[phase]), P['num_classes']))
    y_classifier = np.zeros((len(dataset[phase]), P['num_classes']))
    y_entropy = np.zeros(len(dataset[phase]))
    y_name = []
    batch_stack = 0

    with torch.no_grad():
        with open(os.path.join(P['save_path'], 'entropy.txt'), 'w') as f:
            for batch in dataloader[phase]:
                # Move data to GPU
                image = batch['image'].to(device, non_blocking=True)
                label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
                label_gt = batch['label_vec_true']
                label_vec_true = batch['label_vec_true'].clone().numpy()
                idx = batch['idx']
                id = batch['id']
                h, w = image.shape[-2], image.shape[-1]
                ###################################################### CLIP

                with torch.no_grad():
                    image_features = clip_model.encode_image(image, h, w)
                    image_features /= image_features.norm(dim=-1, keepdim=True) # [16, 512]
                
                    similarity = 100.0 * image_features @ text_weights.T # [16, 20]
                    value, indices = similarity.topk(1)
                    label_i = F.one_hot(indices, num_classes=20)
                    label_vec_clip_1 = torch.sum(label_i, dim=1).float().cpu()

                ###########################################################

                # Forward pass
                optimizer.zero_grad()

                logits = model(image)
                
                if logits.dim() == 1:
                    logits = torch.unsqueeze(logits, 0)
                preds = torch.sigmoid(logits)
                
                pred_max = torch.argmax(preds, dim=1, keepdim=True).cpu()
                # max_label = F.one_hot(pred_max, num_classes=20)
                # max_label = torch.sum(max_label, dim=1).float()
                truefalse = torch.sum(label_vec_clip_1 * label_gt, dim=1)
                
                ####################
                logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).cuda()
                prob = torch.softmax(logits * logit_scale, dim=1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
                ####################
                # print(entropy)

                preds_np = preds.cpu().numpy()
                this_batch_size = preds_np.shape[0]
                y_true_sum = torch.sum(label_gt, dim=1)
                clsnum = np.argmax(label_vec_clip_1, axis=1)

                y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
                y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
                y_entropy[batch_stack : batch_stack+this_batch_size] = entropy.cpu().numpy()
                y_clip[batch_stack : batch_stack+this_batch_size] = label_vec_clip_1
                y_name += id
                batch_stack += this_batch_size
                for i in range(len(id)):
                    f.write(id[i])
                    f.write("   %f" % entropy[i])
                    f.write("   %d" % y_true_sum[i])
                    if truefalse[i]:
                        f.write("   TRUE")
                    else:
                        f.write("   FALSE")
                    f.write("   classifier : ")
                    f.write(pascal_classes[pred_max[i]])
                    f.write("   CLIP : ")
                    f.write(pascal_classes[clsnum[i]]) 
                    
                    f.write('\n')
        f.close()

    # print(y_pred)

    metrics = compute_metrics(y_pred, y_true)
    map_test = metrics['map']
    ap_test = metrics['ap']

    print('Testing procedure completed!')
    print(f'Test mAP : {map_test:.3f}')
    print(f'Test AP : ')
    print(ap_test)

    np.save(os.path.join(P['save_path'], 'test_ap.npy'), ap_test)
    np.save(os.path.join(P['save_path'], 'datas.npy'), 
            {"name" : y_name, "entropy": y_entropy, "pred": y_pred, "true": y_true, "clip": y_clip})