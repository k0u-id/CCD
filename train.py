import numpy as np
import torch
import datasets
import models
from instrumentation import compute_metrics
import os
import torch.nn.functional as F
import clip

from tqdm import tqdm


pascal_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                  "bus", "car", "cat", "chair", "cow",
                  "dining table", "dog", "horse", "motorbike", "person",
                  "potted plant", "sheep", "sofa", "train", "tv monitor"]

coco_classes = ["person", "bicycle", "car", "motorcycle", "airplane",
                "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                "wine glass", "cup", "fork", "knife", "spoon",
                "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut",
                "cake", "chair", "couch", "potted plant", "bed",
                "dining table", "toilet", "tv", "laptop", "mouse",
                "remote",  "keyboard", "cell phone", "microwave", "oven",
                "toaster", "sink", "refrigerator", "book", "clock",
                "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

nus_classes = ["airport", "animal", "beach", "bear", "birds",
               "boats", "book", "bridge", "buildings", "cars",
               "castle", "cat", "cityscape", "clouds", "computer",
               "coral", "cow", "dancing", "dog", "earthquake", 
               "elk", "fire", "fish", "flags", "flowers",
               "food", "fox", "frost", "garden", "glacier",
               "grass", "harbor", "horses", "house", "lake",
               "leaf", "map", "military", "moon", "mountain",
               "nighttime", "ocean", "person", "plane", "plants",
               "police", "protest", "railroad", "rainbow", "reflection",
               "road", "rocks", "running", "sand", "sign",
               "sky", "snow", "soccer", "sports", "statue",
               "street", "sun", "sunset", "surf", "swimmers",
               "tattoo", "temple", "tiger", "tower", "town",
               "toy", "train", "tree", "valley", "vehicle", 
               "water", "waterfall", "wedding", "whales", "window",
               "zebra"]

templates = ['a photo of the {}'] #, 'a photo of many {}.']

def cropping_box(image, pos1, pos2, offset):

    y1, x1 = pos1
    y2, x2 = pos2
    h, w = image.shape[-2], image.shape[-1]

    if y1 > y2:
        ymin = y2 - offset
        ymax = y1 + offset
    else:
        ymin = y1 - offset
        ymax = y2 + offset
    if x1 > x2:
        xmin = x2 - offset
        xmax = x1 + offset
    else:
        xmin = x1 - offset
        xmax = x2 + offset
    
    if ymin < 0:
        ymin = 0
    if ymax > h:
        ymax = h
    if xmin < 0:
        xmin = 0
    if xmax > w:
        xmax = w
    
    cropy = ymax - ymin
    cropx = xmax - xmin
    
    if cropx > cropy:
        scale = 640 / cropx
    else:
        scale = 640 / cropy                

    cropped_img = F.interpolate(image[:, :, ymin:ymax, xmin:xmax], 
                                    (int(cropy * scale), int(cropx * scale)),
                                    mode='bilinear', align_corners=False)

    return cropped_img, (int(xmin), int(ymin), int(xmax), int(ymax))

def run_train(P, logger):
    
    logger.info(P)
    logger.info('###### Train start ######')

    dataset = datasets.get_data(P)
    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(
        dataset['train'],
        batch_size = P['bsize'],
        shuffle = True,
        sampler = None,
        num_workers = P['num_workers'],
        drop_last = False,
        pin_memory = True
    )

    dataloader['thres'] = torch.utils.data.DataLoader(
        dataset['train'],
        batch_size = 1,
        shuffle = True,
        sampler = None,
        num_workers = P['num_workers'],
        drop_last = False,
        pin_memory = True
    )

    for phase in ['infer', 'val']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size = 1,
            shuffle = False,
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = False,
            pin_memory = True
        )
    
    ############################################################# load CLIP

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load('./pretrained/RN50x64.pt', device=device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_weights = []
        for classname in pascal_classes:
        
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)
            
            text_features = clip_model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_features = text_features.mean(dim=0) # [512] (if ResNet, [1024])
            text_features /= text_features.norm()
            
            text_weights.append(text_features) 
        
        text_weights = torch.stack(text_weights, dim=1).cuda() # [512, 20]
        text_weights = text_weights.permute(1, 0) # [20, 512]

    ############################################################# declare models


    model = models.ImageClassifier(P)
    feature_extractor_params = [param for param in list(model.feature_extractor.parameters()) if param.requires_grad]
    onebyone_conv_params = [param for param in list(model.onebyone_conv.parameters()) if param.requires_grad]
    opt_params = [
        {'params': feature_extractor_params, 'lr' : P['lr']},
        {'params': onebyone_conv_params, 'lr' : P['lr_mult'] * P['lr']}
    ]
    optimizer = torch.optim.Adam(opt_params, lr=P['lr'])

    # training loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    bestmap_val = 0
    map_train_grad = 100
    map_train_grad_prev = 100
    save_flag = False

    coeff = P['coeff']
    ratio = P['ratio']
    crop_offset = P['offset_size']
    loss_coeff = P['loss_coeff']
    
    table_fixedclip = []
    table_pseudolabel = []
    table_name = []
    table_flag = torch.ones(len(dataset['infer']))
    
    classwise_logit = torch.zeros(P['num_classes']).cuda()
    class_ones = torch.ones(P['num_classes']).cuda()
    class_zeros = torch.zeros(P['num_classes']).cuda()
    classwise_thres = torch.zeros(P['num_classes']).cuda()
    
    avg_count_bound = P['bound']

    LS_coeff = 1/P['LS_coeff']
    if P['LS']:
        multiplier = 0.05
    else:
        multiplier = 0.001


    ############################################################# train

    for epoch in range(0, P['num_epochs']):

        count_true = 0
        max_count = 0
        max_img_id = 'none'
        avg_count = 0

        temp_logits = torch.zeros(P['num_classes']).cuda()
        temp_counts = torch.zeros(P['num_classes']).cuda()
        pred_probs = torch.zeros(P['num_classes']).cuda()

        if epoch == 2:
            for idx, batch in enumerate(tqdm(dataloader['thres'])):
                with torch.no_grad():
                    id = batch['id']
                    image = batch['image'].to(device, non_blocking=True)
                    label = batch['label'].to(device, non_blocking=True)
                    h, w = image.shape[-2], image.shape[-1]

                    logits_, CAM = model(image, True)
                    preds_ = torch.sigmoid(logits_).detach()
                    preds_ = preds_.squeeze(0)
                    
                    class_above_thres = torch.where(preds_ >= multiplier, class_ones, class_zeros)
                    cls_count = class_above_thres.sum()

                    avg_count += cls_count/100 # this is hyperparam
                    if cls_count > max_count:
                        max_count = cls_count
                        max_img_id = id   

                    if idx % 100 == 0 and idx != 0:
                        logger.info(f'max above thres is {max_count} and image name is {max_img_id} \n avg above thres is {avg_count}')
                        if avg_count_bound < avg_count < avg_count_bound+2:
                            logger.info(f'fixed multiplier is {multiplier}')
                            end_flag = True
                            max_count = 0
                            avg_count = 0
                            break
                        else:
                            multiplier += 0.001
                            max_count = 0
                            avg_count = 0

        if epoch < 2 + P['inf_num']:
            for idx, batch in enumerate(tqdm(dataloader['infer'])):
                with torch.no_grad():
                    id = batch['id']
                    image = batch['image'].to(device, non_blocking=True)
                    ex_image = batch['ex_image']
                    label = batch['label'].to(device, non_blocking=True)
                    h, w = image.shape[-2], image.shape[-1]
                    
                    if epoch == 0:                
                        # global similarity 
                        image_features = clip_model.encode_image(image, h, w)
                        image_features /= image_features.norm(dim=-1, keepdim=True) # [1, 512] if ResNet, [1, 1024]
                        global_similarity = 100.0 * image_features @ text_weights.T # [1, 20]
                        global_softscore = torch.softmax(global_similarity, dim=1).float()

                        tempscore = global_softscore
                        entropy = -torch.sum(tempscore * torch.log(tempscore + 1e-10), dim=1)

                        if entropy > 2:
                            table_flag[idx] = 0
                        else:
                            table_flag[idx] = 1

                            _, topk_indices = tempscore.topk(1)
                            topk_flag = F.one_hot(topk_indices, num_classes=P['num_classes']) 
                            topk_flag = torch.sum(topk_flag, dim=1).float()
                            temp_counts += topk_flag.squeeze(0)
                            topk_softscore = tempscore * topk_flag
                            classwise_logit += topk_softscore.squeeze(0)

                        table_fixedclip.append(tempscore)
                        table_pseudolabel.append(tempscore) # [1, 20]
                        table_name.append(id[0]) 

                    else:
                        # update with CAM alignment
                        logits_, CAM = model(image, True)
                        preds_ = torch.sigmoid(logits_).detach()
                        preds_ = preds_.squeeze(0)
                        
                        class_above_thres = torch.where(preds_ >= multiplier, class_ones, class_zeros)
                        cls_count = class_above_thres.sum()
                        
                        label_count = torch.sum(label)
                        avg_count += cls_count/len(dataset['infer'])
                        if cls_count > max_count:
                            max_count = cls_count
                            max_img_id = id                   

                        if P['update_label'] and epoch > 1 :
                            
                            cls_indices = class_above_thres.nonzero()
                            temp_label = torch.zeros_like(table_pseudolabel[idx])

                            for cls_idx in cls_indices:
                                temp_cam = CAM[:, cls_idx.item()]
                                temp_cam = F.interpolate(temp_cam.unsqueeze(0), (h, w), mode='bilinear', align_corners=False)    
                                temp_cam_norm = (temp_cam - temp_cam.amin(dim=(2,3), keepdim=True)) \
                                                / (temp_cam.amax(dim=(2, 3), keepdim=True) - temp_cam.amin((2, 3), keepdim=True) + 1e-5)

                                # this is getting bounding box
                                temp_above = (temp_cam_norm > 0.95).nonzero()
                                _, _, BOX1_H, BOX1_W = temp_above[0]
                                _, _, BOX2_H, BOX2_W = temp_above[-1]
                                crop_clip, crop_box = cropping_box(image, (BOX1_H, BOX1_W), (BOX2_H, BOX2_W), crop_offset)

                                crop_clip_features = clip_model.encode_image(crop_clip, crop_clip.shape[-2], crop_clip.shape[-1])
                                crop_clip_features /= crop_clip_features.norm(dim=-1, keepdim=True) 
                                crop_similarity = 100.0 * crop_clip_features @ text_weights.T # [1, 20]
                                crop_softscore = torch.softmax(crop_similarity, dim=1) # [1, 20]
                                temp_label = torch.cat((temp_label, crop_softscore), dim=0)
                            
                            temp_label = temp_label.amax(dim=0, keepdim=True)
                            temp_logits += (temp_label / len(dataset['infer'])).squeeze(0)

                            if P['LS']:
                                temp_label = torch.where(temp_label >= LS_coeff,
                                                            temp_label, LS_coeff*class_ones.unsqueeze(0))
                            
                            if P['local_temp']:
                                temp_label = temp_label / clip_bias # normalize bias
                            
                            table_pseudolabel[idx] = (ratio * table_pseudolabel[idx] + (1 - ratio) * temp_label).clamp(0, 1)

        if epoch == 0:
            clip_bias = classwise_logit / temp_counts
            for idx in range(len(table_pseudolabel)):
                if P['LS']:
                    table_pseudolabel[idx] = torch.where(table_pseudolabel[idx] >= LS_coeff,
                                                        table_pseudolabel[idx], LS_coeff*class_ones.unsqueeze(0))
                if P['global_temp']:
                    table_pseudolabel[idx] = (table_pseudolabel[idx] / clip_bias.unsqueeze(0)).clamp(0, 1)
                
            if coeff > 0:
                total_alpha = model.alpha #* pred_alpha
                model.alpha = coeff*total_alpha

        if epoch == 2:
            logger.info(f'max above thres is {max_count} and image name is {max_img_id} \n avg above thres is {avg_count}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                train_batch_count = 0
                batch_stack = 0
                train_loss = 0
                y_train_pred = np.zeros((len(dataset[phase]), P['num_classes']))
                y_train_lab = np.zeros((len(dataset[phase]), P['num_classes']))

            else:
                model.eval()
                y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
                y_true = np.zeros((len(dataset[phase]), P['num_classes']))
                batch_stack = 0
            
            with torch.set_grad_enabled(phase == 'train'):
                for batch in tqdm(dataloader[phase]):
                    
                    # Move data to GPU
                    id = batch['id']
                    image = batch['image'].to(device, non_blocking=True)
                    ex_image = batch['ex_image'].to(device, non_blocking=True)
                    label = batch['label'].to(device, non_blocking=True)
                    label_np = batch['label'].clone().numpy()

                    # idx = batch['idx']
                    N, _, H, W = image.shape

                    batch_label = torch.zeros_like(label).cuda()
                    batch_flag = torch.zeros(N, 1).cuda()

                    # Forward pass
                    optimizer.zero_grad()
                    logits = model(image)
                    
                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, 0)
                    logits_target = logits.clone().detach()

                    preds = torch.sigmoid(logits)
                    
                    if phase == 'train':
                        
                        for j in range(N):
                            # find the idx with id
                            idx = table_name.index(id[j])
                            # infer label with idx
                            batch_label[j] = table_pseudolabel[idx]
                            batch_flag[j] = table_flag[idx]

                        loss_ce = F.binary_cross_entropy_with_logits(logits, batch_label, reduction='none')

                        if epoch >= 2 and P['use_consist'] > 0:
                            ex_logits = model(ex_image)
                            loss_consist = F.binary_cross_entropy_with_logits(ex_logits, torch.sigmoid(logits_target), reduction='none')
                            batch_flag = (batch_flag + loss_coeff).clamp(min=0, max=1)                        
                            loss_ce = (batch_flag * loss_ce).mean()
                            loss_consist = loss_consist.mean()
                            loss = loss_ce + P['use_consist'] * loss_consist
                        else:
                            loss = loss_ce.mean()

                        loss.backward()
                        optimizer.step()
                        train_batch_count += 1
                        train_loss += loss.item() 

                        this_batch_size = label_np.shape[0]

                        y_train_pred[batch_stack : batch_stack+this_batch_size] = preds.detach().cpu().numpy()
                        y_train_lab[batch_stack : batch_stack+this_batch_size] = batch_label.detach().cpu().numpy()
                        batch_stack += this_batch_size

                    else:
                        preds_np = preds.cpu().numpy()
                        # print(preds_np)
                        this_batch_size = preds_np.shape[0]
                        y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
                        y_true[batch_stack : batch_stack+this_batch_size] = label_np
                        batch_stack += this_batch_size


        meanloss = train_loss / train_batch_count
        logger.info(f"Epoch {epoch} : loss mean {meanloss}")
        metrics_train = compute_metrics(y_train_pred, y_train_lab)
        metrics = compute_metrics(y_pred, y_true)
        del y_train_pred
        del y_train_lab
        del y_pred
        del y_true

        map_train = metrics_train['map']
        map_val = metrics['map']
        
        logger.info(f"Epoch {epoch} : train mAP {map_train:.3f}")
        logger.info(f"Epoch {epoch} : test mAP {map_val:.3f}")

        if epoch >= 1:
            map_train_grad = map_train - map_train_prev
        map_train_prev = map_train      

        if bestmap_val < map_val:
            bestmap_val = map_val
            bestmap_epoch = epoch
            bestmap_ap = metrics['ap']
            
            logger.info(f'Saving model weight for best val mAP {bestmap_val:.3f}')
            path = os.path.join(P['save_path'], 'bestmodel.pt')
            torch.save((model.state_dict(), P), path)

        if map_train_grad - map_train_grad_prev > 0 and save_flag is False:
            bestccd_map = map_val
            bestccd_epoch = epoch
            bestccd_ap = metrics['ap']

            logger.info(f'Saving model weight for best val mAP {bestccd_map:.3f}')
            ccd_path = os.path.join(P['save_path'], 'bestccd.pt')
            torch.save((model.state_dict(), P), ccd_path)
            save_flag = True

        map_train_grad_prev = map_train_grad

    logger.info('Training procedure completed!')
    logger.info(f'best model(with GT) is trained until epoch {bestmap_epoch} with test mAP {bestmap_val:.3f}')
    logger.info(bestmap_ap)
    logger.info(f'best model(without GT) is trained until epoch {bestccd_epoch} with test mAP {bestccd_map:.3f}')
    logger.info(bestccd_ap)

