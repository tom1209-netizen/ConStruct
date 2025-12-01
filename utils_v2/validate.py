# Implementation adapted from PBIP by Qingchen Tang
# Source: https://github.com/QingchenTang/PBIP

"""
CAM generation and visualization utilities
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import ttach as tta
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from .crf import DenseCRF
from .evaluate import ConfusionMatrixAllClass
from .hierarchical_utils import merge_to_parent_predictions, merge_subclass_cams_to_parent
from .pyutils import AverageMeter


def fuse_cams_with_weights(cam2, cam3, cam4, cam_weights=None):
    """
    Fuse CAMs from three scales using either learned per-class weights or
    the legacy fixed coefficients.
    """
    if cam_weights is None:
        return 0.3 * cam2 + 0.3 * cam3 + 0.4 * cam4
    if cam_weights.shape[-1] > 3:
        cam_weights = cam_weights[..., -3:]
    cam_weights = cam_weights / (cam_weights.sum(dim=-1, keepdim=True) + 1e-8)
    w2 = cam_weights[..., 0].unsqueeze(-1).unsqueeze(-1)
    w3 = cam_weights[..., 1].unsqueeze(-1).unsqueeze(-1)
    w4 = cam_weights[..., 2].unsqueeze(-1).unsqueeze(-1)
    return w2 * cam2 + w3 * cam3 + w4 * cam4


def get_seg_label(cams, inputs, label):
    """Generate segmentation labels from CAMs"""
    with torch.no_grad():
        b, c, h, w = inputs.shape
        label = label.view(b, -1, 1, 1).cpu().data.numpy()
        cams = cams.cpu().data.numpy()
        cams = np.maximum(cams, 0)
        
        # Normalize CAMs to [0,1]
        channel_max = np.max(cams, axis=(2, 3), keepdims=True)
        channel_min = np.min(cams, axis=(2, 3), keepdims=True)
        cams = (cams - channel_min) / (channel_max - channel_min + 1e-6)
        cams = cams * label
        cams = torch.from_numpy(cams).float()
        
        # Resize to input dimensions
        cams = F.interpolate(cams, size=(h, w), mode="bilinear", align_corners=True)
        cam_max = torch.max(cams, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        cam_all = torch.cat([cams, bg_cam], dim=1)

    return cams



def validate(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    """Validation function with test-time augmentation AND CRF post-processing."""
    model.eval()
    avg_meter = AverageMeter()
    avg_cam_matrix = ConfusionMatrixAllClass(num_classes=cfg.dataset.cls_num_classes + 1)

    crf_processor = DenseCRF()
    MEAN = [0.66791496, 0.47791372, 0.70623304] # Mean values for normalization, specific to the dataset
    STD = [0.1736589,  0.22564577, 0.19820057] # Standard deviation values.
    std_tensor = torch.tensor(STD, device='cuda').view(1, 3, 1, 1)
    mean_tensor = torch.tensor(MEAN, device='cuda').view(1, 3, 1, 1)
    
    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1])
    ])

    with torch.no_grad():
        for data in tqdm(data_loader,
                         total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data
            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()

            outputs = model(inputs)
            (cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, cls5, cam5,
                prototypes, k_list, feature_map_for_diversity) = outputs 
            
            cam_weights = feature_map_for_diversity #outputs[11] if len(outputs) > 11 else None
            
            cls1 = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_test)
            cls2 = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_test)
            cls3 = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_test)
            cls4 = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_test)
            cls5 = merge_to_parent_predictions(cls5, k_list, method=cfg.train.merge_test) 
            

            
            # cls_loss = cls_loss = cfg.train.l1 * loss1 + cfg.train.l2 * loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4 + cfg.train.l5 * loss5 
                       
            # === Classification accuracy from the strongest classifier head (cls4) === 
            cls4_pred = (torch.sigmoid(cls4) > 0.5).float() 
            cls_label_bin = cls_label.float() 
            all_correct = (cls4_pred == cls_label_bin).all(dim=1).float().mean().item() * 100 
            per_class_acc = (cls4_pred == cls_label_bin).sum(dim=0) / cls4_pred.shape[0]
            avg_per_class_acc = per_class_acc.mean().item() * 100
            avg_meter.add({
                        "cls_all_acc": all_correct,       # % of images where ALL labels are correct
                        "cls_avg_acc": avg_per_class_acc  # mean accuracy across classes
                    }) 
            #======================================================= 
            
            # all_cls_acc4 = (cls4_acc_check == cls_label).all(dim=1).float().sum() / cls4_acc_check.shape[0] * 100
            # avg_cls_acc4 = ((cls4_acc_check == cls_label).sum(dim=0) / cls4_acc_check.shape[0]).mean() * 100
            # avg_meter.add({"all_cls_acc4": all_cls_acc4, "avg_cls_acc4": avg_cls_acc4, "cls_loss": cls_loss})
            
            cam1 = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_test)
            cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
            cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
            cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)
            cam5 = merge_subclass_cams_to_parent(cam5, k_list, method=cfg.train.merge_test)

            for cam in [cam1, cam2, cam3, cam4, cam5]:
                cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False) 
                
            # # Interpolate each CAM from [14, 14] to [224, 224] before averaging
            cam1 = F.interpolate(cam1, size=(224, 224), mode='bilinear', align_corners=False)
            cam2 = F.interpolate(cam2, size=(224, 224), mode='bilinear', align_corners=False)
            cam3 = F.interpolate(cam3, size=(224, 224), mode='bilinear', align_corners=False)
            cam4 = F.interpolate(cam4, size=(224, 224), mode='bilinear', align_corners=False)
            cam5 = F.interpolate(cam5, size=(224, 224), mode='bilinear', align_corners=False)

            # print("cam1", cam1.shape, "cam2", cam2.shape, "cam3", cam3.shape, "cam4", cam4.shape, "cam5" , cam5.shape)

            # Average all interpolated CAMs
            avg_cam = torch.stack([cam1, cam2, cam3, cam4, cam5], dim=0).mean(dim=0)

            # print("avg_cam after averaging", avg_cam.sshape)
            # for tta_trans in tta_transform:
            #     augmented_tensor = tta_trans.augment_image(inputs)
            #     tta_outputs = model(augmented_tensor)
            
            #     cam2 = tta_outputs[3]
            #     cam3 = tta_outputs
            #     cam4 = tta_outputs[7]
            #     k_list = tta_outputs[9]
            #     cam_weights = tta_outputs[11] if len(tta_outputs) > 11 else None
                
            #     cam1 = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_test) 
            #     cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
            #     cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
            #     cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)
            #     cam5 = merge_subclass_cams_to_parent(cam5, k_list, method=cfg.train.merge_test) 
         
                
                # cam1 = get_seg_label(cam1, augmented_tensor, cls_label).cuda()
                # cam2 = get_seg_label(cam2, augmented_tensor, cls_label).cuda()
                # cam2 = tta_trans.deaugment_mask(cam2)
                
                # cam3 = get_seg_label(cam3, augmented_tensor, cls_label).cuda()
                # cam3 = tta_trans.deaugment_mask(cam3)

                # cam4 = get_seg_label(cam4, augmented_tensor, cls_label).cuda()
                # cam4 = tta_trans.deaugment_mask(cam4)

                # fused = fuse_cams_with_weights(cam2, cam3, cam4, cam_weights)
                # fused_cams.append(fused.unsqueeze(dim=0))

            cam_max = torch.max(avg_cam, dim=1, keepdim=True)[0]
            bg_cam = (1 - cam_max) ** 10

            full_probs_tensor = torch.cat([avg_cam, bg_cam], dim=1)
            
            full_probs_tensor = F.softmax(full_probs_tensor, dim=1)
            probs_np = full_probs_tensor.cpu().numpy()

            img_denorm_tensor = (inputs * std_tensor) + mean_tensor
            img_denorm_tensor = torch.clamp(img_denorm_tensor * 255, 0, 255).byte()
            img_np = img_denorm_tensor.permute(0, 2, 3, 1).cpu().numpy()

            # START REFINEMENT 
            # refined_mask, _ = crf_processor.process(probs=probs_np, images=img_np)
            # fuse_label234 = torch.from_numpy(refined_mask).long().cuda()
            
            # avg_cam_matrix.update(labels.detach().clone(), fuse_label234.clone())
            # -------------- END REFINEMENT -------------- 
            USE_CRF = False
            if USE_CRF:
                refined_mask, _ = crf_processor.process(probs=probs_np, images=img_np)
                fuse_label234 = torch.from_numpy(refined_mask).long().cuda()
                avg_cam_matrix.update(labels.detach().clone(), fuse_label234.clone())
            else:
                pseudo_label = torch.argmax(full_probs_tensor, dim=1)
                
                avg_cam_matrix.update(labels.detach().clone(), pseudo_label.clone().cuda()) 


    _, _, iu_per_class, dice_per_class, _, fw_iu = avg_cam_matrix.compute()

    mIoU = iu_per_class[:-1].mean().item() * 100
    mean_dice = dice_per_class[:-1].mean().item() * 100
    fw_iu = fw_iu.item() * 100
    avg_cam_score = avg_cam_matrix.compute()[2]
    
    # Get validation accuracy from avg_meter
    # AverageMeter.get() doesn't support default values, so use try-except
    try:
        val_acc = avg_meter.get('cls_all_acc')
    except KeyError:
        val_acc = 0.0  # Default if not found
    
    # print("avg_cam_score", avg_cam_score) 
    model.train()
    return mIoU, mean_dice, fw_iu, iu_per_class, dice_per_class, val_acc


def validate_test(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    """Validation function with test-time augmentation AND CRF post-processing."""
    model.eval()
    avg_meter = AverageMeter()
    avg_cam_matrix = ConfusionMatrixAllClass(num_classes=cfg.dataset.cls_num_classes + 1)

    crf_processor = DenseCRF()
    MEAN = [0.66791496, 0.47791372, 0.70623304] # Mean values for normalization, specific to the dataset
    STD = [0.1736589,  0.22564577, 0.19820057] # Standard deviation values.
    std_tensor = torch.tensor(STD, device='cuda').view(1, 3, 1, 1)
    mean_tensor = torch.tensor(MEAN, device='cuda').view(1, 3, 1, 1)
    
    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1])
    ])

    with torch.no_grad():
        for data in tqdm(data_loader,
                         total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data
            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()

            outputs = model(inputs)
            (cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, cls5, cam5,
                prototypes, k_list, feature_map_for_diversity) = outputs 
            
            cam_weights = feature_map_for_diversity #outputs[11] if len(outputs) > 11 else None
            
            cls1 = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_test)
            cls2 = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_test)
            cls3 = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_test)
            cls4 = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_test)
            cls5 = merge_to_parent_predictions(cls5, k_list, method=cfg.train.merge_test) 
            

            
            # cls_loss = cls_loss = cfg.train.l1 * loss1 + cfg.train.l2 * loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4 + cfg.train.l5 * loss5 
                       
            # === Classification accuracy from the strongest classifier head (cls4) === 
            cls4_pred = (torch.sigmoid(cls4) > 0.5).float() 
            cls_label_bin = cls_label.float() 
            all_correct = (cls4_pred == cls_label_bin).all(dim=1).float().mean().item() * 100 
            per_class_acc = (cls4_pred == cls_label_bin).sum(dim=0) / cls4_pred.shape[0]
            avg_per_class_acc = per_class_acc.mean().item() * 100
            avg_meter.add({
                        "cls_all_acc": all_correct,       # % of images where ALL labels are correct
                        "cls_avg_acc": avg_per_class_acc  # mean accuracy across classes
                    }) 
            #======================================================= 
            
            # all_cls_acc4 = (cls4_acc_check == cls_label).all(dim=1).float().sum() / cls4_acc_check.shape[0] * 100
            # avg_cls_acc4 = ((cls4_acc_check == cls_label).sum(dim=0) / cls4_acc_check.shape[0]).mean() * 100
            # avg_meter.add({"all_cls_acc4": all_cls_acc4, "avg_cls_acc4": avg_cls_acc4, "cls_loss": cls_loss})
            
            cam1 = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_test)
            cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
            cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
            cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)
            cam5 = merge_subclass_cams_to_parent(cam5, k_list, method=cfg.train.merge_test)

            for cam in [cam1, cam2, cam3, cam4, cam5]:
                cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False) 
                
            # # Interpolate each CAM from [14, 14] to [224, 224] before averaging
            cam1 = F.interpolate(cam1, size=(224, 224), mode='bilinear', align_corners=False)
            cam2 = F.interpolate(cam2, size=(224, 224), mode='bilinear', align_corners=False)
            cam3 = F.interpolate(cam3, size=(224, 224), mode='bilinear', align_corners=False)
            cam4 = F.interpolate(cam4, size=(224, 224), mode='bilinear', align_corners=False)
            cam5 = F.interpolate(cam5, size=(224, 224), mode='bilinear', align_corners=False)

            # print("cam1", cam1.shape, "cam2", cam2.shape, "cam3", cam3.shape, "cam4", cam4.shape, "cam5" , cam5.shape)

            # Average all interpolated CAMs
            avg_cam = torch.stack([cam1, cam2, cam3, cam4, cam5], dim=0).mean(dim=0)

            # print("avg_cam after averaging", avg_cam.sshape)
            # for tta_trans in tta_transform:
            #     augmented_tensor = tta_trans.augment_image(inputs)
            #     tta_outputs = model(augmented_tensor)
            
            #     cam2 = tta_outputs[3]
            #     cam3 = tta_outputs
            #     cam4 = tta_outputs[7]
            #     k_list = tta_outputs[9]
            #     cam_weights = tta_outputs[11] if len(tta_outputs) > 11 else None
                
            #     cam1 = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_test) 
            #     cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
            #     cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
            #     cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)
            #     cam5 = merge_subclass_cams_to_parent(cam5, k_list, method=cfg.train.merge_test) 
         
                
                # cam1 = get_seg_label(cam1, augmented_tensor, cls_label).cuda()
                # cam2 = get_seg_label(cam2, augmented_tensor, cls_label).cuda()
                # cam2 = tta_trans.deaugment_mask(cam2)
                
                # cam3 = get_seg_label(cam3, augmented_tensor, cls_label).cuda()
                # cam3 = tta_trans.deaugment_mask(cam3)

                # cam4 = get_seg_label(cam4, augmented_tensor, cls_label).cuda()
                # cam4 = tta_trans.deaugment_mask(cam4)

                # fused = fuse_cams_with_weights(cam2, cam3, cam4, cam_weights)
                # fused_cams.append(fused.unsqueeze(dim=0))

            cam_max = torch.max(avg_cam, dim=1, keepdim=True)[0]
            bg_cam = (1 - cam_max) ** 10

            full_probs_tensor = torch.cat([avg_cam, bg_cam], dim=1)
            
            full_probs_tensor = F.softmax(full_probs_tensor, dim=1)
            probs_np = full_probs_tensor.cpu().numpy()

            img_denorm_tensor = (inputs * std_tensor) + mean_tensor
            img_denorm_tensor = torch.clamp(img_denorm_tensor * 255, 0, 255).byte()
            img_np = img_denorm_tensor.permute(0, 2, 3, 1).cpu().numpy()

            # START REFINEMENT 
            # refined_mask, _ = crf_processor.process(probs=probs_np, images=img_np)
            # fuse_label234 = torch.from_numpy(refined_mask).long().cuda()
            
            # avg_cam_matrix.update(labels.detach().clone(), fuse_label234.clone())
            # -------------- END REFINEMENT -------------- 
            USE_CRF = True
            if USE_CRF:
                refined_mask, _ = crf_processor.process(probs=probs_np, images=img_np)
                fuse_label234 = torch.from_numpy(refined_mask).long().cuda()
                avg_cam_matrix.update(labels.detach().clone(), fuse_label234.clone())
            else:
                pseudo_label = torch.argmax(full_probs_tensor, dim=1)
                
                avg_cam_matrix.update(labels.detach().clone(), pseudo_label.clone().cuda()) 


    _, _, iu_per_class, dice_per_class, _, fw_iu = avg_cam_matrix.compute()

    mIoU = iu_per_class[:-1].mean().item() * 100
    mean_dice = dice_per_class[:-1].mean().item() * 100
    fw_iu = fw_iu.item() * 100
    avg_cam_score = avg_cam_matrix.compute()[2]
    
    # Get validation accuracy from avg_meter
    # AverageMeter.get() doesn't support default values, so use try-except
    try:
        val_acc = avg_meter.get('cls_all_acc')
    except KeyError:
        val_acc = 0.0  # Default if not found
    
    # print("avg_cam_score", avg_cam_score) 
    model.train()
    return mIoU, mean_dice, fw_iu, iu_per_class, dice_per_class, val_acc 

def generate_cam(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    model.eval()

    ## Instantiate the CRF processor
    crf_processor = DenseCRF()

    ## Get MEAN/STD for de-normalization
    MEAN = [0.66791496, 0.47791372, 0.70623304] # Mean values for normalization, specific to the dataset
    STD = [0.1736589,  0.22564577, 0.19820057] # Standard deviation values.
    std = torch.tensor(STD, device='cuda').view(1, 3, 1, 1)
    mean = torch.tensor(MEAN, device='cuda').view(1, 3, 1, 1)

    with torch.no_grad():
        for data in tqdm(data_loader,
                         total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data

            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()

            outputs = model(inputs)
            cam1 = outputs[1]
            cam2 = outputs[3]
            cam3 = outputs[5]
            cam4 = outputs[7]
            k_list = outputs[9]
            cam_weights = outputs[11] if len(outputs) > 11 else None

            cam1 = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_test)
            cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
            cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
            cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)

            cam1 = get_seg_label(cam1, inputs, cls_label).cuda()
            cam2 = get_seg_label(cam2, inputs, cls_label).cuda()
            cam3 = get_seg_label(cam3, inputs, cls_label).cuda()
            cam4 = get_seg_label(cam4, inputs, cls_label).cuda()

            fuse234 = fuse_cams_with_weights(cam2, cam3, cam4, cam_weights)
            
            # This is the original, blobby prediction. We will replace this.
            # output_fuse234 = torch.argmax(fuse234, dim=1).long()

            ## Prepare inputs for the CRF
            # The CRF needs a probability map, so we apply softmax to the fused CAMs.
            # We add a background channel as the CRF expects probabilities for all classes including background.
            cam_max = torch.max(fuse234, dim=1, keepdim=True)[0]
            bg_cam = (1 - cam_max) ** 10 # Using your background estimation method
            full_probs_tensor = torch.cat([fuse234, bg_cam], dim=1) # Note: The paper uses 5 classes (4 + background)
            
            num_actual_classes = cfg.dataset.cls_num_classes + 1
            full_probs_tensor = F.softmax(full_probs_tensor, dim=1)
            probs_np = full_probs_tensor.cpu().numpy() # Shape: [B, 5, H, W]

            # De-normalize the input image for the CRF's bilateral filter
            img_denorm_tensor = (inputs * std) + mean
            img_denorm_tensor = torch.clamp(img_denorm_tensor * 255, 0, 255).byte()
            img_np = img_denorm_tensor.permute(0, 2, 3, 1).cpu().numpy() # Shape: [B, H, W, C]

            ## CRF Step 4: Apply the CRF
            refined_mask, _ = crf_processor.process(probs=probs_np, images=img_np)
            # The output `refined_mask` is a numpy array of shape [B, H, W] with integer class labels.

            # Convert the refined mask back to a tensor for saving logic
            output_refined = torch.from_numpy(refined_mask).long()

            PALETTE = [
                [255, 0, 0] ,   # TUM
                [0, 255, 0],   # STR
                [0, 0, 255],   # LYM
                [153, 0, 255], # NEC
                [255, 255, 255],     # BACK - Assuming background is class 4 now
            ]

            for i in range(len(output_refined)):
                pred_mask = Image.fromarray(output_refined[i].cpu().squeeze().numpy().astype(np.uint8)).convert('P')
                flat_palette = [val for sublist in PALETTE for val in sublist]
                pred_mask.putpalette(flat_palette)
                pred_mask.save(os.path.join(cfg.work_dir.pred_dir, name[i])) 
    model.train()
    return
