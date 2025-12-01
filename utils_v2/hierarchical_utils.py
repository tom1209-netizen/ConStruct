# Implementation adapted from PBIP by Qingchen Tang
# Source: https://github.com/QingchenTang/PBIP

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import torchvision.transforms.functional as TF
from PIL import Image

# This module implements the adaptive thresholding from Equation 4
class MaskAdapter_DynamicThreshold(nn.Module):
    def __init__(self, alpha, mask_cam=False):
        super(MaskAdapter_DynamicThreshold, self).__init__()

        self.alpha = alpha
        self.mask_cam = mask_cam

        print(f"MaskAdapter_DynamicThreshold:")
        print(f"  alpha: {alpha}")
        print(f"  mask_cam: {mask_cam}")

    def forward(self, x):
        binary_mask = []
        for i in range(x.shape[0]):
            th = torch.max(x[i]) * self.alpha

            # Creates a binary mask where values >= threshold are 1, others are 0
            binary_mask.append(
                torch.where(x[i] >= th, torch.ones_like(x[0]), torch.zeros_like(x[0]))
            )
        binary_mask = torch.stack(binary_mask, dim=0)

        if self.mask_cam:
            return x * binary_mask
        else:
            return binary_mask

class FeatureExtractor:
    """
    Extracts foreground and background features from images using CAMs and CONCH.
    
    This class implements the feature extraction pipeline:
    1. Resizes CAMs and images to CONCH input size (224x224)
    2. Applies adaptive thresholding to create binary masks
    3. Separates images into foreground/background regions
    4. Extracts features using CONCH vision model
    """
    def __init__(self, mask_adapter, conch_size=224):
        """
        Initialize the feature extractor.
        
        Args:
            mask_adapter: MaskAdapter_DynamicThreshold instance for adaptive thresholding
            conch_size: Target size for CONCH model input (default: 224)
        """
        self.mask_adapter = mask_adapter
        self.conch_size = conch_size
        
    def prepare_cam_mask(self, cam, N):
        """
        Prepares CAMs for mask generation by resizing and reshaping.
        
        Example with batch_size=8, 12 subclasses:
            Input: cam [8, 12, 224, 224] - CAMs at original resolution
            Process:
                1. Resize to CONCH size: cam_224 [8, 12, 224, 224] (if already 224, no change)
                2. Reshape: [8*12, 1, 224, 224] = [96, 1, 224, 224] (flatten batch and class dims)
                3. Apply thresholding: cam_224_mask [96, 1, 224, 224] (binary masks)
            Output:
                cam_224: [96, 1, 224, 224] - reshaped CAMs
                cam_224_mask: [96, 1, 224, 224] - binary masks (0 or 1)
        
        Args:
            cam: Class Activation Maps, shape [batch_size, num_classes, H, W]
            N: Batch size (used for reshaping)
        
        Returns:
            cam_224: Resized and reshaped CAMs, shape [N*num_classes, 1, conch_size, conch_size]
            cam_224_mask: Binary masks from adaptive thresholding, same shape as cam_224
        """
        # Resize CAMs to CONCH input size (224x224)
        # Example: [8, 12, 224, 224] -> [8, 12, 224, 224] (if already 224) or [8, 12, 56, 56] -> [8, 12, 224, 224]
        cam_224 = F.interpolate(cam, (self.conch_size, self.conch_size), 
                              mode="bilinear", align_corners=True)
        # Reshape to flatten batch and class dimensions for independent thresholding
        # Example: [8, 12, 224, 224] -> [96, 1, 224, 224] (8*12=96 CAMs, each treated independently)
        cam_224 = cam_224.reshape(N * cam.size(1), 1, self.conch_size, self.conch_size)

        # Apply adaptive thresholding to create binary masks
        # Each CAM gets its own threshold = max(CAM) * alpha
        # Example: If max(cam_224[0]) = 0.8 and alpha=0.2, threshold = 0.16
        #          Values >= 0.16 become 1, others become 0
        cam_224_mask = self.mask_adapter(cam_224)
        
        return cam_224, cam_224_mask
        
    def prepare_image(self, img):
        """
        Resizes input images to CONCH input size.
        
        Example:
            Input: img [8, 3, 512, 512] - batch of 8 RGB images at original resolution
            Output: [8, 3, 224, 224] - resized to CONCH input size
        
        Args:
            img: Input images, shape [batch_size, C, H, W]
        
        Returns:
            Resized images, shape [batch_size, C, conch_size, conch_size]
        """
        return F.interpolate(img, (self.conch_size, self.conch_size), 
                           mode="bilinear", align_corners=True)
        
    # Implements the element-wise multiplication from Equation 5
    def extract_features(self, img_224, cam_224, cam_224_mask, label):
        """
        Separates images into foreground and background regions based on CAMs and labels.
        
        This implements Equation 5 from the paper:
        - X_FG = b * M * X  (foreground: high CAM activation AND inside mask)
        - X_BG = (1-b) * (1-M) * X  (background: low CAM activation AND outside mask)
        
        Example with batch_size=8, 12 subclasses, 3 channels:
            Input:
                img_224: [8, 3, 224, 224] - resized images
                cam_224: [8, 12, 224, 224] - CAMs for all subclasses
                cam_224_mask: [8, 12, 224, 224] - binary masks (from thresholding)
                label: [8, 12] - binary labels, e.g., label[0, 2] = 1 means sample 0 belongs to subclass 2
            
            Process:
                1. Find positive samples: batch_indices = [0, 1, 2], class_indices = [2, 5, 8]
                   (3 samples with positive labels)
                2. Select relevant data:
                   - img_selected: [3, 3, 224, 224] - images for positive samples
                   - cam_selected: [3, 224, 224] - CAMs for the specific subclasses
                   - mask_selected: [3, 224, 224] - masks for the specific subclasses
                3. Create foreground/background:
                   - fg_features = cam * img (weighted by CAM activation)
                   - bg_features = (1-cam) * img (inverse weighted)
                   - fg_masks: binary mask for foreground regions
                   - bg_masks: inverse mask for background regions
            
            Output:
                fg_features: [3, 3, 224, 224] - foreground regions (high CAM values)
                bg_features: [3, 3, 224, 224] - background regions (low CAM values)
                fg_masks: [3, 1, 224, 224] - binary mask (1 = foreground region)
                bg_masks: [3, 1, 224, 224] - binary mask (1 = background region)
        
        Args:
            img_224: Resized images, shape [batch_size, C, H, W]
            cam_224: Resized CAMs, shape [batch_size, num_classes, H, W]
            cam_224_mask: Binary masks from thresholding, shape [batch_size, num_classes, H, W]
            label: Binary labels, shape [batch_size, num_classes]
        
        Returns:
            fg_features: Foreground image regions, shape [N, C, H, W] where N = number of positive samples
            bg_features: Background image regions, shape [N, C, H, W]
            fg_masks: Foreground binary masks, shape [N, 1, H, W]
            bg_masks: Background binary masks, shape [N, 1, H, W]
        """
        # Find all positions where labels are positive (== 1)
        # Example: if label[0, 2] = 1, label[1, 5] = 1, label[2, 8] = 1
        #          then batch_indices = [0, 1, 2], class_indices = [2, 5, 8]
        batch_indices, class_indices = torch.where(label == 1)
        
        # Select images, CAMs, and masks for positive samples only
        # Example: img_224[0, 1, 2] -> [3, 3, 224, 224] (3 positive samples)
        img_selected = img_224[batch_indices]  # [N, C, H, W]
        # Example: cam_224[0, 2], cam_224[1, 5], cam_224[2, 8] -> [3, 224, 224]
        cam_selected = cam_224[batch_indices, class_indices]  # [N, H, W]
        # Example: cam_224_mask[0, 2], cam_224_mask[1, 5], cam_224_mask[2, 8] -> [3, 224, 224]
        mask_selected = cam_224_mask[batch_indices, class_indices]  # [N, H, W]
        
        # Add channel dimension for broadcasting
        # Example: [3, 224, 224] -> [3, 1, 224, 224]
        cam_expanded = cam_selected.unsqueeze(1)  # [N, 1, H, W]
        mask_expanded = mask_selected.unsqueeze(1)  # [N, 1, H, W]
        
        # X_FG = b * M * X
        # Foreground: multiply image by CAM activation (b) and mask (M)
        # Example: If cam[0, 0, 100, 100] = 0.8 and mask[0, 0, 100, 100] = 1,
        #          then fg_features[0, :, 100, 100] = 0.8 * img[0, :, 100, 100] (weighted by activation)
        #          This emphasizes high-activation regions that pass the threshold
        fg_features = cam_expanded * img_selected  # [N, C, H, W]
        
        # X_BG = (1-b) * (1-M) * X
        # Background: multiply image by inverse CAM (1-b) and inverse mask (1-M)
        # Example: If cam[0, 0, 100, 100] = 0.8, then (1-cam) = 0.2
        #          bg_features[0, :, 100, 100] = 0.2 * img[0, :, 100, 100] (weighted by low activation)
        #          This emphasizes low-activation regions outside the mask
        bg_features = (1 - cam_expanded) * img_selected  # [N, C, H, W]
        
        # fg_masks and bg_masks are used to zero out irrelevant regions before feeding to CONCH
        # These binary masks ensure only relevant regions contribute to feature extraction
        # Example: fg_masks[0, 0, 100, 100] = 1 means pixel (100, 100) is in foreground
        #          bg_masks[0, 0, 100, 100] = 0 means pixel (100, 100) is NOT in background
        fg_masks = mask_expanded  # [N, 1, H, W] - 1 = foreground region
        bg_masks = 1 - mask_expanded  # [N, 1, H, W] - 1 = background region
        
        return fg_features, bg_features, fg_masks, bg_masks
        
    # Extracts features from the masked images
    def get_masked_features(self, fg_features, bg_features, fg_masks, bg_masks, model_conch):
        """
        Extracts feature vectors from foreground and background regions using CONCH.
        
        This applies the binary masks to zero out irrelevant regions, then passes
        the masked images through CONCH's encode_image to get feature vectors.
        
        Example with 3 positive samples:
            Input:
                fg_features: [3, 3, 224, 224] - foreground image regions (weighted by CAM)
                bg_features: [3, 3, 224, 224] - background image regions (weighted by inverse CAM)
                fg_masks: [3, 1, 224, 224] - binary mask (1 = foreground pixel)
                bg_masks: [3, 1, 224, 224] - binary mask (1 = background pixel)
            
            Process:
                1. Apply masks to zero out irrelevant regions:
                   - fg_features * fg_masks: pixels outside mask become 0
                   - bg_features * bg_masks: pixels outside mask become 0
                2. Pass through CONCH encode_image:
                   - encode_image processes [3, 3, 224, 224] -> [3, 512] (global feature vector)
            
            Output:
                fg_img_features: [3, 512] - feature vector for foreground region
                bg_img_features: [3, 512] - feature vector for background region
                Each feature vector represents the visual content of that region
        
        Args:
            fg_features: Foreground image regions, shape [N, C, H, W]
            bg_features: Background image regions, shape [N, C, H, W]
            fg_masks: Foreground binary masks, shape [N, 1, H, W]
            bg_masks: Background binary masks, shape [N, 1, H, W]
            model_conch: CONCH model with encode_image method
        
        Returns:
            fg_img_features: Foreground feature vectors, shape [N, D] where D = feature dimension (e.g., 512)
            bg_img_features: Background feature vectors, shape [N, D]
        """
        # Apply masks to zero out irrelevant regions before feature extraction
        # Example: fg_features[0, :, 100, 100] * fg_masks[0, 0, 100, 100]
        #          If mask = 0, pixel becomes 0 (ignored)
        #          If mask = 1, pixel keeps its weighted value (included)
        # Gets the feature vector for the foreground region and background region
        # CONCH encode_image processes the masked images and outputs global feature vectors
        # Example: [3, 3, 224, 224] -> [3, 512] (one 512-dim vector per sample)
        with torch.no_grad():
            # old: fg_imgfg_img_features = clip_model.vision_model(fg_features * fg_masks) _features
            fg_img_features = model_conch.encode_image(fg_features * fg_masks)  # [N, D]
            bg_img_features = model_conch.encode_image(bg_features * bg_masks)  # [N, D]
            
        return fg_img_features, bg_img_features

    # The main function that orchestrates the whole process for a batch
    def process_batch(self, inputs, cam, label, model_conch):
        """
        Main pipeline: extracts foreground and background features from a batch of images.
        
        Complete workflow:
        1. Check if there are any positive samples (early exit if none)
        2. Resize CAMs to CONCH input size (224x224)
        3. Apply adaptive thresholding to create binary masks
        4. Separate images into foreground/background regions
        5. Extract feature vectors using CONCH
        
        Example with batch_size=8, 12 subclasses, 3 channels:
            Input:
                inputs: [8, 3, 512, 512] - batch of 8 RGB images at original resolution
                cam: [8, 12, 224, 224] - CAMs for 12 subclasses (already at 224x224)
                label: [8, 12] - binary labels, e.g., label[0, 2] = 1, label[1, 5] = 1
                model_conch: CONCH model instance
            
            Process:
                1. Check: torch.any(label == 1) -> True (has positive samples)
                2. Resize CAMs: cam_224 [8, 12, 224, 224] (already 224, no change)
                3. Threshold: cam_224_mask [8, 12, 224, 224] (binary masks)
                4. Extract regions:
                   - Find positive samples: batch_indices = [0, 1], class_indices = [2, 5]
                   - fg_features: [2, 3, 224, 224] - foreground regions
                   - bg_features: [2, 3, 224, 224] - background regions
                   - fg_masks: [2, 1, 224, 224] - foreground masks
                   - bg_masks: [2, 1, 224, 224] - background masks
                5. Extract features:
                   - fg_features: [2, 512] - foreground feature vectors
                   - bg_features: [2, 512] - background feature vectors
            
            Output:
                Dictionary with:
                - 'fg_features': [2, 512] - foreground feature vectors
                - 'bg_features': [2, 512] - background feature vectors
                - 'fg_masks': [2, 1, 224, 224] - foreground masks
                - 'bg_masks': [2, 1, 224, 224] - background masks
                - 'cam_224': [8, 12, 224, 224] - resized CAMs
                - 'cam_224_mask': [8, 12, 224, 224] - binary masks
        
        Args:
            inputs: Input images, shape [batch_size, C, H, W]
            cam: Class Activation Maps, shape [batch_size, num_classes, H, W]
            label: Binary labels, shape [batch_size, num_classes]
            model_conch: CONCH model instance
        
        Returns:
            Dictionary containing extracted features and masks, or None if no positive samples
        """
        # Early exit if no positive samples in the batch
        # Example: if all labels are 0, return None (no features to extract)
        if not torch.any(label == 1):
            return None
            
        # Resize CAMs to CONCH input size (224x224)
        # Example: cam [8, 12, 56, 56] -> cam_224 [8, 12, 224, 224]
        #          or cam [8, 12, 224, 224] -> cam_224 [8, 12, 224, 224] (no change)
        cam_224 = F.interpolate(cam, (self.conch_size, self.conch_size), 
                               mode="bilinear", align_corners=True)
       
        # Applies adaptive thresholding to get the binary mask
        # Each CAM gets its own threshold = max(CAM) * alpha
        # Example: If max(cam_224[0, 2]) = 0.8 and alpha=0.2, threshold = 0.16
        #          Values >= 0.16 become 1, others become 0
        # Output: cam_224_mask [8, 12, 224, 224] - binary masks (0 or 1)
        cam_224_mask = self.mask_adapter(cam_224)
        
        # Separates the image into FG and BG regions
        # This implements Equation 5: X_FG = b*M*X, X_BG = (1-b)*(1-M)*X
        # Example: If 2 samples have positive labels, returns:
        #          fg_features: [2, 3, 224, 224], bg_features: [2, 3, 224, 224]
        #          fg_masks: [2, 1, 224, 224], bg_masks: [2, 1, 224, 224]
        fg_features, bg_features, fg_masks, bg_masks = self.extract_features(
            inputs, cam_224, cam_224_mask, label
        )
        
        # Extracts features from these regions using CONCH
        # Applies masks to zero out irrelevant regions, then passes through encode_image
        # Example: [2, 3, 224, 224] -> [2, 512] (one feature vector per sample)
        fg_features, bg_features = self.get_masked_features(
            fg_features, bg_features, fg_masks, bg_masks, model_conch
        )
        
        # Return all extracted features and intermediate results
        return {
            'fg_features': fg_features,  # [N, 512] - foreground feature vectors
            'bg_features': bg_features,  # [N, 512] - background feature vectors
            'fg_masks': fg_masks,  # [N, 1, 224, 224] - foreground masks
            'bg_masks': bg_masks,  # [N, 1, 224, 224] - background masks
            'cam_224': cam_224,  # [batch_size, num_classes, 224, 224] - resized CAMs
            'cam_224_mask': cam_224_mask  # [batch_size, num_classes, 224, 224] - binary masks
        }

    def print_debug_info(self, batch_info):
        if batch_info is None:
            print("No foreground samples in batch")
            return
                
        print("\nFeature extraction debug info:")
        print(f"Number of foreground samples: {batch_info['fg_masks'].shape[0]}")
        print(f"Foreground features shape: {batch_info['fg_features'].shape}")
        print(f"Background features shape: {batch_info['bg_features'].shape}")
        print(f"CAM shape: {batch_info['cam_224'].shape}")
        print(f"Mask shape: {batch_info['cam_224_mask'].shape}")
        print(f"Foreground mask mean: {batch_info['fg_masks'].mean().item():.4f}")
        print(f"Background mask mean: {batch_info['bg_masks'].mean().item():.4f}")


def pair_features(fg_features, bg_features, text_features, labels):
    """
    Pair foreground/background image features with their corresponding text features.
    
    Args:
        fg_features: Foreground image features, shape [N, D]
        bg_features: Background image features, shape [N, D]
        text_features: Text features for all classes/subclasses, shape [num_classes, D]
        labels: Binary labels indicating which classes are present, shape [N, num_classes]
    
    Returns:
        Dictionary with paired features:
        - 'fg_features': [N, D] - foreground image features
        - 'bg_features': [N, D] - background image features
        - 'fg_text': [N, D] - foreground text features (for positive class)
        - 'bg_text': [N, num_other_classes, D] - background text features (for other classes)
    """
    batch_indices, class_indices = torch.where(labels == 1)
    
    paired_fg_features = [] 
    paired_bg_features = [] 
    paired_fg_text = [] 
    paired_bg_text = [] 

    for i in range(len(batch_indices)):
        curr_class = class_indices[i]
        
        curr_fg = fg_features[i]  # [D]
        curr_bg = bg_features[i]  # [D]
        
        curr_fg_text = text_features[curr_class]  # [D]
        
        bg_text_indices = [j for j in range(text_features.shape[0]) if j != curr_class]
        curr_bg_text = text_features[bg_text_indices]  # [num_other_classes, D]
        
        paired_fg_features.append(curr_fg)
        paired_bg_features.append(curr_bg)
        paired_fg_text.append(curr_fg_text)
        paired_bg_text.append(curr_bg_text)
    
    paired_fg_features = torch.stack(paired_fg_features)  # [N, D]
    paired_bg_features = torch.stack(paired_bg_features)  # [N, D]
    paired_fg_text = torch.stack(paired_fg_text)         # [N, D]
    paired_bg_text = torch.stack(paired_bg_text)         # [N, num_other_classes, D]
    
    return {
        'fg_features': paired_fg_features,  
        'bg_features': paired_bg_features,  
        'fg_text': paired_fg_text,         
        'bg_text': paired_bg_text       
    }


def merge_to_parent_predictions(predictions, k_list, method='max'):
    """
    Merge subclass predictions to parent class predictions.
    
    Args:
        predictions: Subclass predictions, shape [batch_size, total_subclasses]
        k_list: List of number of subclasses per parent class, e.g., [10, 10, 10, 10]
        method: 'max' or 'mean' for merging strategy
    
    Returns:
        Parent class predictions, shape [batch_size, num_parent_classes]
    """
    parent_preds = []
    start_idx = 0
    
    for k in k_list:
        if k > 1:
            class_preds = predictions[:, start_idx:start_idx + k]
            
            if method == 'max':
                class_probs = torch.softmax(class_preds, dim=1)
                parent_pred = (class_probs * class_preds).sum(dim=1)
            else:  # method == 'mean'
                parent_pred = torch.mean(class_preds, dim=1)
            
            parent_preds.append(parent_pred)
        else:
            parent_preds.append(predictions[:, start_idx])
        
        start_idx += k
    parent_preds = torch.stack(parent_preds, dim=1)
    
    return parent_preds


def merge_subclass_cams_to_parent(cams, k_list, method='max'):
    """
    Merge subclass CAMs to parent class CAMs.
    
    Args:
        cams: Subclass CAMs, shape [batch_size, total_subclasses, H, W]
        k_list: List of number of subclasses per parent class
        method: 'max' or 'mean' for merging strategy
    
    Returns:
        Parent class CAMs, shape [batch_size, num_parent_classes, H, W]
    """
    batch_size, _, H, W = cams.shape
    num_parent_classes = len(k_list)

    parent_cams = torch.zeros(batch_size, num_parent_classes, H, W, 
                            device=cams.device, dtype=cams.dtype)
    
    start_idx = 0
    for parent_idx, k in enumerate(k_list):
        if k > 1: 
            subclass_cams = cams[:, start_idx:start_idx + k, :, :]
            
            if method == 'max':
                B, k, H, W = subclass_cams.shape
                cams_flat = subclass_cams.view(B, k, H*W)  # [B, k, H*W]
                cams_probs = torch.softmax(cams_flat, dim=1)  # [B, k, H*W]
                parent_cam_flat = (cams_probs * cams_flat).sum(dim=1)  # [B, H*W]
                parent_cam = parent_cam_flat.view(B, H, W)  # [B, H, W]
            else:  # method == 'mean'
                parent_cam = torch.mean(subclass_cams, dim=1)    # [B, H, W]
            
            parent_cams[:, parent_idx, :, :] = parent_cam
        else:
            parent_cams[:, parent_idx, :, :] = cams[:, start_idx, :, :]
        
        start_idx += k
    
    return parent_cams


def expand_parent_to_subclass_labels(parent_labels, k_list):
    """
    Expand parent class labels to subclass labels.
    
    Args:
        parent_labels: Parent class labels, shape [batch_size, num_parent_classes]
        k_list: List of number of subclasses per parent class
    
    Returns:
        Subclass labels, shape [batch_size, total_subclasses]
    """
    batch_size = parent_labels.size(0)
    total_subclasses = sum(k_list)
    
    subclass_labels = torch.zeros(batch_size, total_subclasses, 
                                device=parent_labels.device, 
                                dtype=parent_labels.dtype)
    start_idx = 0
    for parent_idx, k in enumerate(k_list):
        parent_label = parent_labels[:, parent_idx:parent_idx+1]  # [batch_size, 1]
        subclass_labels[:, start_idx:start_idx+k] = parent_label.repeat(1, k)
        start_idx += k
    
    return subclass_labels
