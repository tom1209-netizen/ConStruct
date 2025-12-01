import pickle as pkl
from functools import partial
import os
import sys  
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from model.segform import mix_transformer

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_dir)

from model.ver2.text_encoder import PromptLearner
from model.ver2.image_encoder import CONCHFeatureExtractor 
 
class AdaptiveLayer(nn.Module):
    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ClsNetwork(nn.Module):
    def __init__(self,
        model_conch = None, 
        cls_num_classes=4,
        num_prototypes_per_class=None,
        prototype_feature_dim=512,
        conch_img_feat_dim = 768, 
        stride=[4, 2, 2, 1],
        pretrained=True,
        n_ratio=0.5, 
        class_prompts = None, 
        device = None, 
                 
    ):
        super().__init__()
        self.device = device 
        self.cls_num_classes = cls_num_classes
        self.class_prompts = class_prompts
        # self.num_prototypes_per_class = num_prototypes_per_class
        self.k_list = [len(prompts) for cls, prompts in class_prompts.items()] 
        self.total_prototypes = sum(self.k_list)
        self.stride = stride
        
        self.device = device 
        # Backbone Encoder (Same as original)
        self.image_encoder = CONCHFeatureExtractor(model_conch, device="cuda")   
        
        self.prompt_learner =  PromptLearner(
                model_conch=model_conch,
                n_ctx=16,
                ctx_init="a histopathology image of a"  # ← strong initialization
            ).to(self.device)
        
        # with torch.no_grad():
        #     init_protos = self.prompt_learner.get_prototypes(class_prompts)['features'] 
            
        # self.register_buffer('prototypes', init_protos.clone()) 
        # print(f"Initialized {self.prototypes.shape[0]} prototypes") 
        # === FIXED: Prototypes are now a learnable nn.Parameter ===
        with torch.no_grad():
            init_protos = self.prompt_learner.get_prototypes(class_prompts)['features']  # [total_prototypes, 512]

        # This is now a proper learnable parameter → gradients flow correctly
        self.prototypes = nn.Parameter(init_protos.clone(), requires_grad=True)
        print(f"Initialized {self.prototypes.shape[0]} LEARNABLE prototypes as nn.Parameter") 
        
        self.l_fc1 = AdaptiveLayer(prototype_feature_dim, n_ratio, conch_img_feat_dim)
        self.l_fc2 = AdaptiveLayer(prototype_feature_dim, n_ratio, conch_img_feat_dim)
        self.l_fc3 = AdaptiveLayer(prototype_feature_dim, n_ratio, conch_img_feat_dim)
        self.l_fc4 = AdaptiveLayer(prototype_feature_dim, n_ratio, conch_img_feat_dim)

        # Other components from the original model are kept for compatibility
        self.pooling = F.adaptive_avg_pool2d
       
        # The learnable temperature parameters for cosine similarity are kept.
        self.logit_scales = nn.Parameter(torch.ones(5) * (1 / 0.07)) 


    def get_param_groups(self):
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    def forward(self, batch_of_images):
        _token_embedding_out = self.image_encoder(batch_of_images) 
        conch_features = self.image_encoder.reshape_to_spatial(_token_embedding_out) 
        feats = [
            conch_features['block_2'],
            conch_features['block_5'],
            conch_features['block_8'],
            conch_features['block_11'],
            conch_features['final_norm'],
        ]
        # print("final_norm", conch_features['final_norm'].shape)
        # ← PERFECT: keep prototypes fresh and trainable
        # current_protos = self.prompt_learner.get_prototypes(self.class_prompts)['features']
        # self.prototypes.copy_(current_protos)
        # prototypes = self.prototypes  # (N, 512)
# === FIXED: Keep prototypes up-to-date with prompt_learner WITHOUT breaking the graph ===
        # We do this safely by detaching only the data copy (not the computation graph of the prompt_learner)
        with torch.no_grad():
            fresh_protos = self.prompt_learner.get_prototypes(self.class_prompts)['features']
        self.prototypes.data.copy_(fresh_protos)  # Only update the values, no graph connection

        # Use the learnable parameter (which has gradients)
        prototypes = self.prototypes  # [total_prototypes, 512] 
        # Project to 768
        proj_heads = [self.l_fc1, self.l_fc2, self.l_fc3, self.l_fc4, self.l_fc4]
        projected_ps = [head(prototypes) for head in proj_heads]
        scales = self.logit_scales 
    
        all_cls = []
        all_cam = []
    
        for i, f in enumerate(feats):
            B, C, H, W = f.shape
            x_flat = f.permute(0, 2, 3, 1).reshape(-1, C)
    
            x_norm = F.normalize(x_flat, dim=-1)
            p_norm = F.normalize(projected_ps[i], dim=-1)
    
            logits = scales[i] * (x_norm @ p_norm.t())
            cam = logits.view(B, H, W, -1).permute(0, 3, 1, 2)
            cls_score = cam.mean(dim=[2, 3])
    
            all_cls.append(cls_score)
            all_cam.append(cam.detach() if i < 4 else cam)  # keep grad on last
        
        
 
        return (
            all_cls[0], all_cam[0],
            all_cls[1], all_cam[1],
            all_cls[2], all_cam[2],
            all_cls[3], all_cam[3],
            all_cls[4], all_cam[4],  # final with grad
            self.prototypes,
            self.k_list,
            feats[-1]
        ) 
