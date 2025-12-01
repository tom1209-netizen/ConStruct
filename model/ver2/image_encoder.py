import torch

class CONCHFeatureExtractor:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.spatial_size = 14  # 14x14 = 196 patches in CONCH ViT-L/14

        # Remove any old hooks
        for m in self.model.modules():
            m._forward_hooks.clear()
            m._forward_pre_hooks.clear()

        # Storage
        self.feat = {}

        def make_hook(name):
            def hook(module, input, output):
                self.feat[name] = output.detach()
            return hook

        # Tap pyramid levels
        blocks = self.model.visual.trunk.blocks
        for i in [2, 5, 8, 11]:
            blocks[i].register_forward_hook(make_hook(f'block_{i}'))

        # Absolute last layer after LayerNorm (strongest patch features)
        self.model.visual.trunk.norm.register_forward_hook(make_hook('final_norm'))

        print("CONCHFeatureExtractor ready: block_2, block_5, block_8, block_11, final_norm + 512-dim CLS")

    @torch.no_grad()
    def __call__(self, images: torch.Tensor):
        """
        Input: images -> (B, 3, 224, 224) or (3, 224, 224)
        Output: dict with:
                'cls': (B, 512) ← global image embedding
                'block_2': (B, 196, 768)
                'block_5': (B, 196, 768)
                ...
                'final_norm': (B, 196, 768) ← best patch features
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)  # add batch dim
        images = images.to(self.device)
        self.feat.clear()

        # This triggers all hooks + returns the final 512-dim embedding
        cls_embedding = self.model.encode_image(images)  # (B, 512)

        # Extract patch tokens (drop CLS token)
        patch_features = {k: v[:, 1:, :] for k, v in self.feat.items()}
        
        # Return original flat format
        return {
            'cls': cls_embedding,           # (B, 512)
            **patch_features                # (B, 196, 768) each
        }

    def reshape_to_spatial(self, features_dict):
        """
        Convert all (B, 196, 768) → (B, 768, 14, 14)
        Usage:
            out = extractor(rgb_images)
            spatial_out = extractor.reshape_to_spatial(out)
        """
        spatial = {}
        for k, v in features_dict.items():
            if k == 'cls':
                spatial[k] = v
            else:
                # v: (B, 196, 768)
                B, N, C = v.shape
                assert N == 196, f"Expected 196 patches, got {N}"
                x = v.reshape(B, self.spatial_size, self.spatial_size, C)   # (B, 14, 14, 768)
                x = x.permute(0, 3, 1, 2)                                   # (B, 768, 14, 14)
                spatial[k] = x.contiguous()
        return spatial


# ========================================
# USAGE (unchanged + optional spatial view)
# ========================================
# if __name__ == "__main__":
#     # Create once
#     conch_extractor = CONCHFeatureExtractor(model_conch, device="cuda")

#     # Normal usage (exactly like before)
#     out = conch_extractor(rgb_images)
#     print(out['cls'].shape)         # → torch.Size([B, 512])
#     print(out['final_norm'].shape)  # → torch.Size([B, 196, 768])
#     print(out.keys())

#     # When you want CNN-style 14×14 maps → just call:
#     spatial_out = conch_extractor.reshape_to_spatial(out)

#     print("\nSpatial view:")
#     for k, v in spatial_out.items():
#         if k != 'cls':
#             print(f"  {k:12s} → {v.shape}")  # → torch.Size([B, 768, 14, 14]) 
     