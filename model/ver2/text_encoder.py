import torch
import torch.nn as nn
from conch.open_clip_custom import get_tokenizer

_tokenizer = get_tokenizer()


class PromptLearner(nn.Module):
    """
    Pure CoOp-style Prompt Learner for CONCH
    - Only learns a shared prompt: "X X ... X" (n_ctx tokens)
    - No class names stored → lightweight & flexible
    - Designed to generate rich text prototypes from any descriptions
    """
    def __init__(
        self,
        model_conch,
        n_ctx: int = 16,
        ctx_init: str = "a histopathology image of a",   # ← recommended
        device=None
    ):
        super().__init__()
        self.model_conch = model_conch
        self.n_ctx = n_ctx
        self.device = device or next(model_conch.parameters()).device
        self.dtype = torch.float32
        self.ctx_dim = model_conch.text.text_projection.shape[0]  # auto-detect (512 or 768)

        # --------------------- Learnable Context Tokens ---------------------
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            tokens = torch.tensor(_tokenizer.encode(ctx_init), device=self.device, dtype=torch.long)
            with torch.no_grad():
                emb = model_conch.text.token_embedding(tokens.unsqueeze(0)).type(self.dtype)
            ctx_vectors = emb[0, 1:1 + n_ctx]  # skip <SOT>
            print(f"[PromptLearner] Initialized with: '{ctx_init}'")
        else:
            ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype, device=self.device)
            nn.init.normal_(ctx_vectors, std=0.02)
            print(f"[PromptLearner] Randomly initialized {n_ctx} tokens")

        self.ctx = nn.Parameter(ctx_vectors)  # ← this is all we learn!

    def get_prototypes(self, class_prompts: dict[int, list[str]]) -> dict:
        """
        Generate text embeddings for all descriptions in class_prompts
        Using the learned prompt + each description
        """
        features, texts, labels = [], [], []

        # Expand shared ctx to batch
        ctx = self.ctx.unsqueeze(0)  # (1, n_ctx, D)

        for cls_idx, descriptions in class_prompts.items():
            for desc in descriptions:
                # Tokenize description
                tokens = _tokenizer.encode(desc)
                tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device)
                desc_len = len(tokens) - 1  # exclude <SOT>

                # Pad & embed
                padded = torch.nn.functional.pad(tokens_tensor, (0, 77 - len(tokens)))
                padded = padded.unsqueeze(0)  # (1, 77)
                with torch.no_grad():
                    suffix_emb = self.model_conch.text.token_embedding(padded).type(self.dtype)  # (1, 77, D)

                # Build full sequence: [X X ... X] + [description tokens]
                full_seq = torch.cat([
                    ctx,                                 # (1, n_ctx, D)
                    suffix_emb[:, 1:, :]                 # (1, 76, D) → skip <SOT> of description
                ], dim=1)  # (1, n_ctx + 76, D)

                # Forward through text tower
                x = full_seq + self.model_conch.text.positional_embedding[:full_seq.shape[1]]
                x = x.permute(1, 0, 2)      # (L, 1, D)
                x = self.model_conch.text.transformer(x)
                x = x.permute(1, 0, 2)      # (1, L, D)

                # EOT position = n_ctx + desc_len
                eot_pos = self.n_ctx + desc_len
                feat = x[:, eot_pos, :]

                # Final projection
                feat = self.model_conch.text.ln_final(feat)
                feat = feat @ self.model_conch.text.text_projection
                feat = feat / feat.norm(dim=-1, keepdim=True)

                features.append(feat)
                texts.append(desc)
                labels.append(cls_idx)

        features = torch.cat(features, dim=0)  # (N, D)

        return {
            'features': features,
            'texts': texts,
            'labels': torch.tensor(labels, device=self.device),
        }

# ========================================================

# if __name__ == "__main__":
#     # Create prompt learner (no classnames needed!)
#     prompt_learner = PromptLearner(
#         model_conch=model_conch,
#         n_ctx=16,
#         ctx_init="a histopathology image of a"  # ← strong initialization
#     ).cuda()

#     # Your multi-description dictionary
#     class_prompts = {
#         0: ["invasive carcinoma", "malignant tumor cells", "invasive cancer"],
#         1: ["benign tissue", "fibrous stroma", "non-malignant tissue"],
#         2: ["immune infiltrate", "lymphocytes", "inflammatory cells"],
#         3: ["necrosis", "dead tissue", "necrotic area"]
#     }

#     # Generate prototypes
#     result = prompt_learner.get_prototypes(class_prompts)

#     print(f"\nPromptLearner ready with {prompt_learner.n_ctx} learnable tokens")
#     print(f"Generated {result['features'].shape[0]} prototype embeddings")
#     print(f"Shape: {result['features'].shape}")
#     print(f"Norm: {result['features'].norm(dim=-1)[:5]}")
#     print("\nPrototypes:")
#     for i, (label, text) in enumerate(zip(result['labels'].tolist(), result['texts'])):
#         print(f"  {label} | {text}")   