import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_jeffreys_similarity(distributions):
    """Calculates Jeffrey's Similarity for a list of distributions."""
    num_distributions = len(distributions)
    if num_distributions <= 1:
        # If there's only one prototype for a class, diversity is undefined (or perfect).
        return torch.tensor(0.0, device=distributions[0].device)

    total_similarity = 0.0
    num_pairs = 0

    # Ensure distributions are valid (sum to 1 and are non-negative)
    distributions = [F.softmax(d, dim=0) for d in distributions]

    for i in range(num_distributions):
        for j in range(i + 1, num_distributions):
            U = distributions[i]
            V = distributions[j]
            
            # Add a small epsilon to avoid log(0) issues in KL-divergence
            eps = 1e-10
            U_safe = U + eps
            V_safe = V + eps
            
            # Kullback-Leibler Divergence
            dkl_uv = F.kl_div(U_safe.log(), V_safe, reduction='sum')
            dkl_vu = F.kl_div(V_safe.log(), U_safe, reduction='sum')
            
            # Jeffrey's Divergence (symmetrized KL)
            jeffreys_divergence = dkl_uv + dkl_vu
            
            # Jeffrey's Similarity
            similarity = torch.exp(-jeffreys_divergence)
            total_similarity += similarity
            num_pairs += 1
            
    return total_similarity / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=distributions[0].device)

class PrototypeDiversityRegularizer(nn.Module):
    """
    Stronger diversity regularizer that combines:
    1) Prototype repulsion (pairwise cosine margin within a class).
    2) Peaky assignments per pixel (low entropy per pixel).
    3) Balanced prototype usage within a class (mass close to uniform).
    """
    def __init__(
        self,
        num_prototypes_per_class,
        omega_window=7,
        omega_min_mass=0.05,
        temperature=0.07,
        sharpness_weight=0.1,
        coverage_weight=0.1,
        repulsion_weight=0.5,
        repulsion_margin=0.2,
        jeffreys_weight=0.0,
        pool_size=None,
        debug=False,
        debug_every=200,
    ):
        super().__init__()
        self.num_prototypes_per_class = num_prototypes_per_class
        self.omega_min_mass = omega_min_mass
        self.temperature = temperature
        self.sharpness_weight = sharpness_weight
        self.coverage_weight = coverage_weight
        self.repulsion_weight = repulsion_weight
        self.repulsion_margin = repulsion_margin
        self.jeffreys_weight = jeffreys_weight
        self.pool_size = pool_size
        self.debug = debug
        self.debug_every = debug_every

    def _repulsion(self, prototypes):
        # prototypes: [k, d], already normalized
        sim = prototypes @ prototypes.t()  # [k, k]
        eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
        sim_off = sim * (1 - eye)
        penalty = F.relu(sim_off - self.repulsion_margin).pow(2)
        return penalty.mean()

    def forward(self, feature_map, prototypes, gt_mask, global_step=None):
        """
        feature_map: [B, D, H, W] (we use the deepest feature map)
        prototypes: [total_prototypes, D]
        gt_mask: [B, H, W] pseudo or ground-truth mask
        """
        if self.pool_size is not None:
            feature_map = F.adaptive_avg_pool2d(feature_map, output_size=(self.pool_size, self.pool_size))
            gt_mask = F.interpolate(gt_mask.unsqueeze(1).float(), size=(self.pool_size, self.pool_size), mode="nearest").squeeze(1).long()

        B, D, H, W = feature_map.shape
        total_prototypes = prototypes.shape[0]
        num_classes = total_prototypes // self.num_prototypes_per_class

        feature_flat = feature_map.permute(0, 2, 3, 1).reshape(B, -1, D)  # [B, HW, D]
        mask_flat = gt_mask.view(B, -1)  # [B, HW]

        loss_batch = []
        eps = 1e-8

        for b in range(B):
            class_losses = []
            for c in range(num_classes):
                idxs = (mask_flat[b] == c).nonzero(as_tuple=True)[0]
                if idxs.numel() == 0:
                    continue
                feats = feature_flat[b, idxs]  # [N, D]
                feats = F.normalize(feats, dim=-1)

                start = c * self.num_prototypes_per_class
                end = start + self.num_prototypes_per_class
                proto = prototypes[start:end]  # [k, D]
                proto = F.normalize(proto, dim=-1)

                # Assignment over prototypes for this class
                logits = feats @ proto.t() / max(self.temperature, 1e-4)
                assign = F.softmax(logits, dim=1)  # [N, k]
                proto_logits = logits  # [N, k]

                # (1) Repulsion between prototypes
                repulsion = self._repulsion(proto)

                # (2) Sharpness: encourage per-pixel assignment to be peaky
                entropy = -(assign * (assign + eps).log()).sum(dim=1).mean()

                # (3) Coverage: encourage prototype usage to be balanced
                mass = assign.sum(dim=0)  # [k]
                mass = mass / (mass.sum() + eps)
                target = torch.full_like(mass, 1.0 / self.num_prototypes_per_class)
                coverage = F.mse_loss(mass, target)

                # Penalize collapsed prototypes (very low mass)
                mass_floor = F.relu(self.omega_min_mass - mass).mean()

                # (4) Jeffrey's similarity on per-prototype activation distributions 
                jeff = torch.tensor(0.0, device=feature_map.device)
                if self.jeffreys_weight > 0:
                    # Build per-prototype spatial distributions: softmax over pixels
                    proto_logits_norm = proto_logits - proto_logits.max(dim=0, keepdim=True).values
                    distributions_v = [
                        F.softmax(proto_logits_norm[:, i], dim=0)
                        for i in range(proto_logits_norm.shape[1])
                    ]
                    jeff = calculate_jeffreys_similarity(distributions_v)

                class_loss = (
                    self.repulsion_weight * repulsion +
                    self.sharpness_weight * entropy +
                    self.coverage_weight * coverage +
                    mass_floor +
                    self.jeffreys_weight * jeff
                )
                class_losses.append(class_loss)

            if class_losses:
                loss_batch.append(torch.stack(class_losses).mean())

        if not loss_batch:
            return torch.tensor(0.0, device=feature_map.device)

        loss = torch.stack(loss_batch).mean()

        if self.debug and global_step is not None and global_step % self.debug_every == 0:
            print(f"[Diversity] step {global_step} loss {loss.item():.4f}")
        return loss
