import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextLocalMixing(nn.Module):
    """
    Text local mixing for constructing scale-specific textual contexts.

    Input:
        x: [B, L, D]

    Output:
        out: [B, L, D]
    """
    def __init__(self, dim: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()

        padding = kernel_size // 2

        self.depthwise = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim
        )

        self.pointwise = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1
        )

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        x = x.transpose(1, 2)       # [B, D, L]
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)       # [B, L, D]
        return x


class ImageLocalMixing(nn.Module):
    """
    Image local mixing for constructing scale-specific visual contexts.

    Input:
        x: [B, L, D], where L = H * W

    Output:
        out: [B, L, D]
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        grid_size=(14, 14),
        dropout: float = 0.1
    ):
        super().__init__()

        self.grid_size = grid_size
        padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim
        )

        self.pointwise = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1
        )

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        H, W = self.grid_size

        if L != H * W:
            raise ValueError(
                f"Image token length L={L} does not match grid_size={self.grid_size}. "
                f"Expected L={H * W}."
            )

        x = x.transpose(1, 2).contiguous()  # [B, D, L]
        x = x.view(B, D, H, W)              # [B, D, H, W]

        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.act(x)
        x = self.dropout(x)

        x = x.flatten(2).transpose(1, 2)    # [B, L, D]
        return x


class ScaleSpecificEvidenceHead(nn.Module):
    """
    Estimate scale-specific evidence activation.

    Input:
        x: [B, L, D]

    Output:
        e: [B, L, 1]
    """
    def __init__(self, dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class MultiScaleEvidenceAttentionBlock(nn.Module):
    """
    MSEA Block: Multi-scale Evidence Attention with shared Q/K/V projections.

    This block is a single-modality module.
    It can be used for text or image depending on local_mixing_type.

    Input:
        x: [B, L, D]

    Output:
        x_e: [B, L, D]
    """
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        scales=(1, 3, 5),
        local_mixing_type: str = "text",
        image_grid_size=(14, 14),
        evidence_hidden_dim: int = 256,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        local_dropout: float = 0.1,
        tau: float = 1.0,
        eps: float = 1e-6
    ):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads."
        assert local_mixing_type in ["text", "image"]

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scales = scales
        self.tau = tau
        self.eps = eps
        self.local_mixing_type = local_mixing_type

        # ---------------------------------------------------------
        # 1. Scale-specific local mixing operators
        # ---------------------------------------------------------
        if local_mixing_type == "text":
            self.local_mixers = nn.ModuleList([
                TextLocalMixing(
                    dim=dim,
                    kernel_size=s,
                    dropout=local_dropout
                )
                for s in scales
            ])
        else:
            self.local_mixers = nn.ModuleList([
                ImageLocalMixing(
                    dim=dim,
                    kernel_size=s,
                    grid_size=image_grid_size,
                    dropout=local_dropout
                )
                for s in scales
            ])

        self.scale_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in scales
        ])

        # ---------------------------------------------------------
        # 2. Shared Q/K/V projections across all scales
        # ---------------------------------------------------------
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(attn_dropout)

        self.out_proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # ---------------------------------------------------------
        # 3. Scale-specific evidence activation head
        # Shared across scales
        # ---------------------------------------------------------
        self.evidence_head = ScaleSpecificEvidenceHead(
            dim=dim,
            hidden_dim=evidence_hidden_dim,
            dropout=proj_dropout
        )

        # ---------------------------------------------------------
        # 4. Learnable scale weights
        # ---------------------------------------------------------
        self.scale_logits = nn.Parameter(torch.zeros(len(scales)))

        # ---------------------------------------------------------
        # 5. Output normalization
        # ---------------------------------------------------------
        self.out_norm = nn.LayerNorm(dim)

    def _reshape_to_heads(self, x):
        """
        x: [B, L, D]
        return: [B, H, L, Dh]
        """
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def _merge_heads(self, x):
        """
        x: [B, H, L, Dh]
        return: [B, L, D]
        """
        B, H, L, Dh = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, L, H * Dh)
        return x

    def _shared_attention(self, x, mask=None):
        """
        Shared multi-head self-attention for one scale.

        Args:
            x: [B, L, D]
            mask: [B, L], optional. 1 for valid tokens, 0 for padding.

        Returns:
            out: [B, L, D]
        """
        Q = self._reshape_to_heads(self.q_proj(x))  # [B, H, L, Dh]
        K = self._reshape_to_heads(self.k_proj(x))  # [B, H, L, Dh]
        V = self._reshape_to_heads(self.v_proj(x))  # [B, H, L, Dh]

        attn_logits = torch.matmul(Q, K.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(self.head_dim)  # [B, H, L, L]

        if mask is not None:
            # mask: [B, L] -> [B, 1, 1, L]
            mask = mask.unsqueeze(1).unsqueeze(2).bool()
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)       # [B, H, L, Dh]
        out = self._merge_heads(out)      # [B, L, D]
        out = self.out_proj(out)
        out = self.proj_dropout(out)

        return out

    def _weighted_sum(self, tensors, weights):
        """
        tensors: list of tensors with the same shape
        weights: [S]
        """
        out = 0.0
        for w, x in zip(weights, tensors):
            out = out + w * x
        return out

    def forward(self, x, mask=None, return_aux=False):
        """
        Args:
            x: [B, L, D]
            mask: [B, L], optional
            return_aux: whether to return intermediate variables

        Returns:
            x_e: [B, L, D]
            aux_info: optional dict
        """
        scale_outputs = []
        scale_evidences = []
        scale_contexts = []

        # =========================================================
        # 1. Multi-scale contextual representation
        # 2. Shared Q/K/V attention for each scale
        # 3. Scale-specific evidence activation
        # =========================================================
        for mixer, ln in zip(self.local_mixers, self.scale_norms):
            local_context = mixer(x)                 # [B, L, D]
            x_s = ln(x + local_context)              # [B, L, D]

            o_s = self._shared_attention(x_s, mask)  # [B, L, D]
            e_s = self.evidence_head(o_s)            # [B, L, 1]

            scale_contexts.append(x_s)
            scale_outputs.append(o_s)
            scale_evidences.append(e_s)

        # =========================================================
        # 4. Learnable normalized scale weights
        # =========================================================
        alpha = F.softmax(self.scale_logits, dim=0)  # [S]

        # =========================================================
        # 5. Cross-scale evidence mean
        # =========================================================
        mu = self._weighted_sum(scale_evidences, alpha)  # [B, L, 1]

        # =========================================================
        # 6. Cross-scale evidence fluctuation
        # =========================================================
        delta = 0.0
        for w, e_s in zip(alpha, scale_evidences):
            delta = delta + w * (e_s - mu).pow(2)        # [B, L, 1]

        # =========================================================
        # 7. Cross-scale evidence stability
        # =========================================================
        stability = torch.exp(-delta / (self.tau + self.eps))  # [B, L, 1]

        # =========================================================
        # 8. Stable evidence gate
        # =========================================================
        gate = mu * stability  # [B, L, 1]

        # =========================================================
        # 9. Multi-scale attention output aggregation
        # =========================================================
        o_bar = self._weighted_sum(scale_outputs, alpha)  # [B, L, D]

        # =========================================================
        # 10. Residual evidence enhancement
        # =========================================================
        x_e = self.out_norm(x + gate * o_bar)  # [B, L, D]

        if not return_aux:
            return x_e

        aux_info = {
            "scale_weights": alpha.detach(),          # [S]
            "mean_evidence": mu,                      # [B, L, 1]
            "evidence_fluctuation": delta,            # [B, L, 1]
            "evidence_stability": stability,          # [B, L, 1]
            "evidence_gate": gate,                    # [B, L, 1]
            "scale_outputs": scale_outputs,           # list of [B, L, D]
            "scale_evidences": scale_evidences,       # list of [B, L, 1]
        }

        return x_e, aux_info

    def stability_regularization(self, aux_info):
        """
        Optional stability regularization:
        L_stab = Mean(mu * delta)
        """
        return torch.mean(
            aux_info["mean_evidence"] * aux_info["evidence_fluctuation"]
        )

class MultiScaleEvidenceAttentionModule(nn.Module):
    """
    MSEA: Multi-scale Evidence Attention Module.

    This module replaces the original first module.
    It takes BERT text features and ViT image patch features as input,
    and outputs stabilized textual and visual evidence representations.

    Inputs:
        text_feat:  [B, Lt, D], e.g., [B, 32, 768]
        image_feat: [B, Lv, D], e.g., [B, 196, 768]

    Outputs:
        text_evidence:  [B, Lt, D]
        image_evidence: [B, Lv, D]
    """
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        text_scales=(1, 3, 5),
        image_scales=(1, 3, 5),
        image_grid_size=(14, 14),
        evidence_hidden_dim: int = 256,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        local_dropout: float = 0.1,
        tau_text: float = 1.0,
        tau_image: float = 1.0,
        return_aux_default: bool = False
    ):
        super().__init__()

        self.return_aux_default = return_aux_default

        self.text_msea = MultiScaleEvidenceAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            scales=text_scales,
            local_mixing_type="text",
            image_grid_size=image_grid_size,
            evidence_hidden_dim=evidence_hidden_dim,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            local_dropout=local_dropout,
            tau=tau_text
        )

        self.image_msea = MultiScaleEvidenceAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            scales=image_scales,
            local_mixing_type="image",
            image_grid_size=image_grid_size,
            evidence_hidden_dim=evidence_hidden_dim,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            local_dropout=local_dropout,
            tau=tau_image
        )

    def forward(
        self,
        text_feat,
        image_feat,
        text_mask=None,
        image_mask=None,
        return_aux=None
    ):
        """
        Args:
            text_feat:  [B, Lt, D]
            image_feat: [B, Lv, D]
            text_mask:  [B, Lt], optional
            image_mask: [B, Lv], optional
            return_aux: whether to return auxiliary variables

        Returns:
            text_evidence:  [B, Lt, D]
            image_evidence: [B, Lv, D]
            aux_info: optional dict
        """
        if return_aux is None:
            return_aux = self.return_aux_default

        if return_aux:
            text_evidence, text_aux = self.text_msea(
                text_feat,
                mask=text_mask,
                return_aux=True
            )

            image_evidence, image_aux = self.image_msea(
                image_feat,
                mask=image_mask,
                return_aux=True
            )

            aux_info = {
                "text_aux": text_aux,
                "image_aux": image_aux
            }

            return text_evidence, image_evidence, aux_info

        text_evidence = self.text_msea(
            text_feat,
            mask=text_mask,
            return_aux=False
        )

        image_evidence = self.image_msea(
            image_feat,
            mask=image_mask,
            return_aux=False
        )

        return text_evidence, image_evidence

    def stability_regularization(self, aux_info):
        """
        Optional stability regularization for both modalities.
        """
        loss_text = self.text_msea.stability_regularization(aux_info["text_aux"])
        loss_image = self.image_msea.stability_regularization(aux_info["image_aux"])
        return loss_text + loss_image