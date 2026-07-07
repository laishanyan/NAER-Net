import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, dropout=0.1, use_ln=False):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        ]
        if use_ln:
            layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EvidenceAwareAttentionShiftBlock(nn.Module):
    """
    EASM block for one modality.

    It performs:
    1. shared Q/K/V projection;
    2. semantic self-attention aggregation;
    3. multi-scale dynamic evidence-shift aggregation;
    4. cross-scale evidence stability estimation;
    5. attention-shift adaptive mixing;
    6. residual + FFN output.

    Input:
        x: [B, L, D]

    Output:
        x_e: [B, L, D]
    """

    def __init__(
        self,
        dim=768,
        num_heads=8,
        scales=(1, 3, 5),
        mode="text",
        image_grid_size=(14, 14),
        shift_hidden_dim=128,
        evidence_hidden_dim=256,
        mix_hidden_dim=256,
        ffn_hidden_dim=1024,
        attn_dropout=0.1,
        proj_dropout=0.1,
        ffn_dropout=0.1,
        tau=1.0,
        eps=1e-6,
    ):
        super().__init__()

        assert mode in ["text", "image"]
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scales = scales
        self.mode = mode
        self.image_grid_size = image_grid_size
        self.tau = tau
        self.eps = eps

        # Shared Stage-I projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # Dynamic evidence-shift weight generator
        # input: [q_i; k_j; |q_i-k_j|; q_i*k_j]
        self.shift_score = MLP(
            in_dim=4 * dim,
            out_dim=1,
            hidden_dim=shift_hidden_dim,
            dropout=proj_dropout,
            use_ln=False,
        )

        # Scale-specific evidence activation
        # input: [O_att; O_shift^s; |O_att-O_shift^s|]
        self.evidence_head = MLP(
            in_dim=3 * dim,
            out_dim=1,
            hidden_dim=evidence_hidden_dim,
            dropout=proj_dropout,
            use_ln=False,
        )

        # Attention-shift adaptive mixing gate
        # input: [O_att; O_shift_bar; |O_att-O_shift_bar|]
        self.mix_gate = MLP(
            in_dim=3 * dim,
            out_dim=1,
            hidden_dim=mix_hidden_dim,
            dropout=proj_dropout,
            use_ln=False,
        )

        # Learnable scale weights
        self.scale_logits = nn.Parameter(torch.zeros(len(scales)))

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden_dim, dim),
            nn.Dropout(ffn_dropout),
        )

    def _reshape_heads(self, x):
        # [B, L, D] -> [B, H, L, Dh]
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        # [B, H, L, Dh] -> [B, L, D]
        B, H, L, Dh = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, L, H * Dh)

    def _semantic_attention(self, q, k, v, mask=None):
        """
        q, k, v: [B, L, D]
        mask: [B, L], 1 for valid tokens, 0 for padding
        """
        Q = self._reshape_heads(q)
        K = self._reshape_heads(k)
        V = self._reshape_heads(v)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(self.head_dim)

        if mask is not None:
            key_mask = mask.unsqueeze(1).unsqueeze(2).bool()
            attn_logits = attn_logits.masked_fill(~key_mask, float("-inf"))

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        out = self.proj_dropout(out)

        return out

    def _get_text_neighbors(self, x, scale, mask=None):
        """
        Get 1D local neighbors for text.

        x: [B, L, D]
        return:
            neigh: [B, L, K, D], K=scale
            neigh_mask: [B, L, K]
        """
        B, L, D = x.shape
        pad = scale // 2

        x_ = x.transpose(1, 2)               # [B, D, L]
        x_ = F.pad(x_, (pad, pad))           # [B, D, L+2p]
        neigh = x_.unfold(2, scale, 1)       # [B, D, L, K]
        neigh = neigh.permute(0, 2, 3, 1)    # [B, L, K, D]

        if mask is None:
            neigh_mask = torch.ones(B, L, scale, device=x.device, dtype=torch.bool)
        else:
            m = mask.float().unsqueeze(1)    # [B, 1, L]
            m = F.pad(m, (pad, pad), value=0)
            neigh_mask = m.unfold(2, scale, 1).squeeze(1) > 0.5  # [B, L, K]

        return neigh, neigh_mask

    def _get_image_neighbors(self, x, scale, mask=None):
        """
        Get 2D local neighbors for image patches.

        x: [B, L, D], L = H * W
        return:
            neigh: [B, L, K, D], K=scale*scale
            neigh_mask: [B, L, K]
        """
        B, L, D = x.shape
        H, W = self.image_grid_size
        assert L == H * W, f"Expected L={H*W}, but got L={L}."

        x_grid = x.transpose(1, 2).contiguous().view(B, D, H, W)

        neigh = F.unfold(
            x_grid,
            kernel_size=scale,
            padding=scale // 2,
            stride=1,
        )  # [B, D*K, L]

        K_num = scale * scale
        neigh = neigh.view(B, D, K_num, L)
        neigh = neigh.permute(0, 3, 2, 1).contiguous()  # [B, L, K, D]

        if mask is None:
            neigh_mask = torch.ones(B, L, K_num, device=x.device, dtype=torch.bool)
        else:
            m = mask.float().view(B, 1, H, W)
            m = F.unfold(
                m,
                kernel_size=scale,
                padding=scale // 2,
                stride=1,
            )  # [B, K, L]
            neigh_mask = m.permute(0, 2, 1).contiguous() > 0.5  # [B, L, K]

        return neigh, neigh_mask

    def _local_neighbors(self, x, scale, mask=None):
        if self.mode == "text":
            return self._get_text_neighbors(x, scale, mask)
        return self._get_image_neighbors(x, scale, mask)

    def _evidence_shift_aggregation(self, q, k, v, scale, mask=None):
        """
        Dynamic evidence-shift aggregation.

        q, k, v: [B, L, D]
        return:
            o_shift: [B, L, D]
        """
        B, L, D = q.shape

        k_neigh, neigh_mask = self._local_neighbors(k, scale, mask)  # [B, L, K, D]
        v_neigh, _ = self._local_neighbors(v, scale, mask)           # [B, L, K, D]

        K_num = k_neigh.size(2)

        q_expand = q.unsqueeze(2).expand(B, L, K_num, D)

        pair_feat = torch.cat(
            [
                q_expand,
                k_neigh,
                torch.abs(q_expand - k_neigh),
                q_expand * k_neigh,
            ],
            dim=-1,
        )  # [B, L, K, 4D]

        shift_logits = self.shift_score(pair_feat).squeeze(-1)  # [B, L, K]
        shift_logits = shift_logits.masked_fill(~neigh_mask, float("-inf"))

        shift_weight = torch.softmax(shift_logits, dim=-1)      # [B, L, K]
        shift_weight = torch.nan_to_num(shift_weight, nan=0.0)

        o_shift = torch.sum(shift_weight.unsqueeze(-1) * v_neigh, dim=2)  # [B, L, D]

        return o_shift, shift_weight

    def _weighted_sum(self, tensors, weights):
        out = 0.0
        for w, x in zip(weights, tensors):
            out = out + w * x
        return out

    def forward(self, x, mask=None, return_aux=False):
        """
        x: [B, L, D]
        mask: [B, L], optional

        return:
            x_e: [B, L, D]
        """

        # =========================================================
        # Stage I: shared Q/K/V projection
        # =========================================================
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # =========================================================
        # Stage II-A: semantic attention aggregation
        # =========================================================
        o_att = self._semantic_attention(q, k, v, mask=mask)  # [B, L, D]

        # =========================================================
        # Stage II-B: multi-scale dynamic evidence-shift aggregation
        # =========================================================
        shift_outputs = []
        shift_weights = []
        scale_evidences = []

        for s in self.scales:
            o_shift_s, eta_s = self._evidence_shift_aggregation(
                q=q,
                k=k,
                v=v,
                scale=s,
                mask=mask,
            )  # [B, L, D], [B, L, K]

            evidence_input = torch.cat(
                [
                    o_att,
                    o_shift_s,
                    torch.abs(o_att - o_shift_s),
                ],
                dim=-1,
            )  # [B, L, 3D]

            e_s = torch.sigmoid(self.evidence_head(evidence_input))  # [B, L, 1]

            shift_outputs.append(o_shift_s)
            shift_weights.append(eta_s)
            scale_evidences.append(e_s)

        # =========================================================
        # Cross-scale evidence stability
        # =========================================================
        alpha = F.softmax(self.scale_logits, dim=0)  # [S]

        mu = self._weighted_sum(scale_evidences, alpha)  # [B, L, 1]

        delta = 0.0
        for w, e_s in zip(alpha, scale_evidences):
            delta = delta + w * (e_s - mu).pow(2)

        stability = torch.exp(-delta / (self.tau + self.eps))  # [B, L, 1]
        gate = mu * stability                                  # [B, L, 1]

        o_shift_bar = self._weighted_sum(shift_outputs, alpha)  # [B, L, D]

        # =========================================================
        # Attention-shift adaptive mixing
        # =========================================================
        mix_input = torch.cat(
            [
                o_att,
                o_shift_bar,
                torch.abs(o_att - o_shift_bar),
            ],
            dim=-1,
        )  # [B, L, 3D]

        lam = torch.sigmoid(self.mix_gate(mix_input))  # [B, L, 1]

        o_mix = lam * o_att + (1.0 - lam) * o_shift_bar  # [B, L, D]

        # =========================================================
        # Evidence-gated residual + FFN
        # =========================================================
        y = self.norm1(x + gate * o_mix)
        x_e = self.norm2(y + self.ffn(y))

        if not return_aux:
            return x_e

        aux_info = {
            "o_att": o_att,
            "shift_outputs": shift_outputs,
            "shift_weights": shift_weights,
            "scale_evidences": scale_evidences,
            "scale_weights": alpha.detach(),
            "mean_evidence": mu,
            "evidence_fluctuation": delta,
            "evidence_stability": stability,
            "evidence_gate": gate,
            "mix_lambda": lam,
            "o_shift_bar": o_shift_bar,
            "o_mix": o_mix,
        }

        return x_e, aux_info

    def stability_regularization(self, aux_info):
        return torch.mean(
            aux_info["mean_evidence"] * aux_info["evidence_fluctuation"]
        )

class EvidenceAwareAttentionShiftMixingModule(nn.Module):
    """
    EASM: Evidence-aware Attention-Shift Mixing Module.

    This module replaces the original MSEA/MIESM module.
    Its output is consistent with the original first module.

    Inputs:
        text_feat:  [B, 32, 768]
        image_feat: [B, 196, 768]

    Outputs:
        text_evidence:  [B, 32, 768]
        image_evidence: [B, 196, 768]
    """

    def __init__(
        self,
        dim=768,
        num_heads=8,
        text_scales=(1, 3, 5),
        image_scales=(1, 3, 5),
        image_grid_size=(14, 14),
        shift_hidden_dim=128,
        evidence_hidden_dim=256,
        mix_hidden_dim=256,
        ffn_hidden_dim=1024,
        attn_dropout=0.1,
        proj_dropout=0.1,
        ffn_dropout=0.1,
        tau_text=1.0,
        tau_image=1.0,
        return_aux_default=False,
    ):
        super().__init__()

        self.return_aux_default = return_aux_default

        self.text_easm = EvidenceAwareAttentionShiftBlock(
            dim=dim,
            num_heads=num_heads,
            scales=text_scales,
            mode="text",
            image_grid_size=image_grid_size,
            shift_hidden_dim=shift_hidden_dim,
            evidence_hidden_dim=evidence_hidden_dim,
            mix_hidden_dim=mix_hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            ffn_dropout=ffn_dropout,
            tau=tau_text,
        )

        self.image_easm = EvidenceAwareAttentionShiftBlock(
            dim=dim,
            num_heads=num_heads,
            scales=image_scales,
            mode="image",
            image_grid_size=image_grid_size,
            shift_hidden_dim=shift_hidden_dim,
            evidence_hidden_dim=evidence_hidden_dim,
            mix_hidden_dim=mix_hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            ffn_dropout=ffn_dropout,
            tau=tau_image,
        )

    def forward(
        self,
        text_feat,
        image_feat,
        text_mask=None,
        image_mask=None,
        return_aux=None,
    ):
        if return_aux is None:
            return_aux = self.return_aux_default

        if return_aux:
            text_evidence, text_aux = self.text_easm(
                text_feat,
                mask=text_mask,
                return_aux=True,
            )

            image_evidence, image_aux = self.image_easm(
                image_feat,
                mask=image_mask,
                return_aux=True,
            )

            aux_info = {
                "text_aux": text_aux,
                "image_aux": image_aux,
            }

            return text_evidence, image_evidence, aux_info

        text_evidence = self.text_easm(
            text_feat,
            mask=text_mask,
            return_aux=False,
        )

        image_evidence = self.image_easm(
            image_feat,
            mask=image_mask,
            return_aux=False,
        )

        return text_evidence, image_evidence

    def stability_regularization(self, aux_info):
        loss_text = self.text_easm.stability_regularization(aux_info["text_aux"])
        loss_image = self.image_easm.stability_regularization(aux_info["image_aux"])
        return loss_text + loss_image