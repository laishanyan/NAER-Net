import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPool(nn.Module):
    """
    Mean pooling with optional mask.
    Input:
        x:    [B, L, D]
        mask: [B, L], 1 for valid tokens, 0 for padding
    Output:
        pooled: [B, D]
    """
    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)

        mask = mask.unsqueeze(-1).float()  # [B, L, 1]
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return x.sum(dim=1) / denom


class EvidenceDecomposition(nn.Module):
    """
    Decompose evidence into local-level and global-level representations.

    Input:
        X_e: [B, L, D]

    Output:
        X_l: [B, D]  local evidence
        X_h: [B, D]  global evidence
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.local_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        self.global_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        self.pool = MeanPool()
        self.norm_l = nn.LayerNorm(dim)
        self.norm_h = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # Local evidence: token/patch-level transformation followed by pooling
        x_l = self.local_proj(x)          # [B, L, D]
        x_l = self.pool(x_l, mask)        # [B, D]
        x_l = self.norm_l(x_l)

        # Global evidence: directly aggregate enhanced evidence
        x_h = self.pool(x, mask)          # [B, D]
        x_h = self.global_proj(x_h)       # [B, D]
        x_h = self.norm_h(x_h)

        return x_l, x_h


class HEFCell(nn.Module):
    """
    Hierarchical Evidence Fusion Cell.

    It contains:
        1. Additive fusion
        2. Multiplicative interaction
        3. Bidirectional cross-attention
        4. Adaptive operator weighting

    Input:
        T: [B, D]
        V: [B, D]

    Output:
        C: [B, D]
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()

        self.dim = dim

        self.t_to_v_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.v_to_t_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.operator_gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3)
        )

        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, T, V):
        """
        T: [B, D]
        V: [B, D]
        """

        # 1. Additive evidence fusion
        F_add = T + V                       # [B, D]

        # 2. Multiplicative evidence interaction
        F_mul = T * V                       # [B, D]

        # 3. Bidirectional cross-modal attention
        T_seq = T.unsqueeze(1)              # [B, 1, D]
        V_seq = V.unsqueeze(1)              # [B, 1, D]

        T_from_V, _ = self.t_to_v_attn(
            query=T_seq,
            key=V_seq,
            value=V_seq,
            need_weights=False
        )                                   # [B, 1, D]

        V_from_T, _ = self.v_to_t_attn(
            query=V_seq,
            key=T_seq,
            value=T_seq,
            need_weights=False
        )                                   # [B, 1, D]

        F_att = T_from_V.squeeze(1) + V_from_T.squeeze(1)  # [B, D]

        # 4. Adaptive weighting over three fusion operators
        gate_input = torch.cat(
            [T, V, T * V, torch.abs(T - V)],
            dim=-1
        )                                   # [B, 4D]

        alpha = F.softmax(self.operator_gate(gate_input), dim=-1)
        alpha_add = alpha[:, 0:1]           # [B, 1]
        alpha_mul = alpha[:, 1:2]           # [B, 1]
        alpha_att = alpha[:, 2:3]           # [B, 1]

        C = (
            alpha_add * F_add
            + alpha_mul * F_mul
            + alpha_att * F_att
        )                                   # [B, D]

        C = self.out_proj(C)
        C = self.norm(C + T + V)

        return C


class InconsistencyAwareRecalibration(nn.Module):
    """
    Inconsistency-aware recalibration.

    Given local fused evidence C_l and global fused evidence C_h,
    estimate their inconsistency and recalibrate their contributions.

    Input:
        C_l: [B, D]
        C_h: [B, D]

    Output:
        C_l_tilde: [B, D]
        C_h_tilde: [B, D]
        delta: [B, 1]
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.proj_l = nn.Linear(dim, dim)
        self.proj_h = nn.Linear(dim, dim)

        self.gate = nn.Sequential(
            nn.Linear(dim * 3 + 1, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim * 2),
            nn.Sigmoid()
        )

        self.norm_l = nn.LayerNorm(dim)
        self.norm_h = nn.LayerNorm(dim)

    def forward(self, C_l, C_h):
        # Project into shared consistency space
        Z_l = self.proj_l(C_l)              # [B, D]
        Z_h = self.proj_h(C_h)              # [B, D]

        # Cosine similarity
        sim = F.cosine_similarity(Z_l, Z_h, dim=-1, eps=1e-8)
        sim = sim.unsqueeze(-1)             # [B, 1]

        # Inconsistency intensity
        delta = 1.0 - sim                   # [B, 1]

        gate_input = torch.cat(
            [C_l, C_h, torch.abs(C_l - C_h), delta],
            dim=-1
        )                                   # [B, 3D + 1]

        gates = self.gate(gate_input)       # [B, 2D]
        g_l, g_h = gates.chunk(2, dim=-1)   # [B, D], [B, D]

        C_l_tilde = self.norm_l(g_l * C_l)
        C_h_tilde = self.norm_h(g_h * C_h)

        return C_l_tilde, C_h_tilde, delta


class AsymmetricEvidenceAggregation(nn.Module):
    """
    Asymmetric aggregation between local and global recalibrated evidence.

    Input:
        C_l_tilde: [B, D]
        C_h_tilde: [B, D]

    Output:
        F: [B, D]
        lambdas: [B, 2]
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.weight_net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2)
        )

        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, C_l_tilde, C_h_tilde):
        weight_input = torch.cat(
            [C_l_tilde, C_h_tilde, torch.abs(C_l_tilde - C_h_tilde)],
            dim=-1
        )                                   # [B, 3D]

        lambdas = F.softmax(self.weight_net(weight_input), dim=-1)
        lambda_l = lambdas[:, 0:1]          # [B, 1]
        lambda_h = lambdas[:, 1:2]          # [B, 1]

        Fused = lambda_l * C_l_tilde + lambda_h * C_h_tilde
        Fused = self.out_proj(Fused)
        Fused = self.norm(Fused + C_l_tilde + C_h_tilde)

        return Fused, lambdas


class HIEF(nn.Module):
    """
    Hierarchical Inconsistency-aware Evidence Fusion.

    Input:
        T_e: [B, L_t, D]
        V_e: [B, L_v, D]
        text_mask:  [B, L_t], optional
        image_mask: [B, L_v], optional

    Output:
        F: [B, D]
        aux: dictionary containing intermediate results
    """
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()

        self.text_decomp = EvidenceDecomposition(dim, dropout)
        self.visual_decomp = EvidenceDecomposition(dim, dropout)

        self.local_fusion = HEFCell(dim, num_heads, dropout)
        self.global_fusion = HEFCell(dim, num_heads, dropout)

        self.recalibration = InconsistencyAwareRecalibration(dim, dropout)
        self.aggregation = AsymmetricEvidenceAggregation(dim, dropout)

    def forward(self, T_e, V_e, text_mask=None, image_mask=None):
        """
        T_e: [B, L_t, D]
        V_e: [B, L_v, D]
        """

        # 1. Evidence decomposition
        T_l, T_h = self.text_decomp(T_e, text_mask)       # [B, D], [B, D]
        V_l, V_h = self.visual_decomp(V_e, image_mask)    # [B, D], [B, D]

        # 2. Hierarchical evidence fusion
        C_l = self.local_fusion(T_l, V_l)                 # [B, D]
        C_h = self.global_fusion(T_h, V_h)                # [B, D]

        # 3. Inconsistency-aware recalibration
        C_l_tilde, C_h_tilde, delta = self.recalibration(C_l, C_h)

        # 4. Asymmetric evidence aggregation
        Fused, lambdas = self.aggregation(C_l_tilde, C_h_tilde)

        aux = {
            "T_l": T_l,
            "T_h": T_h,
            "V_l": V_l,
            "V_h": V_h,
            "C_l": C_l,
            "C_h": C_h,
            "C_l_tilde": C_l_tilde,
            "C_h_tilde": C_h_tilde,
            "delta": delta,
            "lambda": lambdas
        }

        return Fused, aux


class NADERClassifier(nn.Module):
    """
    Example classifier using HIEF.

    Input:
        T_e: [B, L_t, D]
        V_e: [B, L_v, D]

    Output:
        logits: [B, num_classes]
    """
    def __init__(
        self,
        dim=768,
        num_heads=8,
        num_classes=3,
        dropout=0.1
    ):
        super().__init__()

        self.hief = HIEF(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

    def forward(self, T_e, V_e, text_mask=None, image_mask=None):
        Fused, aux = self.hief(
            T_e,
            V_e,
            text_mask=text_mask,
            image_mask=image_mask
        )

        logits = self.classifier(Fused)

        return logits, aux