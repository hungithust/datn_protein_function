"""AMPR: Adaptive Multimodal Protein Representation model."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger('ampr')


class ProjectionHead(nn.Module):
    """Linear(input_dim → d_hidden) + ReLU + LayerNorm."""

    def __init__(self, input_dim, d_hidden=512):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_hidden)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_hidden)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.norm(x)
        return x


class GatingNetwork(nn.Module):
    """
    Compute adaptive weights for 3 modalities.

    Input: (batch, 3*d_hidden) — concatenated [h_seq, h_3di, h_ppi]
    Output: (batch, 3) — softmax weights [α_seq, α_3di, α_ppi]
    """

    def __init__(self, d_hidden=512, d_hidden_gate=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_hidden, d_hidden_gate),
            nn.ReLU(),
            nn.Linear(d_hidden_gate, 3),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_concat):
        logits = self.mlp(h_concat)
        alphas = self.softmax(logits)
        return alphas


class AMPRModel(nn.Module):
    """
    Adaptive Multimodal Protein Representation.

    Fuses 3 modalities (sequence, structure, PPI) via adaptive gating,
    then classifies with either Linear head or BioBERT-fixed weights.
    """

    def __init__(self, d_hidden=512, n_terms=489, dropout_3di=0.15, dropout_ppi=0.25,
                 classifier='both', go_emb_dim=768, ppi_dim=128,
                 structure_modality='prostt5',
                 gnn_input_dim=26, gnn_hidden_dim=256):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_terms = n_terms
        self.dropout_3di = dropout_3di
        self.dropout_ppi = dropout_ppi
        self.classifier_type = classifier
        self.structure_modality = structure_modality

        self.proj_seq = ProjectionHead(1024, d_hidden)
        self.proj_ppi = ProjectionHead(ppi_dim, d_hidden)

        if structure_modality == 'prostt5':
            self.proj_3di = ProjectionHead(1024, d_hidden)
            self.gnn = None
        elif structure_modality == 'gnn':
            from ampr.models.gnn_encoder import GNNEncoder
            self.proj_3di = None
            self.gnn = GNNEncoder(input_dim=gnn_input_dim,
                                  hidden_dim=gnn_hidden_dim,
                                  output_dim=d_hidden)
        else:
            raise ValueError(f"unknown structure_modality: {structure_modality}")

        self.gating = GatingNetwork(d_hidden)
        self.linear_head = nn.Linear(d_hidden, n_terms) if classifier in ['linear', 'both'] else None
        self.go_emb_proj = nn.Linear(go_emb_dim, d_hidden) if classifier in ['biobert', 'both'] else None

        logger.info(f"[MODEL] AMPRModel initialized")
        logger.info(f"[MODEL]   structure_modality={structure_modality}")
        logger.info(f"[MODEL]   d_hidden={d_hidden}, n_terms={n_terms}")
        logger.info(f"[MODEL]   classifier={classifier}")

    def forward(self, x_seq, x_3di, x_ppi, go_emb=None,
                cmap=None, cmap_mask=None, seq_1hot=None,
                return_alphas=False):
        """
        Forward pass.

        Args:
            x_seq: (batch, 1024)
            x_3di: (batch, 1024) — used in prostt5 mode; ignored in gnn mode
            x_ppi: (batch, ppi_dim)
            go_emb: (n_terms, go_emb_dim) if using BioBERT head
            cmap: (batch, L, L) float32 — required in gnn mode
            cmap_mask: (batch, L) bool — required in gnn mode
            seq_1hot: (batch, L, 26) float32 — required in gnn mode
            return_alphas: bool

        Returns:
            logits: (batch, n_terms)
            alphas: (batch, 3) if return_alphas=True
        """
        batch_size = x_seq.size(0)

        h_seq = self.proj_seq(x_seq)
        h_ppi = self.proj_ppi(x_ppi)

        if self.structure_modality == 'prostt5':
            h_3di = self.proj_3di(x_3di)
        else:  # gnn
            assert cmap is not None and cmap_mask is not None and seq_1hot is not None, \
                "GNN mode requires cmap, cmap_mask, seq_1hot in batch"
            h_3di = self.gnn(seq_1hot, cmap, cmap_mask)

        if self.training:
            # Inverted modality dropout: scale kept vectors by 1/(1-p) so
            # expected magnitude is identical at train and eval time.
            mask_3di = (torch.rand(batch_size, 1, device=h_3di.device) > self.dropout_3di).float()
            mask_ppi = (torch.rand(batch_size, 1, device=h_ppi.device) > self.dropout_ppi).float()
            h_3di = h_3di * mask_3di / (1.0 - self.dropout_3di + 1e-8)
            h_ppi = h_ppi * mask_ppi / (1.0 - self.dropout_ppi + 1e-8)

        h_concat = torch.cat([h_seq, h_3di, h_ppi], dim=-1)
        alphas = self.gating(h_concat)

        z = alphas[:, 0:1] * h_seq + alphas[:, 1:2] * h_3di + alphas[:, 2:3] * h_ppi

        if self.classifier_type == 'linear' or (self.classifier_type == 'both' and go_emb is None):
            logits = self.linear_head(z)
        elif self.classifier_type == 'biobert' or (self.classifier_type == 'both' and go_emb is not None):
            go_proj = self.go_emb_proj(go_emb)
            logits = torch.matmul(z, go_proj.t())
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")

        if return_alphas:
            return logits, alphas
        return logits


class AMPRModelV3(nn.Module):
    """
    Phase 3 AMPR — ESM-2 residue + GNN cmap + DeepGO PPI.

    Forward takes a single batch dict (output of collate_variable_length)
    instead of positional tensors — simpler for trainer.
    """

    def __init__(self, n_terms: int,
                 seq_dim: int = 1280, seq_n_heads: int = 8, seq_n_layers: int = 2,
                 gnn_node_dim: int = 256, gnn_n_layers: int = 3,
                 ppi_dim: int = 256,
                 d_hidden: int = 512, fusion_n_heads: int = 8, fusion_n_layers: int = 2,
                 classifier: str = 'both', go_emb_dim: int = 768,
                 cmap_threshold: float = 10.0, dropout: float = 0.1):
        super().__init__()
        from ampr.models.attention_pool import DeepAttentionPool
        from ampr.models.seq_encoder import SeqTransformer
        from ampr.models.gnn_encoder import GNNEncoder
        from ampr.models.cross_modal_fusion import CrossModalFusion

        self.n_terms = n_terms
        self.classifier_type = classifier
        self.cmap_threshold = cmap_threshold

        # Sequence path
        self.seq_encoder = SeqTransformer(d_model=seq_dim, n_heads=seq_n_heads,
                                          n_layers=seq_n_layers, dropout=dropout)
        self.seq_pool = DeepAttentionPool(input_dim=seq_dim, dropout=dropout)
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_dim, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden),
        )

        # GNN path — Node init from ESM-2 residue, GCN on cmap-threshold adjacency
        self.gnn_node_init = nn.Linear(seq_dim, gnn_node_dim)
        self.gnn = GNNEncoder(input_dim=gnn_node_dim, hidden_dim=gnn_node_dim,
                              output_dim=gnn_node_dim, n_layers=gnn_n_layers,
                              cmap_threshold=cmap_threshold)
        self.gnn_pool = DeepAttentionPool(input_dim=gnn_node_dim, dropout=dropout)
        self.gnn_proj = nn.Sequential(
            nn.Linear(gnn_node_dim, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden),
        )

        # PPI path
        self.ppi_proj = nn.Sequential(
            nn.Linear(ppi_dim, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden),
        )

        # Fusion
        self.fusion = CrossModalFusion(d_model=d_hidden, n_heads=fusion_n_heads,
                                       n_layers=fusion_n_layers, dropout=dropout)

        # Heads
        if classifier in ('linear', 'both'):
            self.linear_head = nn.Sequential(
                nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_hidden, n_terms),
            )
        if classifier in ('biobert', 'both'):
            self.go_emb_proj = nn.Linear(d_hidden, go_emb_dim)

        logger.info(f"[MODEL] AMPRModelV3 n_terms={n_terms} d_hidden={d_hidden} "
                    f"seq={seq_n_layers}L gnn={gnn_n_layers}L fusion={fusion_n_layers}L "
                    f"classifier={classifier}")

    def forward(self, batch: dict, go_emb=None) -> torch.Tensor:
        x_res = batch['x_seq_residue']
        seq_mask = batch['seq_mask']
        cmap = batch['cmap']
        x_ppi = batch['x_ppi']
        ppi_mask = batch['ppi_mask']

        # Seq path
        seq_enc = self.seq_encoder(x_res, seq_mask)
        h_seq = self.seq_pool(seq_enc, seq_mask)
        h_seq = self.seq_proj(h_seq)

        # GNN path — GNNEncoder.forward(seq_feat, cmap, mask) -> (B, output_dim) pooled
        node_init = self.gnn_node_init(x_res)
        gnn_res = self.gnn(node_init, cmap, seq_mask)  # (B, gnn_node_dim) already pooled
        h_gnn = self.gnn_proj(gnn_res)

        # PPI path
        h_ppi = self.ppi_proj(x_ppi)
        h_ppi = h_ppi * ppi_mask.float().unsqueeze(-1)

        # Fusion
        z = self.fusion(h_seq, h_gnn, h_ppi, ppi_mask)

        # Head(s)
        if self.classifier_type == 'linear':
            return self.linear_head(z)
        if self.classifier_type == 'biobert':
            assert go_emb is not None
            return torch.matmul(self.go_emb_proj(z), go_emb.t())
        # both
        lin = self.linear_head(z)
        if go_emb is None:
            return lin
        bio = torch.matmul(self.go_emb_proj(z), go_emb.t())
        return 0.5 * lin + 0.5 * bio
