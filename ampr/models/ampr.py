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
                 classifier='both', go_emb_dim=768, ppi_dim=128):

        super().__init__()

        self.d_hidden = d_hidden
        self.n_terms = n_terms
        self.dropout_3di = dropout_3di
        self.dropout_ppi = dropout_ppi
        self.classifier_type = classifier

        self.proj_seq = ProjectionHead(1024, d_hidden)
        self.proj_3di = ProjectionHead(1024, d_hidden)
        self.proj_ppi = ProjectionHead(ppi_dim, d_hidden)

        self.gating = GatingNetwork(d_hidden)

        self.linear_head = nn.Linear(d_hidden, n_terms) if classifier in ['linear', 'both'] else None
        self.go_emb_proj = nn.Linear(go_emb_dim, d_hidden) if classifier in ['biobert', 'both'] else None

        logger.info(f"[MODEL] AMPRModel initialized")
        logger.info(f"[MODEL]   d_hidden={d_hidden}, n_terms={n_terms}")
        logger.info(f"[MODEL]   dropout: 3di={dropout_3di}, ppi={dropout_ppi}")
        logger.info(f"[MODEL]   classifier={classifier}")

    def forward(self, x_seq, x_3di, x_ppi, go_emb=None, return_alphas=False):
        """
        Forward pass.

        Args:
            x_seq: (batch, 1024)
            x_3di: (batch, 1024)
            x_ppi: (batch, 128)
            go_emb: (n_terms, 768) if using BioBERT head
            return_alphas: bool

        Returns:
            logits: (batch, n_terms)
            alphas: (batch, 3) if return_alphas=True
        """
        batch_size = x_seq.size(0)

        h_seq = self.proj_seq(x_seq)
        h_3di = self.proj_3di(x_3di)
        h_ppi = self.proj_ppi(x_ppi)

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
