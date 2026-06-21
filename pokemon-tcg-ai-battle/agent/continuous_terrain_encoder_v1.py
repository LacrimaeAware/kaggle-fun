"""Continuous Terrain Encoder V1.

This module is intentionally not a live agent. It defines the learned
state-action representation used by the Branch B continuous-terrain experiment.

The architecture keeps semantic features separate from search metadata so an
evaluation can test whether learned card/effect/delta structure adds signal
beyond criticality/search uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TerrainDims:
    effect_dim: int
    dynamic_dim: int
    zone_count: int
    global_dim: int
    action_type_count: int
    delta_dim: int
    action_scalar_dim: int
    metadata_dim: int


class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContinuousTerrainEncoderV1(nn.Module):
    """Learned state-action geometry with semantic and full metadata variants."""

    def __init__(
        self,
        *,
        n_cards: int,
        dims: TerrainDims,
        card_dim: int = 32,
        zone_dim: int = 16,
        action_type_dim: int = 16,
        dropout: float = 0.15,
        use_card_embedding: bool = True,
        use_effects: bool = True,
        use_target_entity: bool = True,
        use_deltas: bool = True,
        use_metadata: bool = True,
    ):
        super().__init__()
        self.dims = dims
        self.card_dim = card_dim
        self.use_card_embedding = bool(use_card_embedding)
        self.use_effects = bool(use_effects)
        self.use_target_entity = bool(use_target_entity)
        self.use_deltas = bool(use_deltas)
        self.use_metadata = bool(use_metadata)

        self.card_embedding = nn.Embedding(max(1, n_cards), card_dim)
        self.zero_card = nn.Parameter(torch.zeros(card_dim), requires_grad=False)
        self.effect_encoder = MLP([dims.effect_dim, 64, 32], dropout)
        self.dynamic_encoder = MLP([dims.dynamic_dim, 64, 32], dropout)
        self.zone_embedding = nn.Embedding(max(1, dims.zone_count), zone_dim)
        entity_in = card_dim + 32 + 32 + zone_dim
        self.entity_encoder = MLP([entity_in, 128, 64], dropout)

        self.zone_phi = MLP([64, 128, 64], dropout)
        self.zone_rho = MLP([128, 64], dropout)
        self.root_encoder = MLP([dims.zone_count * 64 + dims.global_dim, 256, 128], dropout)

        self.action_type_embedding = nn.Embedding(max(1, dims.action_type_count), action_type_dim)
        self.delta_encoder = MLP([dims.delta_dim, 64, 32], dropout)
        action_in = action_type_dim + card_dim + 64 + 32 + 32 + dims.action_scalar_dim
        self.action_encoder = MLP([action_in, 128, 64], dropout)

        self.state_proj = nn.Linear(128, 64)
        self.action_proj = nn.Linear(64, 64)
        self.semantic_latent = MLP([64 * 4, 128, 64], dropout)
        self.metadata_branch = MLP([dims.metadata_dim, 32, 16], dropout)
        self.full_latent = MLP([80, 64], dropout)

        self.policy_head = nn.Linear(64, 1)
        self.high_regret_head = nn.Linear(64, 1)
        self.unacceptable_head = nn.Linear(64, 1)
        self.acceptable_head = nn.Linear(64, 1)
        self.instability_head = nn.Linear(64, 1)
        self.residual_head = nn.Linear(64, 1)

        self.log_sigma = nn.ParameterDict({
            "ranking": nn.Parameter(torch.tensor(0.0)),
            "high_regret": nn.Parameter(torch.tensor(0.0)),
            "unacceptable": nn.Parameter(torch.tensor(0.0)),
            "acceptable": nn.Parameter(torch.tensor(0.0)),
            "instability": nn.Parameter(torch.tensor(0.0)),
            "residual": nn.Parameter(torch.tensor(0.0)),
            "contrastive": nn.Parameter(torch.tensor(0.0)),
        })

    def _card_emb(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.long()
        known = idx >= 0
        safe = torch.where(known, idx, torch.zeros_like(idx))
        emb = self.card_embedding(safe)
        emb = torch.where(known.unsqueeze(-1), emb, self.zero_card.view(*([1] * (emb.dim() - 1)), -1))
        if not self.use_card_embedding:
            emb = emb * 0.0
        return emb

    def encode_entities(
        self,
        entity_card_ids: torch.Tensor,
        entity_effects: torch.Tensor,
        entity_dynamic: torch.Tensor,
        entity_zone_ids: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        card = self._card_emb(entity_card_ids)
        effects = self.effect_encoder(entity_effects if self.use_effects else entity_effects * 0.0)
        dynamic = self.dynamic_encoder(entity_dynamic if self.use_target_entity else entity_dynamic * 0.0)
        zone = self.zone_embedding(entity_zone_ids.long())
        ent = self.entity_encoder(torch.cat([card, effects, dynamic, zone], dim=-1))
        phi = self.zone_phi(ent)

        zone_embeddings = []
        for z in range(self.dims.zone_count):
            zm = (entity_zone_ids == z) & entity_mask.bool()
            zf = phi.masked_fill(~zm.unsqueeze(-1), 0.0)
            count = zm.sum(dim=1, keepdim=True).clamp_min(1).float()
            mean_pool = zf.sum(dim=1) / count
            max_pool = phi.masked_fill(~zm.unsqueeze(-1), -1e9).max(dim=1).values
            has_zone = zm.any(dim=1, keepdim=True)
            max_pool = torch.where(has_zone, max_pool, torch.zeros_like(max_pool))
            zone_embeddings.append(self.zone_rho(torch.cat([mean_pool, max_pool], dim=-1)))
        return torch.cat(zone_embeddings, dim=-1)

    def encode_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        zones = self.encode_entities(
            batch["entity_card_ids"],
            batch["entity_effects"],
            batch["entity_dynamic"],
            batch["entity_zone_ids"],
            batch["entity_mask"],
        )
        return self.root_encoder(torch.cat([zones, batch["global_features"]], dim=-1))

    def encode_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        atype = self.action_type_embedding(batch["action_type"].long())
        acting = self._card_emb(batch["action_card_id"])
        target = self.entity_encoder(torch.cat([
            self._card_emb(batch["target_card_id"]),
            self.effect_encoder(batch["target_effects"] if self.use_effects else batch["target_effects"] * 0.0),
            self.dynamic_encoder(batch["target_dynamic"] if self.use_target_entity else batch["target_dynamic"] * 0.0),
            self.zone_embedding(batch["target_zone_id"].long()),
        ], dim=-1))
        action_effect = self.effect_encoder(batch["action_effects"] if self.use_effects else batch["action_effects"] * 0.0)
        delta = self.delta_encoder(batch["option_deltas"] if self.use_deltas else batch["option_deltas"] * 0.0)
        return self.action_encoder(torch.cat([
            atype,
            acting,
            target,
            action_effect,
            delta,
            batch["action_scalars"],
        ], dim=-1))

    def latent(self, batch: dict[str, torch.Tensor], *, variant: str = "full") -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state_proj(self.encode_state(batch))
        action = self.action_proj(self.encode_action(batch))
        fusion = torch.cat([state, action, state * action, torch.abs(state - action)], dim=-1)
        z_sem = self.semantic_latent(fusion)
        if variant == "semantic" or not self.use_metadata:
            return z_sem, z_sem
        meta = self.metadata_branch(batch["metadata"])
        z_full = self.full_latent(torch.cat([z_sem, meta], dim=-1))
        return z_full, z_sem

    def forward(self, batch: dict[str, torch.Tensor], *, variant: str = "full") -> dict[str, torch.Tensor]:
        z, z_sem = self.latent(batch, variant=variant)
        return {
            "z": z,
            "z_semantic": z_sem,
            "policy_logit": self.policy_head(z).squeeze(-1),
            "high_regret_logit": self.high_regret_head(z).squeeze(-1),
            "unacceptable_logit": self.unacceptable_head(z).squeeze(-1),
            "acceptable_logit": self.acceptable_head(z).squeeze(-1),
            "instability": torch.sigmoid(self.instability_head(z).squeeze(-1)),
            "residual": torch.tanh(self.residual_head(z).squeeze(-1)),
        }

    def weighted_loss(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        log_sigma = self.log_sigma[name].clamp(-4.0, 4.0)
        return torch.exp(-log_sigma) * loss + log_sigma


def supervised_contrastive_loss(
    z: torch.Tensor,
    action_family: torch.Tensor,
    profile: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.15,
) -> torch.Tensor:
    """Small supervised contrastive loss for cross-game terrain geometry."""
    if z.shape[0] < 4:
        return z.sum() * 0.0
    zn = F.normalize(z, dim=-1)
    sim = zn @ zn.T / temperature
    same_family = action_family[:, None] == action_family[None, :]
    cross_game = group_ids[:, None] != group_ids[None, :]
    dist = torch.cdist(profile.float(), profile.float(), p=2)
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    positives = same_family & cross_game & (dist < 0.35) & ~eye
    valid = positives.any(dim=1)
    if not valid.any():
        return z.sum() * 0.0
    logits = sim.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_log_prob = (log_prob * positives.float()).sum(dim=1) / positives.float().sum(dim=1).clamp_min(1.0)
    return -pos_log_prob[valid].mean()
