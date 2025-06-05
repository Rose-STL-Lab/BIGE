import torch
import torch.nn as nn
import torch.nn.functional as F

class ProfileEncoder(nn.Module):
    def __init__(self, latent_dim, profile_dim, meta_dim, biomech_dim, time_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.meta_proj = nn.Linear(meta_dim, latent_dim)
        self.biomech_proj = nn.Linear(biomech_dim, latent_dim)
        self.time_proj = nn.Linear(time_dim, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, profile_dim),
            nn.LayerNorm(profile_dim)
        )

    def forward(self, latents, meta, biomech, time):
        if latents.dim() == 2:
            x = latents.unsqueeze(0)
        else:
            x = latents
        latents = F.normalize(latents, dim=-1)
        meta_emb = self.meta_proj(meta).unsqueeze(1).expand(-1, x.shape[1], -1)
        biomech_emb = self.biomech_proj(biomech).unsqueeze(1).expand(-1, x.shape[1], -1)
        time_emb = self.time_proj(time).unsqueeze(1).expand(-1, x.shape[1], -1)
        x = x + meta_emb + biomech_emb + time_emb
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.dropout(attn_out)
        mean_pooled = attn_out.mean(dim=1)
        max_pooled = attn_out.max(dim=1).values
        std_pooled = attn_out.std(dim=1)
        pooled = torch.cat([mean_pooled, max_pooled, std_pooled], dim=-1)
        return self.net(pooled)

class ProfileDecoder(nn.Module):
    def __init__(self, input_dim, conditioning_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, conditioning_dim)
        )
    def forward(self, profile_plus_meta):
        return self.net(profile_plus_meta)