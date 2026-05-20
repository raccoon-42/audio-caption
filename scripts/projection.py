import torch.nn as nn


class Projection(nn.Module):
    def __init__(self, audio_dim, lm_dim, prefix_len, dropout=0.3, depth=2, use_layernorm=False):
        super().__init__()
        self.prefix_len = prefix_len
        self.lm_dim = lm_dim
        
        if depth == 1:
            self.net = nn.Sequential(
                nn.Linear(audio_dim, prefix_len * lm_dim),
            )
        elif depth == 2:
            layers = [nn.Linear(audio_dim, lm_dim * 2)]
            if use_layernorm:
                layers.append(nn.LayerNorm(lm_dim * 2))
            layers += [
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(lm_dim * 2, prefix_len * lm_dim),
            ]
            self.net = nn.Sequential(*layers)
        else:
            layers = []
            layers += [nn.Linear(audio_dim, lm_dim * 2)]
            if use_layernorm:
                layers.append(nn.LayerNorm(lm_dim * 2))
            layers += [nn.GELU(), nn.Dropout(dropout)]
            
            for _ in range(depth - 2):
                layers += [nn.Linear(lm_dim * 2, lm_dim * 2)]
                if use_layernorm:
                    layers.append(nn.LayerNorm(lm_dim * 2))
                layers += [nn.GELU(), nn.Dropout(dropout)]
                
            layers += [nn.Linear(lm_dim * 2, prefix_len * lm_dim)]
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1, self.prefix_len, self.lm_dim)
