import torch
import torch.nn as nn


class EmbeddingEncoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 weights=None,
                 freeze: bool = False,
                 padding_idx: int = 0,
                 seed = 0):

        super().__init__()
        self.embed_size = embed_size

        if weights is None:
            torch.manual_seed(seed)
            print(f"Rand init emb weights {vocab_size}x{embed_size}")
            self.embed = nn.Embedding(vocab_size,
                                      embed_size,
                                      padding_idx=padding_idx)
        else:
            print(f"Loaded pretrained emb weights {weights.size()}")
            self.embed = nn.Embedding.from_pretrained(weights,
                                                      freeze=freeze,
                                                      padding_idx=padding_idx)

    def forward(self, x):
        return self.embed(x)