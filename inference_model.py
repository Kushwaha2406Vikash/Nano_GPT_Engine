import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from transformer_block import Block

block_size = 6
embedding_dim = 64
n_heads = 2
n_layers = 2

sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")
vocab_size = sp.get_piece_size()

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[
            Block(embedding_dim, block_size, n_heads)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx):
        B,T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


model = TinyGPT()
model.load_state_dict(torch.load("tinygpt.pt", map_location="cpu"))
model.eval()

print("Model loaded successfully!")
