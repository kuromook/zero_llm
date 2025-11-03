import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import os

# ===== 1. トークナイザ読み込み =====
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

# ===== 2. データセット定義 =====
class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.data = torch.tensor(sp.encode(text, out_type=int), dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

# ===== 3. モデル定義（最小GPT風） =====
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_layer=2, n_head=4, block_size=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head),
            num_layers=n_layer
        )
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits

# ===== 4. データ準備 =====
with open("data/input.txt", encoding="utf-8") as f:
    text = f.read()

block_size = 64
dataset = TextDataset(text, block_size)
print(f"len(data): {len(dataset.data)}")
print(f"len(dataset): {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ===== 5. モデル初期化 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniGPT(vocab_size=sp.vocab_size(), block_size=block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# ===== 6. 学習ループ =====
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, sp.vocab_size()), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | loss: {total_loss/len(dataloader):.4f}")

# ===== 7. モデル保存 =====
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/model_final.pth")
print("✅ モデルを checkpoints/model_final.pth に保存しました")

# ===== 8. モデル再読み込み =====
model.load_state_dict(torch.load("checkpoints/model_final.pth", map_location=device))
model.eval()
print("✅ モデルを再読み込みしました")

# ===== 9. テキスト生成関数 =====
def generate_text(model, sp, start_text="こんにちは", max_new_tokens=50):
    model.eval()
    ids = sp.encode(start_text, out_type=int)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    for _ in range(max_new_tokens):
        logits = model(x)
        next_id = torch.argmax(logits[0, -1]).item()
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
    return sp.decode(x[0].tolist())

# ===== 10. テキスト生成 =====
generated = generate_text(model, sp, "AIは", 40)
print("🧠 生成結果:")
print(generated)

