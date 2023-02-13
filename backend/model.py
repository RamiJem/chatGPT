import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32


torch.manual_seed(1337)


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

#print(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoders and decoders
encode = lambda s: [stoi[c]for c in s]
decode = lambda l: "".join([itos[i] for i in l])



data = torch.tensor(encode(text), dtype=torch.long)


# Setting up training and calidation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]





def get_batch(split):
    data = train_data if split == "train" else val_data
    
    ix = torch.randint(len(data)- block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch("train")


class FeedForward(nn.Module):
    """simple linear layer followed by a non-lineaerity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            # projection layer back to residual pathway
            nn.Linear(4*n_embd, n_embd),
        )
    
    def forward(self, x):
        return self.net(x)
    

class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   #(B,T,C)
        q = self.query(x) #(B,T,C)
        # Compute attention scores
        # Normalise the weights (scaling sqrt(channels)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        # Decoder block, only communicates with past Ts
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        # Weighted aggregation
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # projection back to residual pathway
        out = self.proj(out)
        return out


class Block(nn.Module):
    """Transformer block: communication followed by communication"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

   

 
    def forward(self, x):
        # Residual connection: Fork off and come back to input 
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token looks up logits for next token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 32/4=8 dimensions
        # self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx) #(Batch=4, Time=8, C) (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B, T, C) right-aligned and broadcast along B
        # x = self.sa_heads(x) # Apply heads of self-attention (B,T,C)
        # x = self.ffwd(x)
        x = self.blocks(x)
        # Decode after interspersing communication and calculation multiple times 
        logits = self.lm_head(x) # (B,T,vocab_size) decoder language model head
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # predictions
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            # only the last time step
            logits = logits[:, -1, :] # (B, C) since only the last time step
            # softmax to make into probability
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
logits, loss = m(xb, yb)



        
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# Specify a path
PATH = "state_dict_model.pt"

# Save
#torch.save(m.state_dict(), PATH)


for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        
        
        #print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    torch.save(m.state_dict(), PATH)
    
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
# for steps in range(10000):
#     # sampling
#     xb, yb = get_batch("train")

#     # evaluate loss
#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=30)[0].tolist()))

# Load


#print(model2)
# model2 = BigramLanguageModel()
# model2.load_state_dict(torch.load(PATH))
# m2 = model2.to(device)
# m2.eval()
# print(decode(model2.generate(context, max_new_tokens=50)[0].tolist()))

