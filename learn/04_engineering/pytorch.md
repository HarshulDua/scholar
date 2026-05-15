# PyTorch — Deep Dive

> Covers: tensors, autograd, nn.Module, training loops, the Mult-VAE training
> in Scholar, GPU management, and all interview questions.
> For VAE theory and math, see `variational_autoencoder.md`.

---

## 1. Why PyTorch (Not TensorFlow)?

| | PyTorch | TensorFlow 2.x |
|--|---------|----------------|
| Execution | Eager — runs immediately, use pdb | Eager by default but JAX/tf.function pressure |
| Debugging | Normal Python tools | Still awkward for custom ops |
| Research adoption | ~80% of new papers (2023) | ~20% |
| Production | TorchServe, ONNX export | TF Serving, TFLite |
| Dynamic graphs | Yes — control flow in Python | Limited (tf.function freezes graph) |
| Community | HuggingFace, PyG, torchvision | TFHub, Keras |

PyTorch won research because of eager execution. Writing a VAE with a dynamic reparameterization step (where the sampling depends on training vs eval mode) is trivial in PyTorch — just use `if self.training:`. In TF1, this required building separate computation graphs.

---

## 2. Tensors

```python
import torch

# Creation
x = torch.tensor([1.0, 2.0, 3.0])          # from list — dtype inferred
x = torch.zeros(3, 4)                        # 3×4 of zeros
x = torch.ones(2, 3, 4)                      # 2×3×4 of ones
x = torch.randn(128, 384)                    # standard normal N(0,1)
x = torch.arange(0, 10, step=2)              # [0, 2, 4, 6, 8]
x = torch.full((3, 3), fill_value=7)         # 3×3 of sevens

# From NumPy — zero-copy, shares memory
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
x = torch.from_numpy(arr)   # modification to x CHANGES arr!
arr2 = x.numpy()            # only works for CPU tensors

# Properties
x.shape          # torch.Size([128, 384])
x.dtype          # torch.float32, float16, int64, bool, etc.
x.device         # device('cpu') or device('cuda', index=0)
x.ndim           # number of dimensions
x.numel()        # total elements = product of shape
x.requires_grad  # whether gradients are tracked
```

### 2.1 dtypes Matter for Scholar

```python
# FAISS requires float32
embeddings = embeddings.astype(np.float32)      # numpy
embeddings = torch.tensor(embeddings, dtype=torch.float32)  # torch

# Interaction matrix for Mult-VAE
x = torch.zeros(1, vocab_size, dtype=torch.float32)  # not float64!

# CUDA + bfloat16 for Phi-3 training
model = model.to(dtype=torch.bfloat16)
# bfloat16: same exponent range as float32, less mantissa precision
# fp16: smaller exponent range → numerical instability during training
# bfloat16 is better for training, fp16 for inference on older GPUs
```

---

## 3. Operations

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)

# Element-wise
a + b; a - b; a * b; a / b
torch.exp(a); torch.log(a.abs()); torch.sqrt(a.abs())
torch.sigmoid(a); torch.tanh(a)
F.relu(a); F.gelu(a)  # import torch.nn.functional as F

# Reductions
a.sum()           # scalar
a.sum(dim=0)      # sum over rows → (4,)
a.sum(dim=1, keepdim=True)   # sum over cols → (3, 1)
a.mean(); a.max(); a.min()
a.argmax(dim=1)   # index of max per row → (3,)

# Matrix multiply
c = a @ b.T       # (3,4) @ (4,3) → (3,3)
torch.mm(a, b.T)  # same, 2D only
torch.matmul(a, b.T)  # handles batches

# Cosine similarity
F.cosine_similarity(a, b, dim=1)  # (3,) — cosine per row

# Dot product
torch.dot(a[0], b[0])  # scalar — only for 1D tensors
```

---

## 4. Autograd — Automatic Differentiation

PyTorch records every operation on tensors with `requires_grad=True` in a computation graph. Calling `.backward()` traverses this graph in reverse (backpropagation), computing gradients via the chain rule.

```python
# Simple example
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x    # y = x³ + 2x

y.backward()           # dy/dx = 3x² + 2
print(x.grad)          # tensor(14.)  ← 3*(2²) + 2 = 14 ✓
```

### 4.1 The Computation Graph

```python
a = torch.tensor([1.0, 2.0], requires_grad=True)  # leaf
b = torch.tensor([3.0, 4.0], requires_grad=True)  # leaf
c = a * b          # (3, 8) — intermediate node
d = c.sum()        # scalar — output node

d.backward()

print(a.grad)  # d(d)/d(a) = b = [3, 4]
print(b.grad)  # d(d)/d(b) = a = [1, 2]
```

The graph is built dynamically as operations execute. `d.backward()` walks the graph in reverse using the chain rule.

### 4.2 Detaching and No-Grad

```python
# .detach() — create new tensor without gradient history
# use when you need a value for logging but not for backprop
loss_value = loss.detach().item()   # Python float, no grad tracking

# torch.no_grad() — disable ALL gradient tracking in a block
# use for inference, evaluation, validation
with torch.no_grad():
    predictions = model(x)   # no graph built → less memory, faster
    val_loss = criterion(predictions, y)
```

In Scholar's Mult-VAE, `torch.no_grad()` is used in two places:
1. Getting user embeddings at inference: `with torch.no_grad(): mu, _ = model.encode(x)`
2. Evaluating NDCG@10 on the validation set after each epoch

---

## 5. nn.Module — Building Neural Networks

```python
class MultVAE(nn.Module):
    def __init__(self, vocab_size, latent_dim=200, hidden_dim=600):
        super().__init__()   # always call super().__init__()!

        # Registered layers — PyTorch tracks their parameters automatically
        self.enc_fc1 = nn.Linear(vocab_size, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, latent_dim * 2)
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, vocab_size)
        # nn.Linear(in, out) adds weight (out, in) and bias (out,)
        # weight initialized Kaiming uniform, bias initialized zero

    def forward(self, x):
        # forward() defines the computation
        # Called as model(x), not model.forward(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
```

**What `super().__init__()` does:**
Initializes the Module's internal state — parameter registry, buffers, hooks. Without it, assigning `nn.Linear` layers won't register them as submodules and their parameters won't appear in `model.parameters()`.

### 5.1 Layer Types

```python
# Linear (fully connected)
nn.Linear(in_features, out_features, bias=True)
# Parameters: weight (out, in), bias (out,)

# Activation functions (no parameters)
nn.ReLU(); nn.GELU(); nn.Tanh(); nn.Sigmoid()

# Dropout (zero out random activations at training time)
nn.Dropout(p=0.5)   # 50% dropout rate
# At eval time, dropout is disabled automatically

# Batch Normalization
nn.BatchNorm1d(num_features)  # normalizes each feature across the batch
nn.LayerNorm(normalized_shape)  # normalizes across features for each sample

# Embedding layer (for discrete token IDs)
nn.Embedding(num_embeddings, embedding_dim)
# Used in Phi-3: token IDs → dense vectors
```

### 5.2 nn.Sequential — Chaining Layers

```python
# Instead of defining encode() manually:
encoder = nn.Sequential(
    nn.Linear(vocab_size, hidden_dim),
    nn.Tanh(),
    nn.Dropout(0.5),
)
h = encoder(x)   # applies layers in order
```

Scholar's MultVAE uses explicit forward() rather than Sequential to output both mu and logvar from the same encoder pass.

### 5.3 Model Inspection

```python
model = MultVAE(vocab_size=10000, latent_dim=200, hidden_dim=600)

# All parameters
params = list(model.parameters())  # list of tensors
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Specific layer
model.enc_fc1.weight.shape   # (600, 10000)
model.enc_fc1.bias.shape     # (600,)

# Print architecture
print(model)
# MultVAE(
#   (enc_fc1): Linear(in_features=10000, out_features=600, bias=True)
#   (enc_fc2): Linear(in_features=600, out_features=400, bias=True)
#   ...
# )
```

---

## 6. Loss Functions in Scholar

### 6.1 Multinomial Log-Likelihood (Mult-VAE reconstruction)

```python
def forward(self, recon_x, x, mu, logvar):
    # recon_x: log_softmax output from decoder — shape (batch, vocab_size)
    # x: binary interaction vector — shape (batch, vocab_size)

    # Reconstruction loss = negative log-likelihood under multinomial
    # = -sum(x_i * log_softmax(decoder_output)_i) per user, averaged over batch
    recon_loss = -torch.mean(torch.sum(recon_x * x, dim=-1))
    # recon_x * x: only non-zero where user actually interacted (x_i = 1)
    # sum over vocab_size: sum of log-probs for items user interacted with
    # mean over batch: average over users

    # KL divergence: closed-form for Gaussian
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    )

    return recon_loss + self.beta * kl_loss
```

### 6.2 Cross-Entropy (for comparison)

```python
# Binary cross-entropy (Bernoulli model — what Mult-VAE is NOT)
loss = F.binary_cross_entropy_with_logits(logits, targets)

# Categorical cross-entropy (classification)
loss = F.cross_entropy(logits, class_indices)

# When to use what:
# BCE: multi-label classification (each item independently positive/negative)
# CE: multi-class classification (exactly one class is correct)
# Multinomial LL: when items are "drawn from a distribution" (RecSys)
```

---

## 7. Optimizers

```python
# Adam — adaptive learning rates, usually the best default
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),     # exponential moving average coefficients
    eps=1e-8,               # numerical stability
    weight_decay=1e-4       # L2 regularization
)

# AdamW — decoupled weight decay (correct L2 reg for adaptive methods)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# paged_adamw_8bit (bitsandbytes) — used in Scholar's QLoRA training
# Stores optimizer states in 8-bit instead of 32-bit
# Adam maintains: m (momentum) and v (variance), each full model size
# 8-bit compression: m and v stored as 8-bit, dequantized for updates
# Memory savings: 4× less memory for optimizer states
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(model.parameters(), lr=2e-4)
```

**Why Scholar uses Adam for VAE:**
- Adaptive learning rates handle the large parameter count (millions of params in enc_fc1/dec_fc2)
- Weight decay adds L2 regularization to prevent overfitting on small interaction datasets
- Standard lr=1e-3 is known to work well for VAEs

---

## 8. The Training Loop — Scholar's VAE

```python
def train(epochs=100, batch_size=256, lr=1e-3):
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultVAE(vocab_size=len(paper_ids)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Data
    X_train = torch.tensor(matrix[train_idx], dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train), batch_size=256, shuffle=True)

    best_ndcg = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # KL annealing: beta ramps 0 → 0.2 over first 20 epochs
        beta = min(0.2, epoch / 20 * 0.2)
        criterion = MultVAELoss(beta=beta)

        # ── Training phase ──
        model.train()   # enables dropout, uses batch statistics

        for (batch,) in train_loader:
            batch = batch.to(device)             # move to GPU if available

            # Forward pass
            recon_x, mu, logvar = model(batch)   # model(x) calls forward()
            loss = criterion(recon_x, batch, mu, logvar)

            # Backward pass
            optimizer.zero_grad()   # CRITICAL: reset gradients before backward
            # Why? Gradients accumulate by default (+=). If you forget zero_grad,
            # the gradient from the previous batch is ADDED to this batch's gradient.
            loss.backward()         # compute all gradients via chain rule

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # If the gradient norm exceeds 1.0, scale all gradients down

            optimizer.step()        # update parameters: param -= lr * grad

        # ── Validation phase ──
        if epoch % 5 == 0:
            model.eval()   # disables dropout, uses running statistics
            with torch.no_grad():
                # Validate on held-out users
                recon, mu_val, logvar_val = model(X_val.to(device))
                ndcg = _ndcg_at_k(recon.cpu().numpy(), X_val.numpy(), k=10)

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"Epoch {epoch}: NDCG@10 = {ndcg:.4f}, beta = {beta:.3f}")

    # Restore best checkpoint
    if best_state:
        model.load_state_dict(best_state)
    return model
```

### 8.1 Why `optimizer.zero_grad()` Is Critical

PyTorch accumulates gradients. Without `zero_grad()`:
```python
# WRONG:
loss1.backward()   # grad = g1
loss2.backward()   # grad = g1 + g2 (accumulated!)
optimizer.step()   # updates with g1 + g2 — WRONG
```

This is actually useful for gradient accumulation (see Section 8.2), but in a normal training loop you always want fresh gradients.

### 8.2 Gradient Accumulation (for QLoRA)

```python
# Simulate batch_size=8 with actual batch_size=1 (memory constraint)
# This is how Scholar's QLoRA trains on RTX 2050 4GB

GRAD_ACCUM = 8

for i, batch in enumerate(train_loader):
    outputs = model(**batch)
    loss = outputs.loss / GRAD_ACCUM   # scale the loss!
    loss.backward()                    # accumulate gradients

    if (i + 1) % GRAD_ACCUM == 0:
        optimizer.step()
        optimizer.zero_grad()          # reset after update

# Why divide loss by GRAD_ACCUM?
# Without scaling: gradient = sum of 8 individual gradients
# With scaling: gradient = average of 8 individual gradients (same as batch_size=8)
```

---

## 9. GPU Management

```python
# Check availability
torch.cuda.is_available()           # True/False
torch.cuda.device_count()           # number of GPUs
torch.cuda.get_device_name(0)       # "NVIDIA GeForce RTX 2050"
torch.cuda.get_device_properties(0).total_memory / 1e9  # 4.0 GB

# Move tensors and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
x = x.cuda()          # shorthand
x = x.cpu()           # back to CPU

# Memory management
torch.cuda.memory_allocated(0) / 1e9   # currently allocated GB
torch.cuda.memory_reserved(0) / 1e9    # reserved (may differ from allocated)
torch.cuda.empty_cache()               # release cached blocks back to OS

# Mixed precision (bfloat16 — supported on RTX 2050 Ampere)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(dtype=torch.bfloat16):
    output = model(x)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
# GradScaler compensates for bf16 underflow by scaling loss
# For bfloat16 specifically: same dynamic range as float32, less mantissa
# → less underflow risk than fp16 → GradScaler less critical but still good practice
```

---

## 10. Saving and Loading

```python
# Save model weights only (recommended)
torch.save(model.state_dict(), "mult_vae.pt")

# Load
model = MultVAE(vocab_size=10000, latent_dim=200, hidden_dim=600)
model.load_state_dict(torch.load("mult_vae.pt", map_location="cpu"))
model.eval()
# map_location="cpu" prevents errors when loading a GPU-saved model to CPU

# Scholar's checkpoint format (includes vocab_size for reconstruction):
torch.save({
    "model_state": model.state_dict(),
    "vocab_size": model.vocab_size,
    "latent_dim": model.latent_dim,
    "hidden_dim": model.hidden_dim,
}, MODEL_PATH)

# Load:
checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
m = MultVAE(
    vocab_size=checkpoint["vocab_size"],
    latent_dim=checkpoint["latent_dim"],
    hidden_dim=checkpoint["hidden_dim"],
)
m.load_state_dict(checkpoint["model_state"])
m.eval()
```

**`weights_only=False` explanation:**
PyTorch 2.x added `weights_only=True` as the default for security (pickle-based checkpoints can execute arbitrary code). `weights_only=True` only allows tensors and primitive types. Scholar's checkpoint contains non-tensor fields (`vocab_size`, etc.) that require `weights_only=False`. Always be careful about pickle files from untrusted sources.

---

## 11. DataLoader

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset

# TensorDataset wraps existing tensors
X_train = torch.FloatTensor(interaction_matrix)
dataset = TensorDataset(X_train)

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,        # shuffle at start of each epoch
    num_workers=0,       # 0 = main process only (Windows doesn't support fork)
    pin_memory=True,     # pre-pin memory for faster CPU→GPU transfer
    drop_last=False,     # keep the last incomplete batch
)

# Iteration
for (batch,) in loader:   # TensorDataset returns tuples
    batch = batch.to(device)
    ...

# On Windows: num_workers > 0 requires if __name__ == '__main__': guard
# Scholar uses num_workers=0 for Windows compatibility
```

---

## 12. model.train() vs model.eval()

```python
model.train()    # sets all modules to training mode
model.eval()     # sets all modules to eval mode

# What changes:
# nn.Dropout: train=True → randomly zeroes activations; eval=True → identity
# nn.BatchNorm: train=True → uses batch statistics; eval=True → uses running mean/var
# Custom (Mult-VAE reparameterize): train=True → samples from N(mu,sigma); eval=True → returns mu

def reparameterize(self, mu, logvar):
    if self.training:   # torch.nn.Module.training attribute
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    return mu   # deterministic at eval time
```

---

## 13. Interview Questions

**Q: What is autograd and how does backpropagation work in PyTorch?**
A: PyTorch records every operation on tensors with requires_grad=True into a dynamic computation graph. Each operation stores a reference to its inputs and a gradient function. When you call loss.backward(), PyTorch traverses this graph in reverse (from loss to leaf parameters), applying the chain rule at each node to compute gradients. For example, if y = x², backward computes dy/dx = 2x. For a deep network, the chain rule is applied recursively: dL/dw_1 = (dL/doutput) × (doutput/dh_2) × ... × (dh_1/dw_1). The gradients accumulate in .grad attributes of leaf tensors.

**Q: Why is optimizer.zero_grad() necessary before each backward pass?**
A: PyTorch accumulates gradients by default — calling backward() adds the new gradients to any existing values in .grad. This is useful for gradient accumulation (simulating larger batches) but harmful in normal training. If you forget zero_grad(), each batch's gradients stack up, producing incorrect updates. The standard idiom is: forward → zero_grad → backward → step. Note: zero_grad before backward (not after step) because the gradient is needed for the step.

**Q: Explain the difference between model.train() and model.eval().**
A: They control the behavior of stateful layers: (1) Dropout — in train mode, randomly zeros activations at rate p. In eval mode, passes all activations unchanged (equivalent to multiplying by 1-p). (2) BatchNorm — in train mode, normalizes using batch statistics (mean, var of current batch) and updates running statistics. In eval mode, uses the accumulated running statistics. Forgetting to switch to eval() before validation/inference is a common bug — dropout keeps randomly zeroing activations, giving different (usually lower) performance than training.

**Q: What is gradient clipping and when would you use it in Scholar?**
A: Gradient clipping rescales gradients if their norm exceeds a threshold: if ||g|| > max_norm, g = g × (max_norm / ||g||). Used to prevent exploding gradients — situations where gradients grow exponentially large through deep networks, causing parameter updates to jump far outside the useful range. Scholar uses it during Mult-VAE training (clip_grad_norm_=1.0) because the first and last layers are very large (200K dimensions) and the gradients from the decoder (which sees 200K-way softmax) can be large. In QLoRA training, the Trainer handles clipping automatically.

**Q: How does gradient accumulation let you simulate a larger batch size on limited GPU?**
A: Instead of processing n samples at once (would OOM), process them in m smaller batches of size n/m. After each mini-batch, call backward() but don't call optimizer.step(). After m batches, the gradients have accumulated (summed up). Then call step() and zero_grad(). To get the same effect as a true batch of n samples, divide the loss by m before backward() — this ensures the accumulated gradient is the AVERAGE (not the sum) of the m batches' gradients. Scholar's QLoRA uses batch_size=1 with gradient_accumulation_steps=8, effectively training with batch_size=8.

---

## 14. World-Class Resources

- **PyTorch docs:** https://pytorch.org/docs/stable/
- **Andrej Karpathy's micrograd:** https://github.com/karpathy/micrograd — build autograd from scratch in 100 lines
- **PyTorch internals:** http://blog.ezyang.com/2019/05/pytorch-internals/ — deep dive into autograd
- **Mixed precision training:** https://pytorch.org/docs/stable/amp.html
- **Deep Learning with PyTorch (book, free):** https://pytorch.org/deep-learning-with-pytorch
- **fast.ai course:** https://course.fast.ai/ — practical deep learning using PyTorch
