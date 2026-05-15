# Variational Autoencoders — Deep Dive

> Covers: generative model theory, VAE derivation (ELBO), reparameterization trick,
> KL annealing, Mult-VAE for collaborative filtering, and Scholar's exact implementation.

---

## 1. What Problem Do VAEs Solve?

### The Generative Modeling Problem

Given a dataset of papers (represented as user interaction vectors), can we learn a compressed, continuous representation of "what a user likes" that:
1. Captures the underlying structure (users who read NLP papers vs CV papers)
2. Allows generating new predictions (what paper would this user probably enjoy?)
3. Handles uncertainty (we don't know everything about a user from 5 interactions)

A classic autoencoder can do (1) but not (2) or (3). A VAE adds probabilistic structure.

### The Bottleneck Representation Problem

An autoencoder compresses data:
```
input x  →  encoder  →  z (bottleneck)  →  decoder  →  reconstruction x̂
```

The problem: if the bottleneck can be ANY value, there's no structure in the latent space. Nearby points in z-space might decode to completely different outputs. You can't interpolate, sample, or generalize.

**The fix:** force z to follow a known distribution (standard Gaussian N(0,1)). This is what makes a VAE "variational".

---

## 2. Background: Probability Theory

Before the math, you need these building blocks:

**Bayes' theorem:**
```
p(z|x) = p(x|z) × p(z) / p(x)
```
- p(z|x) = posterior: given data x, what's the latent code z?
- p(x|z) = likelihood: given latent z, how likely is data x?
- p(z) = prior: our assumption about z before seeing data (N(0,1))
- p(x) = evidence: probability of data x under the model

The posterior p(z|x) is what we want — it tells us the latent representation for a given input. But it's intractable to compute directly.

**Intractability of p(x):**
```
p(x) = ∫ p(x|z) p(z) dz
```
This integral over ALL possible z is usually intractable — there's no closed form. This is why we need variational inference.

---

## 3. Variational Inference — Approximating the Intractable

Instead of computing p(z|x) exactly, approximate it with a simpler distribution q_φ(z|x) (parameterized by φ). We want q_φ to be as close as possible to the true posterior p(z|x).

**KL divergence** measures how different two distributions are:
```
KL(q || p) = E_q [log q(z|x) - log p(z|x)]
           = ∫ q(z|x) log [q(z|x) / p(z|x)] dz

Properties:
  KL(q || p) ≥ 0  always
  KL(q || p) = 0  iff q = p exactly
```

Goal: minimize KL(q_φ(z|x) || p(z|x)).

---

## 4. The ELBO — Evidence Lower Bound

We can't minimize KL directly because p(z|x) is intractable (requires p(x)). Instead, derive a lower bound on log p(x) that we CAN optimize.

**Full derivation:**
```
log p(x)  =  log ∫ p(x,z) dz
           =  log ∫ q(z|x) × [p(x,z) / q(z|x)] dz    (multiply and divide by q)
           ≥  ∫ q(z|x) × log [p(x,z) / q(z|x)] dz    (Jensen's inequality: log E ≥ E log)
           =  E_{q(z|x)} [log p(x,z) - log q(z|x)]
           =  E_{q(z|x)} [log p(x|z) + log p(z) - log q(z|x)]
           =  E_{q(z|x)} [log p(x|z)] - KL(q(z|x) || p(z))

This is the ELBO (Evidence Lower Bound):

ELBO = E_{q(z|x)} [log p(x|z)] - KL(q(z|x) || p(z))
         └─── reconstruction ───┘   └──── KL penalty ────┘
```

**Maximizing ELBO is equivalent to:**
1. Maximizing the reconstruction quality E[log p(x|z)] — decode z back to x accurately
2. Minimizing KL(q || p) — keep the approximate posterior close to the prior N(0,1)

The KL term acts as a regularizer that prevents the encoder from just memorizing training data by spreading representations around in a ball of radius 1 around the origin.

**Connection to log p(x):**
```
log p(x) = ELBO + KL(q(z|x) || p(z|x))
                   └── always ≥ 0 ──┘

Therefore: log p(x) ≥ ELBO  (ELBO is a lower bound on the log-likelihood)
Maximizing ELBO maximizes a lower bound on log p(x) while also minimizing the gap.
```

---

## 5. VAE Architecture — Encoder and Decoder

The encoder q_φ(z|x) is parameterized as a Gaussian:
```
q_φ(z|x) = N(μ_φ(x), σ_φ²(x))

μ_φ(x)   = neural network output (mean vector)
σ_φ²(x)  = neural network output (variance vector, in log form for stability)
```

The decoder p_θ(x|z) reconstructs x from z.

```python
class VAE(nn.Module):
    def encode(self, x):
        h = torch.tanh(self.enc_fc1(x))    # hidden representation
        out = self.enc_fc2(h)
        mu, logvar = out.chunk(2, dim=-1)   # split into mean and log-variance
        return mu, logvar                   # both shape: (batch, latent_dim)

    def decode(self, z):
        h = torch.tanh(self.dec_fc1(z))
        return self.dec_fc2(h)              # raw logits (shape: batch × vocab_size)
```

The encoder outputs **log(σ²)** rather than σ² because:
- logvar can be any real number (σ² must be positive)
- gradient is smoother in log space
- To get σ: `std = exp(0.5 * logvar)` = `exp(0.5 * log(σ²))` = σ

---

## 6. The Reparameterization Trick

**Problem:** we need to sample z ~ q(z|x) = N(μ, σ²) during the forward pass and then backpropagate through the sampling operation. But you can't take gradients through a random sample.

**Without reparameterization:**
```
z ~ N(μ_φ(x), σ_φ²(x))   ← sampling is not differentiable
↓
loss
↓
dLoss/dφ = ???             ← can't compute
```

**With reparameterization:**
```
ε ~ N(0, I)                ← sample from a FIXED standard normal (not parameterized)
z = μ_φ(x) + σ_φ(x) × ε  ← deterministic function of φ, given ε
↓
loss
↓
dLoss/dφ = dLoss/dz × dz/dμ + dLoss/dz × dz/dσ  ← well-defined!
```

By reparameterizing z = μ + σ × ε, the random part (ε) is separated from the parameterized part (μ, σ). Gradients can flow through μ and σ normally.

```python
def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if self.training:
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 × log(σ²))
        eps = torch.randn_like(std)    # ε ~ N(0, I), same shape as std
        return mu + eps * std          # z = μ + σε
    return mu                          # at inference: use the mean (no noise)
```

**Why `randn_like(std)` not `randn(std.shape)`?**
`randn_like` creates a tensor on the same device as std. If std is on CUDA, ε is on CUDA automatically. `randn(std.shape)` would create a CPU tensor by default.

**At inference time, return mu:** during inference, we want the most likely latent code for the input, not a random sample. Using μ directly gives a deterministic, lower-variance embedding.

---

## 7. The KL Divergence for Gaussian — Closed Form

When q(z|x) = N(μ, σ²I) and p(z) = N(0, I), the KL divergence has a closed form:

```
KL(N(μ, σ²) || N(0, I)) = -0.5 × Σ_j (1 + log(σ_j²) - μ_j² - σ_j²)

Derivation:
KL = ∫ q log(q/p) dz
   = ∫ N(μ,σ²) × [log N(μ,σ²) - log N(0,1)] dz

For each dimension j independently:
   = E_q[-0.5 log(2πσ²) - (z-μ)²/(2σ²)] - E_q[-0.5 log(2π) - z²/2]
   = -0.5 [log(σ²) + 1 - μ² - σ²]
   = -0.5 [1 + log(σ²) - μ² - σ²]

Sum over all latent dimensions:
KL = -0.5 × Σ_j (1 + logvar_j - μ_j² - exp(logvar_j))
```

```python
class MultVAELoss(nn.Module):
    def forward(self, recon_x, x, mu, logvar):
        # Reconstruction: multinomial log-likelihood (see Section 8)
        recon_loss = -torch.mean(torch.sum(recon_x * x, dim=-1))

        # KL divergence: closed form for Gaussian
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        )
        # Note: logvar.exp() = σ², so this is exactly the formula above

        return recon_loss + self.beta * kl_loss
```

---

## 8. Mult-VAE — VAE for Collaborative Filtering

Liang et al. (2018) "Variational Autoencoders for Collaborative Filtering" adapted the VAE for user-item interaction data.

### Why VAE for RecSys?

Standard collaborative filtering methods (matrix factorization, ALS) produce point estimates of user preferences. They don't model uncertainty. A user who has read 2 papers is much more uncertain than one who has read 200. Mult-VAE's latent distribution q(z|x) captures this uncertainty.

### The Interaction Vector

Each user's history is represented as a bag-of-items vector:
```
user_1 reads papers [42, 7, 99, 200, 15]:
  x = [0, 0, ..., 1, ..., 1, ..., 1, ..., 1, ..., 1, ...]
       ^ paper_0        ^ paper_7      ^ paper_42...

x is binary: x_i = 1 if user has read paper i, 0 otherwise
Shape: (vocab_size,) = (n_papers,)

For Scholar with 200K papers: x is a 200,000-dimensional vector!
```

### The Multinomial Likelihood

For the decoder, instead of Gaussian reconstruction loss (MSE), Liang et al. use multinomial log-likelihood:
```
p(x|z) = Multinomial(x ; softmax(f_θ(z)))

Reconstruction loss = -E[log p(x|z)]
                   = -E[Σ_i x_i × log(softmax(f_θ(z))_i)]
                   = -mean over users of: sum over items of: x_i × log_softmax(z)_i
```

**Why multinomial, not Bernoulli or Gaussian?**
- Bernoulli: treats each item independently, misses the fact that reading one paper affects what you'll read next
- Gaussian: appropriate for real-valued ratings (1-5 stars), but interactions are binary
- Multinomial: models the SET of items as a draw from a distribution. "Which item will this user interact with next?" — this is a multinomial draw over all items

```python
def decode(self, z: torch.Tensor) -> torch.Tensor:
    h = torch.tanh(self.dec_fc1(z))
    return F.log_softmax(self.dec_fc2(h), dim=-1)  # returns LOG probabilities

# In loss:
recon_loss = -torch.mean(torch.sum(recon_x * x, dim=-1))
# recon_x is log_softmax output, x is the binary interaction vector
# sum(log_softmax * x) = sum of log-probabilities for the items the user interacted with
# This is the multinomial log-likelihood
```

---

## 9. KL Annealing — Scholar's Constraint

**Problem with standard VAE training:** the KL term sometimes becomes too dominant early in training, causing the encoder to push all representations to the prior (μ≈0, σ²≈1 for everything). The model ignores the input entirely — this is called "posterior collapse."

**KL annealing solution:** start with β=0 (no KL penalty), gradually increase to β=0.2 over training.

```python
# In scholar/recsys/train_vae.py — the exact constraint from CLAUDE.md
for epoch in range(1, epochs + 1):
    # KL annealing: beta ramps 0 → 0.2 over first 20 epochs
    beta = min(0.2, epoch / 20 * 0.2)
    # epoch 1:  beta = 0.01
    # epoch 10: beta = 0.10
    # epoch 20+: beta = 0.20
```

**Why β_max = 0.2, not 1.0?**
β < 1 is the β-VAE formulation (Higgins et al., 2017). Setting β < 1 puts less pressure on the prior, allowing the posterior to deviate from N(0,1). This gives more informative latent codes at the cost of less regularity. For Scholar's recommendation task, informative codes (capturing user preferences) are more important than smooth interpolation between users.

**Linear schedule reason:** abrupt switch from β=0 to β=0.2 can cause training instability. The linear ramp gives the decoder time to learn to reconstruct first (β=0 phase), then the encoder learns to compress while the decoder can already handle it.

---

## 10. Scholar's Complete Mult-VAE Architecture

```python
class MultVAE(nn.Module):
    def __init__(self, vocab_size: int, latent_dim: int = 200, hidden_dim: int = 600):
        super().__init__()
        self.vocab_size = vocab_size   # = number of papers (e.g. 200,000)
        self.latent_dim = latent_dim   # = 200 (user embedding size)

        # Encoder: vocab_size → 600 → 400 (mu + logvar stacked)
        self.enc_fc1 = nn.Linear(vocab_size, hidden_dim)     # 200K → 600
        self.enc_fc2 = nn.Linear(hidden_dim, latent_dim * 2) # 600 → 400

        # Decoder: 200 → 600 → vocab_size
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)     # 200 → 600
        self.dec_fc2 = nn.Linear(hidden_dim, vocab_size)     # 600 → 200K

    def forward(self, x):
        mu, logvar = self.encode(x)      # (batch, 200) each
        z = self.reparameterize(mu, logvar)  # (batch, 200)
        recon_x = self.decode(z)         # (batch, vocab_size)
        return recon_x, mu, logvar
```

**Parameter count:**
```
enc_fc1: 200,000 × 600 + 600 = 120,000,600 params
enc_fc2:      600 × 400 + 400 = 240,400 params
dec_fc1:      200 × 600 + 600 = 120,600 params
dec_fc2:      600 × 200,000 + 200,000 = 120,200,000 params
Total: ~240 million params  (for 200K papers)
```

The huge first and last layers (linear maps to/from vocab_size) dominate. This is why larger corpora need more GPU memory.

**Why tanh activations, not ReLU?**
For user interaction data (binary vectors), tanh keeps activations in [-1, 1] and is smooth everywhere (no dying neuron problem). ReLU can create dead neurons when the input is sparse (binary with mostly zeros).

---

## 11. Getting User Embeddings from the Trained Model

```python
def get_user_embedding(history_paper_ids, all_paper_ids, model):
    # Build the interaction vector for this user
    id_to_idx = {pid: i for i, pid in enumerate(all_paper_ids)}
    x = torch.zeros(1, len(all_paper_ids), dtype=torch.float32)
    for pid in history_paper_ids:
        if pid in id_to_idx:
            x[0, id_to_idx[pid]] = 1.0

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x)     # just the mean, no sampling at inference
    return mu.squeeze(0).numpy()   # (200,) numpy array
```

This 200-dimensional vector IS the user's personalized representation. It's used in:
1. FAISS query via `dense_retriever.query_by_vector(user_emb)` — find papers similar to user's taste
2. LTR feature `user_paper_cosine` — cosine similarity between user_emb and each candidate's FAISS embedding
3. MMR `adaptive_lambda` — diversity of user history embeddings determines exploration vs. exploitation

---

## 12. Cold Start Problem and Fallback

New users with no history have x = 0 (zero vector). The encoder produces μ = something near 0 (since tanh(0) = 0 → enc_fc2(0) = bias ≈ 0).

Scholar's fallback:
```python
def _fallback_sentence_bert(paper_ids: list[str]) -> np.ndarray:
    """Average sentence-BERT embeddings of history papers as a cold-start proxy."""
    # Load abstracts of history papers from DB
    # Encode with all-MiniLM-L6-v2
    # Return the mean embedding (384-dimensional)
```

This is triggered when:
- Model not yet trained (`model is None`)
- Model is near initialization weights (`_is_near_init(model)`)
- Vocabulary size mismatch (model trained on different corpus)

The fallback gives a reasonable 384-dim embedding even without Mult-VAE training. It's less personalized (no collaborative filtering signal) but functional.

---

## 13. Training Loop Details

```python
def train(epochs=100, batch_size=256, lr=1e-3, latent_dim=200, hidden_dim=600):
    # Load interactions from user_history table
    interactions = _load_interactions()
    matrix, user_ids, paper_ids = _build_interaction_matrix(interactions)
    # matrix shape: (n_users, n_papers) — dense binary matrix

    # 80/10/10 train/val/test split
    perm = np.random.permutation(n_users)
    X_train = torch.tensor(matrix[train_idx], dtype=torch.float32)

    # DataLoader batches users (not papers!)
    train_loader = DataLoader(TensorDataset(X_train), batch_size=256, shuffle=True)

    model = MultVAE(vocab_size=len(paper_ids), latent_dim=200, hidden_dim=600)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        beta = min(0.2, epoch / 20 * 0.2)    # KL annealing
        model.train()

        for (batch,) in train_loader:
            batch = batch.to(device)
            recon_x, mu, logvar = model(batch)  # forward pass
            loss = MultVAELoss(beta)(recon_x, batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate every 5 epochs using NDCG@10
        if epoch % 5 == 0:
            ndcg = evaluate_ndcg(model, X_val, k=10)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_state = model.state_dict()

    # Save best model
    torch.save({
        "model_state": best_state,
        "vocab_size": len(paper_ids),
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
    }, MODEL_PATH)
    # Save paper ID → index mapping for inference
    json.dump(paper_ids, open(VOCAB_PATH, "w"))
```

**Why Adam and not SGD?**
Adam is adaptive — it adjusts learning rates per parameter. The first layer (200K × 600) has far more parameters than the latent layers (200 × 600). Adam handles this imbalance better than a single global learning rate.

---

## 14. Evaluation: NDCG@10 for Recommendation

```python
def _ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    # scores: decoder output (predicted interest in each paper)
    # labels: ground truth binary interactions
    top_k = np.argsort(scores)[::-1][:k]   # top-k predicted papers
    dcg = sum(float(labels[i]) / np.log2(rank + 2) for rank, i in enumerate(top_k))
    ideal = sorted(labels.tolist(), reverse=True)[:k]
    idcg = sum(v / np.log2(rank + 2) for rank, v in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
```

For validation, we hold out 20% of each user's interactions as ground truth and predict on the remaining 80%. NDCG@10 measures how well the model ranks those held-out items.

---

## 15. Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| **Matrix Factorization (ALS)** | No uncertainty modeling, deterministic embeddings. SVD++ is fast but gives point estimates with no probabilistic interpretation. |
| **Neural Collaborative Filtering (NCF)** | Discriminative model (predict specific user-item pairs). No latent distribution over users — can't capture uncertainty. |
| **Autoencoder (no VAE)** | Deterministic bottleneck = no structured latent space. Can't interpolate or handle uncertainty. Also tends to overfit on sparse interaction data. |
| **EASE (Embarrassingly Shallow Autoencoders)** | Extremely simple, competitive on dense data but no user representation for personalized FAISS queries. |
| **BERT4Rec** | Sequential recommendation (order matters). Scholar interactions have no temporal order. Overkill. |
| **Collaborative metric learning** | More complex to implement, doesn't add much over Mult-VAE for the scholar use case. |

---

## 16. Interview Questions

**Q: What is the ELBO and why do we maximize it instead of the log-likelihood directly?**
A: ELBO (Evidence Lower Bound) = E[log p(x|z)] - KL(q(z|x) || p(z)). We can't maximize log p(x) directly because it requires integrating over all possible z: p(x) = ∫ p(x|z)p(z)dz, which has no closed form. The ELBO is a lower bound on log p(x) that CAN be computed: log p(x) = ELBO + KL(q||p_posterior) ≥ ELBO. Maximizing ELBO simultaneously maximizes the lower bound AND minimizes the gap (KL to the true posterior).

**Q: Explain the reparameterization trick. Why is it necessary?**
A: To train VAEs with gradient descent, we need to backpropagate through the sampling operation z ~ N(μ,σ²). Sampling is stochastic — you can't take gradients through it. The trick: reparameterize z = μ + σ × ε where ε ~ N(0,I). Now the randomness (ε) is separated from the learned parameters (μ, σ). The gradient can flow through μ and σ normally. This is what makes VAEs trainable end-to-end.

**Q: What is posterior collapse and how does KL annealing fix it?**
A: Posterior collapse occurs when the encoder ignores the input and always outputs approximately the prior (μ≈0, σ≈1 for all inputs). The model learns to reconstruct using only the decoder's prior. It happens because the KL term is minimized (to zero) by pushing q to the prior, and early in training the decoder is good enough to reconstruct without the latent code. KL annealing fixes this by starting with β=0 (no KL penalty), letting the encoder learn meaningful representations first. By the time β increases to its final value, the decoder has learned to depend on z, so it resists collapse.

**Q: Why use multinomial likelihood instead of Bernoulli for interaction prediction?**
A: Bernoulli treats each item independently: P(x) = Π P(x_i). It misses dependencies — reading paper A probably makes paper B more likely. Multinomial models the set of items as draws from a categorical distribution over all items: "what is the next item this user will interact with?" This better captures the competitive nature of recommendation (if you read NLP papers, the whole NLP cluster becomes more likely, not each paper independently).

**Q: How is Mult-VAE's 200-dim user embedding used in the rest of the Scholar pipeline?**
A: Three ways: (1) Direct FAISS query — the 200-dim embedding is passed to `dense_retriever.query_by_vector()` to find papers with similar content embeddings. But note the dimensionality mismatch (200 vs 384) — Scholar uses the sentence-BERT fallback for cold start to get 384-dim vectors, which is a known gap. (2) LTR feature — `user_paper_cosine` is computed as cosine similarity between user_emb and each candidate paper's FAISS embedding. (3) MMR lambda — `adaptive_lambda()` computes how diverse the user's history is using cosine similarities between their history embeddings, then sets the diversity/relevance tradeoff accordingly.

**Q: What's the difference between a VAE and a regular autoencoder?**
A: A regular autoencoder maps each input to a single point in latent space (deterministic encoder). A VAE maps each input to a distribution (encoder outputs μ and σ). The KL divergence term forces this distribution to stay close to N(0,I). The result: (1) structured latent space — nearby z values decode to similar outputs; (2) generative ability — you can sample z from N(0,I) and decode to a realistic-looking output; (3) uncertainty quantification — σ tells you how uncertain the encoder is about the input.

**Q: Why β_max = 0.2 in Scholar's KL annealing, not 1.0?**
A: β < 1 is the β-VAE formulation. Lower β means less pressure on the posterior to match the prior, allowing more informative (but less regularized) latent codes. For recommendation, we care more about capturing precise user preferences (high information content in z) than having a perfectly smooth, interpolatable latent space. β=0.2 gives 80% weight to reconstruction and 20% to regularization — empirically tuned for recommendation tasks by Liang et al.

---

## 17. World-Class Resources

- **Original VAE paper:** Kingma & Welling (2014) "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
- **Mult-VAE paper:** Liang et al. (2018) "Variational Autoencoders for Collaborative Filtering" https://arxiv.org/abs/1802.05814
- **β-VAE paper:** Higgins et al. (2017) "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- **Tutorial on VAEs:** Lillicrap & Quer (2018) blog post on reparameterization https://gregorygundersen.com/blog/2018/04/29/reparameterization/
- **KL annealing:** Fu et al. (2019) "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing" https://arxiv.org/abs/1903.10145
- **Lilian Weng's VAE blog:** https://lilianweng.github.io/posts/2018-08-12-vae/ — one of the best VAE tutorials
- **CS 285 Berkeley RL** — covers VAEs in the context of generative models
