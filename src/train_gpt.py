from tokenizer.char_tokenizer import CharTokenizer
from mlx.utils import tree_flatten
import mlx.core as mx
import mlx.nn as nn
import mlx.nn.losses as losses
import mlx.optimizers as optimizers
import numpy as np
import json

mx.default_stream(mx.gpu)
# Load config
with open("train_config.json", "r") as f:
    config = json.load(f)

# Load and prepare data
with open(config["data"]["dataset_path"], "r") as f:
    text = f.read()

# Extract hyperparameters from config
data_size = len(text)
train_size = int(config["data"]["train_split"] * data_size)
context_window = config["model"]["context_window"]
batch_size = config["model"]["batch_size"]
n_eval_steps = config["training"]["n_eval_steps"]
n_train_steps = config["training"]["n_train_steps"]
eval_interval = config["training"]["eval_interval"]
learning_rate = config["training"]["learning_rate"]
max_new_tokens = config["generation"]["max_new_tokens"]
num_heads = config["model"]["num_heads"]
num_blocks = config["model"]["num_blocks"]
dropout_rate = config["model"]["dropout"]
tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size
embedding_dim = config["model"]["embedding_dim"]
data = mx.array(tokenizer.encode(text), dtype=mx.int16)
train_data = data[:train_size]
val_data = data[train_size:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    # get random starting indices for each batch
    starting_indices_batch = mx.random.randint(
        0, len(data) - context_window, (batch_size,)
    )
    starting_indices_batch = [int(i) for i in starting_indices_batch]
    x = mx.stack([data[i : i + context_window] for i in starting_indices_batch])
    y = mx.stack([data[i + 1 : i + context_window + 1] for i in starting_indices_batch])
    return x, y


def multinomial_sampling(probs, num_samples):
    probs = probs / mx.sum(probs)  # Step 1: Normalize
    cum_probs = mx.cumsum(probs).tolist()  # Step 2: Compute CDF
    # Step 3: Generate random numbers
    random_samples = np.random.rand(num_samples)
    indices = np.searchsorted(cum_probs, random_samples)  # Step 4: Find indices
    return mx.array(indices, dtype=mx.int16).reshape(1, 1)


# bigram model
class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        # What is accomplished using nn.Embedding?
        # It creates a lookup table for the tokens in the vocabulary.
        # The table has vocab_size rows and vocab_size columns.
        # The row index is the token id and the column index is the embedding dimension.
        # The value at position (i, j) is the j-th embedding of the i-th token.
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(context_window, embedding_dim)
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, num_heads) for _ in range(num_blocks)]
        )
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def __call__(self, idx, target=None):
        B, T = idx.shape

        # fetches the token embeddings based on the token index idx
        tok_emb = self.token_embedding_table(idx)  # ( B, T, C)

        # fetches the position embeddings based on the position indices of the tokens
        pos_emb = self.position_embedding_table(mx.arange(T))  # (T, C)

        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if target is None:
            loss = None
        else:
            # reshaped loss
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            target = target.reshape(B * T)
            loss = losses.cross_entropy(logits, target).mean()

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            """
            we are clearing the context window here in case the
            the generated text is longer than the context window.
            So we are only keeping the last context_window tokens in the input.
            """
            idx_cond = idx[:, -context_window:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1:, :]  # (B, C)
            probs = nn.softmax(logits)
            idx_next = multinomial_sampling(probs, num_samples=1)
            idx = mx.concatenate((idx, idx_next), axis=1)
        return idx


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        # In MLX, we can store non-parameter tensors directly as instance variables
        self.causal_mask = None  # Will be initialized in forward pass

    def __call__(self, x):
        _, T, _ = x.shape
        if self.causal_mask is None or self.causal_mask.shape[0] != T:
            self.causal_mask = mx.tril(mx.ones((T, T)))

        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        wei = q @ k.swapaxes(-2, -1) * self.head_size**-0.5  # (B, T, T)
        wei = mx.where(self.causal_mask == 0, float("-inf"), wei)
        wei = mx.softmax(wei, axis=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        attention_matrix = mx.matmul(wei, v)
        return attention_matrix


class MultiHeadAttention(nn.Module):

    """
    Create multiple heads for the attention mechanism.
    Head size should be an integer divisor of the embedding dimension.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def __call__(self, x):
        mhsa = mx.concatenate([head(x) for head in self.heads], axis=-1)
        mhsa = self.proj(mhsa)
        mhsa = self.dropout(mhsa)
        return mhsa


class FeedForward(nn.Module):
    """
    Simple feed-forward network with fixed dimension projection
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout_rate),
        )

    def __call__(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """

    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ff = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def __call__(self, x):
        x += self.sa(self.ln1(x))  # residual connection with LayerNorm applied
        x += self.ff(self.ln2(x))  # residual connection with LayerNorm applied
        return x


class LayerNorm1D:
    def __init__(self, embedding_dim, eps=1e-5):
        self.eps = eps
        self.gamma = mx.ones(embedding_dim)
        self.beta = mx.zeros(embedding_dim)

    def __call__(self, x):
        mean = mx.mean(x, axis=1, keepdims=True)
        var = mx.var(x, axis=1, keepdims=True)

        xhat = (x - mean) / mx.sqrt(var + self.eps)  # zero mean unit variance
        return self.gamma * xhat + self.beta


# MLX requires a stand-alone, explicit loss function to be defined
def loss_fn(model, xb, yb):
    _, loss = model(xb, yb)
    return loss


def estimated_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        eval_losses = mx.zeros(n_eval_steps)
        for k in range(n_eval_steps):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            eval_losses[k] = loss.item()
        out[split] = eval_losses.mean()
    model.train()
    return out


def count_parameters(model):
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    return num_params


m = Bigram()
print(f"Total parameters: {count_parameters(m):,}")


# initialize the generation
init_tokens = mx.zeros((1, 1), dtype=mx.int16)
tokens = m.generate(init_tokens, max_new_tokens=max_new_tokens)
# print(tokenizer.decode(tokens[0].tolist()))


# get the training batch (input and target)
xb, yb = get_batch("train")

# create optimizer
optimizer = optimizers.AdamW(learning_rate=learning_rate)
# loss and gradient function
loss_and_grad = nn.value_and_grad(m, loss_fn)


for _ in range(n_train_steps):
    # sample a batch
    xb, yb = get_batch("train")

    # evaluate the loss and gradient
    loss, grad = loss_and_grad(m, xb, yb)

    # update model parameters
    optimizer.update(m, grad)

    if _ % eval_interval == 0:
        eval_loss = estimated_loss(m)
        print(
            f"step {_} train loss: {eval_loss['train']:.4f}, val loss: {eval_loss['val']:.4f}"
        )


print("_______________________")
init_tokens = mx.zeros((1, 1), dtype=mx.int16)

tokens = m.generate(init_tokens, max_new_tokens=max_new_tokens)

print(tokenizer.decode(tokens[0].tolist()))
