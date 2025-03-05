from tokenizer.char_tokenizer import CharTokenizer
import mlx.core as mx
import mlx.nn as nn
import mlx.nn.losses as losses
import mlx.optimizers as optimizers
import numpy as np

with open("datasets/tiny_shakespeare.txt", "r") as f:
    text = f.read()

# all the hyperparameters
data_size = len(text)
train_size = int(0.8 * data_size)
context_window = 8
batch_size = 32
n_eval_steps = 200
n_train_steps = 10000
eval_interval = 500

tokenizer = CharTokenizer(text)

data = mx.array(tokenizer.encode(text), dtype=mx.int32)
train_data = data[:train_size]
val_data = data[train_size:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    # get random starting indices for each batch
    starting_indices_batch = mx.random.randint(
        0, len(train_data) - context_window, (batch_size,)
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
    return mx.array(indices, dtype=mx.int32).reshape(1, 1)


# bigram model
class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # What is accomplished using nn.Embedding?
        # It creates a lookup table for the tokens in the vocabulary.
        # The table has vocab_size rows and vocab_size columns.
        # The row index is the token id and the column index is the embedding dimension.
        # The value at position (i, j) is the j-th embedding of the i-th token.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def __call__(self, idx, target=None):
        logits = self.token_embedding_table(idx)

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
            logits, _ = self(idx)
            logits = logits[:, -1:, :]  # (B, C)
            probs = nn.softmax(logits)
            idx_next = multinomial_sampling(probs, num_samples=1)
            idx = mx.concatenate((idx, idx_next), axis=1)
        return idx


# MLX requires a stand-alone, explicit loss function to be defined
def loss_fn(model, xb, yb):
    _, loss = model(xb, yb)
    return loss


def estimated_loss(model):
    out = {}
    model.eval()
    # with mx.no_grad() is equivalent to torch.no_grad() context manager
    with mx.stop_gradient():
        for split in ["train", "val"]:
            eval_losses = mx.zeros(n_eval_steps)
            for k in range(n_eval_steps):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                eval_losses[k] = loss.item()
            out[split] = eval_losses.mean()
    model.train()
    return out


m = Bigram(vocab_size=tokenizer.vocab_size)

# initialize the generation
init_tokens = mx.zeros((1, 1), dtype=mx.int32)
tokens = m.generate(init_tokens, max_new_tokens=500)
print(tokenizer.decode(tokens[0].tolist()))


# get the training batch (input and target)
xb, yb = get_batch("train")

# create optimizer
optimizer = optimizers.AdamW(learning_rate=1e-3)
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


# print(loss)

# initialize the generation
print("_______________________")
init_tokens = mx.zeros((1, 1), dtype=mx.int32)

tokens = m.generate(init_tokens, max_new_tokens=500)

print(tokenizer.decode(tokens[0].tolist()))
