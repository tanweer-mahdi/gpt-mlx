from mlx.utils import tree_flatten
import numpy as np
import mlx.core as mx


def multinomial_sampling(probs, num_samples):
    probs = probs / mx.sum(probs)  # Step 1: Normalize
    cum_probs = mx.cumsum(probs).tolist()  # Step 2: Compute CDF
    # Step 3: Generate random numbers
    random_samples = np.random.rand(num_samples)
    indices = np.searchsorted(cum_probs, random_samples)  # Step 4: Find indices
    return mx.array(indices, dtype=mx.int16).reshape(1, 1)


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


def count_parameters(model):
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    return num_params
