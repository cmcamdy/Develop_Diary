from lkj_cholesky import LKJCholesky
import torch_lkj.lkj_cholesky as torch_lkj
import numpyro_lkj.lkj_cholesky as np_lkj
import paddle
import torch
import numpyro
import numpy as np
import jax
from jax import random

def matrix_to_tril_vec(x, diagonal=0):
    print(x)
    idxs = jnp.tril_indices(x.shape[-1], diagonal)
    return x[..., idxs[0], idxs[1]]


def numpyro_cvine(dimension, concentration):
    dm1 = dimension - 1
    offset = 0.5 * jnp.arange(dm1)
    offset_tril = matrix_to_tril_vec(jnp.broadcast_to(offset, (dm1, dm1)))
    print("numpyro_cvine:", dm1, offset, offset_tril)


def numpyro_onion(dimension, concentration):
    dm1 = dimension - 1
    offset = 0.5 * jnp.arange(dm1)
    marginal_concentration = concentration + 0.5 * (dimension - 2)
    
    offset_tril = matrix_to_tril_vec(jnp.broadcast_to(offset, (dm1, dm1)))
    beta_concentration0 = jnp.expand_dims(marginal_concentration, axis=-1) - offset
    beta_concentration1 = offset + 0.5
    # print("numpyro_onion:", dm1, offset, offset_tril, beta_concentration0, beta_concentration1)
    print("numpyro_onion:", marginal_concentration)
    print("numpyro_onion:", beta_concentration0, beta_concentration1)




def torch_onion(dim, concentration):
    (concentration,) = broadcast_all(concentration)
    batch_shape = concentration.size()
    event_shape = torch.Size((dim, dim))
    # This is used to draw vectorized samples from the beta distribution in Sec. 3.2 of [1].
    marginal_conc = concentration + 0.5 * (dim - 2)
    offset = torch.arange(
        dim - 1,
        dtype=concentration.dtype,
        device=concentration.device,
    )
    offset = torch.cat([offset.new_zeros((1,)), offset])
    beta_conc1 = offset + 0.5
    beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset
    print("torch:", marginal_conc)
    print("torch:", offset)
    print("torch:", beta_conc0, beta_conc1)

if __name__ == "__main__":
    
    # PyTorch 随机数种子
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)  # 如果使用多个GPU
    # Python 随机数种子
    # random.seed(0)
    # Numpy 随机数种子
    np.random.seed(0)
    paddle.seed(0)
    # 设置JAX全局随机数种子
    key = random.PRNGKey(0)
    dim = 4
    sample_method = "onion"    
    sample_method = "cvine"    
    
    # torch_l = torch_lkj.LKJCholesky(dimension)
    np_l = np_lkj.LKJCholesky(dimension, sample_method = sample_method)
    # lkj = LKJCholesky(dimension, sample_method = sample_method)
    lkj = LKJCholesky(dim = dim)

    n = np_l.sample(key = key)
    p = paddle_l.sample()
    # t = torch_l.sample()
    
    # print(n)
    # print(p)    
    # print(t)
    
    # tb = torch.distributions.Beta(torch.tensor([0.5]), torch.tensor([0.5]))
    # pb = paddle.distribution.Beta(paddle.to_tensor([0.5]), paddle.to_tensor([0.5]))
    # print(tb.mean, pb.mean)
    # print(tb.variance, pb.variance)
    # print(tb.entropy(), pb.entropy())