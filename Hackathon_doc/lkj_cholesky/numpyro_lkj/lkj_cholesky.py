import numpy as np
import warnings
import jax
from jax import lax, vmap
from jax.experimental.sparse import BCOO
from jax.lax import scan
import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import (
    betaln,
    expi,
    expit,
    gammainc,
    gammaln,
    logit,
    multigammaln,
    ndtr,
    ndtri,
    xlog1py,
    xlogy,
)
from jax.scipy.stats import norm as jax_norm

from numpyro.distributions import constraints
from numpyro.distributions.discrete import _to_logits_bernoulli
from numpyro.distributions.distribution import Distribution, TransformedDistribution
from numpyro.distributions.transforms import (
    AffineTransform,
    CorrMatrixCholeskyTransform,
    ExpTransform,
    PowerTransform,
    SigmoidTransform,
)
from numpyro.distributions.util import (
    betainc,
    betaincinv,
    cholesky_of_inverse,
    gammaincinv,
    lazy_property,
    promote_shapes,
    validate_sample,
    vec_to_tril_matrix,
)
# from numpyro.util import is_prng_key


def signed_stick_breaking_tril(t):
    # make sure that t in (-1, 1)
    eps = jnp.finfo(t.dtype).eps
    t = jnp.clip(t, a_min=(-1 + eps), a_max=(1 - eps))
    # transform t to tril matrix with identity diagonal
    
    r = vec_to_tril_matrix(t, diagonal=-1)

    # apply stick-breaking on the squared values;
    # we omit the step of computing s = z * z_cumprod by using the fact:
    #     y = sign(r) * s = sign(r) * sqrt(z * z_cumprod) = r * sqrt(z_cumprod)
    z = r**2
    # print("signed_stick_breaking_tril   t\n", t)
    print("signed_stick_breaking_tril   r\n", r)
    # print("signed_stick_breaking_tril   z\n", z)
    z1m_cumprod_sqrt = jnp.cumprod(jnp.sqrt(1 - z), axis=-1)
    # print("signed_stick_breaking_tril   z\n", jnp.sqrt(1 - z))
    # print("signed_stick_breaking_tril   z\n", z1m_cumprod_sqrt)
    print("signed_stick_breaking_tril   z\n", add_diag(r, 1))

    pad_width = [(0, 0)] * z.ndim
    pad_width[-1] = (1, 0)
    z1m_cumprod_sqrt_shifted = jnp.pad(
        z1m_cumprod_sqrt[..., :-1], pad_width, mode="constant", constant_values=1.0
    )
    
    print("signed_stick_breaking_tril   z\n", z1m_cumprod_sqrt)
    print("signed_stick_breaking_tril   z\n", z1m_cumprod_sqrt_shifted)
    
    y = add_diag(r, 1) * z1m_cumprod_sqrt_shifted
    return y



def matrix_to_tril_vec(x, diagonal=0):
    # print("matrix_to_tril_vec:", x)
    idxs = jnp.tril_indices(x.shape[-1], diagonal)
    # print("matrix_to_tril_vec:", x[..., idxs[0], idxs[1]])
    return x[..., idxs[0], idxs[1]]

def add_diag(matrix: jnp.ndarray, diag: jnp.ndarray) -> jnp.ndarray:
    """
    Add `diag` to the trailing diagonal of `matrix`.
    """
    idx = jnp.arange(matrix.shape[-1])
    return matrix.at[..., idx, idx].add(diag)

# def is_prng_key(key):
#     warnings.warn("Please use numpyro.util.is_prng_key.", DeprecationWarning)
#     try:
#         if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
#             return key.shape == ()
#         return key.shape == (2,) and key.dtype == np.uint32
#     except AttributeError:
#         return False

class Dirichlet(Distribution):
    arg_constraints = {
        "concentration": constraints.independent(constraints.positive, 1)
    }
    reparametrized_params = ["concentration"]
    support = constraints.simplex

    def __init__(self, concentration, *, validate_args=None):
        if jnp.ndim(concentration) < 1:
            raise ValueError(
                "`concentration` parameter must be at least one-dimensional."
            )
        self.concentration = concentration
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        super(Dirichlet, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        # assert is_prng_key(key)
        shape = sample_shape + self.batch_shape
        samples = random.dirichlet(key, self.concentration, shape=shape)
        return jnp.clip(
            samples, a_min=jnp.finfo(samples).tiny, a_max=1 - jnp.finfo(samples).eps
        )

    @validate_sample
    def log_prob(self, value):
        normalize_term = jnp.sum(gammaln(self.concentration), axis=-1) - gammaln(
            jnp.sum(self.concentration, axis=-1)
        )
        return (
            jnp.sum(jnp.log(value) * (self.concentration - 1.0), axis=-1)
            - normalize_term
        )

    @property
    def mean(self):
        return self.concentration / jnp.sum(self.concentration, axis=-1, keepdims=True)

    @property
    def variance(self):
        con0 = jnp.sum(self.concentration, axis=-1, keepdims=True)
        return self.concentration * (con0 - self.concentration) / (con0**2 * (con0 + 1))

    @staticmethod
    def infer_shapes(concentration):
        batch_shape = concentration[:-1]
        event_shape = concentration[-1:]
        return batch_shape, event_shape

class Beta(Distribution):
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    reparametrized_params = ["concentration1", "concentration0"]
    support = constraints.unit_interval
    pytree_data_fields = ("concentration0", "concentration1", "_dirichlet")

    def __init__(self, concentration1, concentration0, *, validate_args=None):
        self.concentration1, self.concentration0 = promote_shapes(
            concentration1, concentration0
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration1), jnp.shape(concentration0)
        )
        concentration1 = jnp.broadcast_to(concentration1, batch_shape)
        concentration0 = jnp.broadcast_to(concentration0, batch_shape)
        super(Beta, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
        self._dirichlet = Dirichlet(
            jnp.stack([concentration1, concentration0], axis=-1)
        )

    def sample(self, key, sample_shape=()):
        # assert is_prng_key(key)
        return self._dirichlet.sample(key, sample_shape)[..., 0]

    @validate_sample
    def log_prob(self, value):
        return self._dirichlet.log_prob(jnp.stack([value, 1.0 - value], -1))

    @property
    def mean(self):
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def variance(self):
        total = self.concentration1 + self.concentration0
        return self.concentration1 * self.concentration0 / (total**2 * (total + 1))

    def cdf(self, value):
        return betainc(self.concentration1, self.concentration0, value)

    def icdf(self, q):
        return betaincinv(self.concentration1, self.concentration0, q)

class LKJCholesky(Distribution):
    r"""
    LKJ distribution for lower Cholesky factors of correlation matrices. The distribution is
    controlled by ``concentration`` parameter :math:`\eta` to make the probability of the
    correlation matrix :math:`M` generated from a Cholesky factor propotional to
    :math:`\det(M)^{\eta - 1}`. Because of that, when ``concentration == 1``, we have a
    uniform distribution over Cholesky factors of correlation matrices.

    When ``concentration > 1``, the distribution favors samples with large diagonal entries
    (hence large determinent). This is useful when we know a priori that the underlying
    variables are not correlated.

    When ``concentration < 1``, the distribution favors samples with small diagonal entries
    (hence small determinent). This is useful when we know a priori that some underlying
    variables are correlated.

    Sample code for using LKJCholesky in the context of multivariate normal sample::

        def model(y):  # y has dimension N x d
            d = y.shape[1]
            N = y.shape[0]
            # Vector of variances for each of the d variables
            theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))
            # Lower cholesky factor of a correlation matrix
            concentration = jnp.ones(1)  # Implies a uniform distribution over correlation matrices
            L_omega = numpyro.sample("L_omega", dist.LKJCholesky(d, concentration))
            # Lower cholesky factor of the covariance matrix
            sigma = jnp.sqrt(theta)
            # we can also use a faster formula `L_Omega = sigma[..., None] * L_omega`
            L_Omega = jnp.matmul(jnp.diag(sigma), L_omega)

            # Vector of expectations
            mu = jnp.zeros(d)

            with numpyro.plate("observations", N):
                obs = numpyro.sample("obs", dist.MultivariateNormal(mu, scale_tril=L_Omega), obs=y)
            return obs

    :param int dimension: dimension of the matrices
    :param ndarray concentration: concentration/shape parameter of the
        distribution (often referred to as eta)
    :param str sample_method: Either "cvine" or "onion". Both methods are proposed in [1] and
        offer the same distribution over correlation matrices. But they are different in how
        to generate samples. Defaults to "onion".

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method`,
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe
    """

    arg_constraints = {"concentration": constraints.positive}
    reparametrized_params = ["concentration"]
    support = constraints.corr_cholesky
    pytree_data_fields = ("_beta", "concentration")
    pytree_aux_fields = ("dimension", "sample_method")

    def __init__(
        self, dimension, concentration=1.0, sample_method="onion", *, validate_args=None
    ):
        if dimension < 2:
            raise ValueError("Dimension must be greater than or equal to 2.")
        self.dimension = dimension
        self.concentration = concentration
        batch_shape = jnp.shape(concentration)
        event_shape = (dimension, dimension)

        # We construct base distributions to generate samples for each method.
        # The purpose of this base distribution is to generate a distribution for
        # correlation matrices which is propotional to `det(M)^{\eta - 1}`.
        # (note that this is not a unique way to define base distribution)
        # Both of the following methods have marginal distribution of each off-diagonal
        # element of sampled correlation matrices is Beta(eta + (D-2) / 2, eta + (D-2) / 2)
        # (up to a linear transform: x -> 2x - 1)
        Dm1 = self.dimension - 1
        marginal_concentration = concentration + 0.5 * (self.dimension - 2)
        offset = 0.5 * jnp.arange(Dm1)
        if sample_method == "onion":
            # The following construction follows from the algorithm in Section 3.2 of [1]:
            # NB: in [1], the method for case k > 1 can also work for the case k = 1.
            beta_concentration0 = (
                jnp.expand_dims(marginal_concentration, axis=-1) - offset
            )
            beta_concentration1 = offset + 0.5
            self._beta = Beta(beta_concentration1, beta_concentration0)
        elif sample_method == "cvine":
            # The following construction follows from the algorithm in Section 2.4 of [1]:
            # offset_tril is [0, 1, 1, 2, 2, 2,...] / 2
            offset_tril = matrix_to_tril_vec(jnp.broadcast_to(offset, (Dm1, Dm1)))
            beta_concentration = (
                jnp.expand_dims(marginal_concentration, axis=-1) - offset_tril
            )
            # print(offset_tril.shape)
            print(beta_concentration)
            self._beta = Beta(beta_concentration, beta_concentration)
        else:
            raise ValueError("`method` should be one of 'cvine' or 'onion'.")
        self.sample_method = sample_method

        super(LKJCholesky, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def _cvine(self, key, size):
        # C-vine method first uses beta_dist to generate partial correlations,
        # then apply signed stick breaking to transform to cholesky factor.
        # Here is an attempt to prove that using signed stick breaking to
        # generate correlation matrices is the same as the C-vine method in [1]
        # for the entry r_32.
        #
        # With notations follow from [1], we define
        #   p: partial correlation matrix,
        #   c: cholesky factor,
        #   r: correlation matrix.
        # From recursive formula (2) in [1], we have
        #   r_32 = p_32 * sqrt{(1 - p_21^2)*(1 - p_31^2)} + p_21 * p_31 =: I
        # On the other hand, signed stick breaking process gives:
        #   l_21 = p_21, l_31 = p_31, l_22 = sqrt(1 - p_21^2), l_32 = p_32 * sqrt(1 - p_31^2)
        #   r_32 = l_21 * l_31 + l_22 * l_32
        #        = p_21 * p_31 + p_32 * sqrt{(1 - p_21^2)*(1 - p_31^2)} = I
        beta_sample = self._beta.sample(key, size)
        partial_correlation = 2 * beta_sample - 1  # scale to domain to (-1, 1)
        # print("beta_sample", beta_sample)
        # print("partial_correlation", partial_correlation)
        return signed_stick_breaking_tril(partial_correlation)

    def _onion(self, key, size):
        key_beta, key_normal = random.split(key)
        # Now we generate w term in Algorithm 3.2 of [1].
        beta_sample = self._beta.sample(key_beta, size)
        # The following Normal distribution is used to create a uniform distribution on
        # a hypershere (ref: http://mathworld.wolfram.com/HyperspherePointPicking.html)
        print("numpyro:         \n", beta_sample)
        
        normal_sample = random.normal(
            key_normal,
            shape=size
            + self.batch_shape
            + (self.dimension * (self.dimension - 1) // 2,),
        )
        normal_sample = vec_to_tril_matrix(normal_sample, diagonal=0)
        u_hypershere = normal_sample / jnp.linalg.norm(
            normal_sample, axis=-1, keepdims=True
        )
        w = jnp.expand_dims(jnp.sqrt(beta_sample), axis=-1) * u_hypershere

        # put w into the off-diagonal triangular part
        cholesky = jnp.zeros(size + self.batch_shape + self.event_shape)
        
        cholesky = cholesky.at[..., 1:, :-1].set(w)
        # correct the diagonal
        # NB: beta_sample = sum(w ** 2) because norm 2 of u is 1.
        diag = jnp.ones(cholesky.shape[:-1]).at[..., 1:].set(jnp.sqrt(1 - beta_sample))
        print("numpyro:         \n", jnp.sqrt(1 - beta_sample))
        print("numpyro:         \n", jnp.ones(cholesky.shape[:-1]).at[..., 1:])
        print("numpyro:         \n", beta_sample)
        return add_diag(cholesky, diag)

    def sample(self, key, sample_shape=()):
        # assert is_prng_key(key)
        if self.sample_method == "onion":
            return self._onion(key, sample_shape)
        else:
            return self._cvine(key, sample_shape)

    @validate_sample
    def log_prob(self, value):
        # Note about computing Jacobian of the transformation from Cholesky factor to
        # correlation matrix:
        #
        #   Assume C = L@Lt and L = (1 0 0; a \sqrt(1-a^2) 0; b c \sqrt(1-b^2-c^2)), we have
        #   Then off-diagonal lower triangular vector of L is transformed to the off-diagonal
        #   lower triangular vector of C by the transform:
        #       (a, b, c) -> (a, b, ab + c\sqrt(1-a^2))
        #   Hence, Jacobian = 1 * 1 * \sqrt(1 - a^2) = \sqrt(1 - a^2) = L22, where L22
        #       is the 2th diagonal element of L
        #   Generally, for a D dimensional matrix, we have:
        #       Jacobian = L22^(D-2) * L33^(D-3) * ... * Ldd^0
        #
        # From [1], we know that probability of a correlation matrix is propotional to
        #   determinant ** (concentration - 1) = prod(L_ii ^ 2(concentration - 1))
        # On the other hand, Jabobian of the transformation from Cholesky factor to
        # correlation matrix is:
        #   prod(L_ii ^ (D - i))
        # So the probability of a Cholesky factor is propotional to
        #   prod(L_ii ^ (2 * concentration - 2 + D - i)) =: prod(L_ii ^ order_i)
        # with order_i = 2 * concentration - 2 + D - i,
        # i = 2..D (we omit the element i = 1 because L_11 = 1)

        # Compute `order` vector (note that we need to reindex i -> i-2):
        one_to_D = jnp.arange(1, self.dimension)
        order_offset = (3 - self.dimension) + one_to_D
        order = 2 * jnp.expand_dims(self.concentration, axis=-1) - order_offset

        # Compute unnormalized log_prob:
        value_diag = jnp.asarray(value)[..., one_to_D, one_to_D]
        unnormalized = jnp.sum(order * jnp.log(value_diag), axis=-1)

        # Compute normalization constant (on the first proof of page 1999 of [1])
        Dm1 = self.dimension - 1
        alpha = self.concentration + 0.5 * Dm1
        denominator = gammaln(alpha) * Dm1
        numerator = multigammaln(alpha - 0.5, Dm1)
        # pi_constant in [1] is D * (D - 1) / 4 * log(pi)
        # pi_constant in multigammaln is (D - 1) * (D - 2) / 4 * log(pi)
        # hence, we need to add a pi_constant = (D - 1) * log(pi) / 2
        pi_constant = 0.5 * Dm1 * jnp.log(jnp.pi)
        normalize_term = pi_constant + numerator - denominator
        return unnormalized - normalize_term
