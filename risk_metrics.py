"""VaR and Expected Shortfall under historical, normal and Student-t assumptions."""

import numpy as np
from scipy import stats

VALID_METHODS = ('historical', 'normal', 't')
VALID_DISTS = ('normal', 't')


def calculate_var(returns: np.ndarray, confidence: float = 0.99,
                  method: str = 'historical') -> float:
    """Value at Risk as a positive loss, by 'historical', 'normal' or 't'."""
    assert method in VALID_METHODS, f"method must be one of {VALID_METHODS}"
    alpha = 1 - confidence

    if method == 'historical':
        return float(-np.percentile(returns, alpha * 100))
    elif method == 'normal':
        mu = returns.mean()
        sigma = returns.std(ddof=1)
        return float(-(mu + stats.norm.ppf(alpha) * sigma))
    else:
        df_hat, loc_hat, scale_hat = stats.t.fit(returns)
        return float(-stats.t.ppf(alpha, df=df_hat, loc=loc_hat, scale=scale_hat))


def calculate_es(returns: np.ndarray, confidence: float = 0.99,
                 method: str = 'historical') -> float:
    """Expected Shortfall (average loss at or beyond the VaR), by the same three methods."""
    assert method in VALID_METHODS, f"method must be one of {VALID_METHODS}"
    alpha = 1 - confidence

    if method == 'historical':
        var = calculate_var(returns, confidence, method)
        return float(-returns[returns <= -var].mean())
    elif method == 'normal':
        mu = returns.mean()
        sigma = returns.std(ddof=1)
        return float(-(mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha))
    else:
        # Simulated, with the same seed and sample size as Part 2
        df_hat, loc_hat, scale_hat = stats.t.fit(returns)
        rng = np.random.default_rng(seed=42)
        samples = stats.t.rvs(df=df_hat, loc=loc_hat, scale=scale_hat,
                              size=500_000, random_state=rng)
        t_quantile = stats.t.ppf(alpha, df=df_hat, loc=loc_hat, scale=scale_hat)
        return float(-samples[samples <= t_quantile].mean())


def mc_var(mu: float, sigma: float, n_sims: int = 100_000,
           confidence: float = 0.99, dist: str = 'normal',
           df: float = 5.0, seed: int = 42) -> dict:
    """Simulate returns with this mean and std; return {'var': ..., 'es': ...}."""
    assert dist in VALID_DISTS, f"dist must be one of {VALID_DISTS}"
    rng = np.random.default_rng(seed=seed)
    alpha = 1 - confidence

    if dist == 'normal':
        sim_returns = rng.normal(loc=mu, scale=sigma, size=n_sims)
    else:
        # A t with df (above 2) degrees of freedom has std sqrt(df / (df - 2)); rescale it to sigma
        scale_t = sigma * np.sqrt((df - 2) / df)
        sim_returns = mu + scale_t * rng.standard_t(df=df, size=n_sims)

    var = float(-np.percentile(sim_returns, alpha * 100))
    es = float(-sim_returns[sim_returns <= -var].mean())
    return {'var': var, 'es': es}


if __name__ == "__main__":
    # Self-tests on normal returns, where every method should roughly agree
    rng_test = np.random.default_rng(seed=42)
    test_returns = rng_test.normal(loc=0.0004, scale=0.012, size=5000)

    print("risk_metrics.py self-test")
    for method in VALID_METHODS:
        var = calculate_var(test_returns, 0.99, method)
        es = calculate_es(test_returns, 0.99, method)
        print(f"  {method:>12s}  VaR={var:.4%}  ES={es:.4%}")
        assert es >= var, f"ES must be >= VaR for method={method}"

    mc_normal = mc_var(mu=0.0004, sigma=0.012, n_sims=100_000, confidence=0.99)
    print(f"  MC (normal): VaR={mc_normal['var']:.4%}  ES={mc_normal['es']:.4%}")
    assert mc_normal['es'] >= mc_normal['var']

    mc_t = mc_var(mu=0.0004, sigma=0.012, n_sims=100_000,
                  confidence=0.99, dist='t', df=5)
    print(f"  MC (t, df=5): VaR={mc_t['var']:.4%}  ES={mc_t['es']:.4%}")
    assert mc_t['var'] > mc_normal['var'], "t-dist VaR should exceed normal VaR"

    print("All self-tests passed.")
