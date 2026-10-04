# Diagnosing a Flawed Risk Model — VaR, Expected Shortfall & Monte Carlo

## Objective
I checked a junior analyst's Value at Risk model, found why it understates tail risk, and compared it with better ways to measure risk.

## Methodology
- Looked at the descriptive statistics of 2,520 daily returns and found an excess kurtosis of 4.28, which means the tails are much fatter than a normal distribution.
- Compared the analyst's normal VaR with the historical VaR at 95% and 99% on a $10M portfolio.
- Computed 99% VaR and Expected Shortfall under normal, Student-t (fitted df = 4.58) and historical methods.
- Priced a European call option with naive Monte Carlo and with antithetic variates, and compared both with the Black-Scholes price.
- Used a risk_metrics.py module (calculate_var, calculate_es, mc_var) and ran its self-tests.
- Had an AI write a VaR backtest function, revised my prompt once, and checked the result against my own count.

## Key Findings
- At 99%, the normal VaR understated the historical VaR by 12.7% ($40,393 on the portfolio). At 95% it was slightly too large, so a 95% check would have made the model look safe.
- The Student-t gave a VaR close to the historical one and the highest Expected Shortfall, so it handles fat tails better than the normal.
- Antithetic variates cut the Monte Carlo standard error by 1.26x with the same number of paths.
- The normal 99% VaR was breached on 1.71% of days instead of the 1% it promises.
