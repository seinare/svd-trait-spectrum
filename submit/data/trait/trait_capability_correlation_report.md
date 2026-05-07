# TRAIT-Capability First-Layer Correlation Analysis

Scope: four dense instruct models with complete nine-point TRAIT data: Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, and Qwen3-8B.

Response variable is `Delta TRAIT = TRAIT(alpha) - TRAIT(0)`. Predictors are the three fitted capability latent responses at the same model and alpha. The main pooled model uses standardized TRAIT deltas and standardized capability predictors, plus model fixed effects. Alpha=0 rows are excluded from regression because both sides are mechanically zero.

## Strongest Standardized Pooled Terms

| trait | capability | beta | approx t | R2 |
| --- | --- | ---: | ---: | ---: |
| Agreeableness | Factual Knowledge | 1.002 | 4.91 | 0.584 |
| Conscientiousness | Factual Knowledge | 0.743 | 3.13 | 0.437 |
| Conscientiousness | Language Understanding | -0.499 | -2.20 | 0.437 |
| Extraversion | Factual Knowledge | -0.492 | -1.85 | 0.292 |
| Extraversion | Language Understanding | 0.439 | 1.73 | 0.292 |
| Narcissism | Factual Knowledge | -0.398 | -1.41 | 0.205 |
| Openness | Factual Knowledge | 0.353 | 1.25 | 0.208 |
| Narcissism | Language Understanding | 0.348 | 1.29 | 0.205 |
| Narcissism | Deductive Reasoning | 0.343 | 1.71 | 0.205 |
| Machiavellianism | Deductive Reasoning | -0.331 | -1.63 | 0.183 |
| Openness | Deductive Reasoning | -0.316 | -1.58 | 0.208 |
| Agreeableness | Language Understanding | -0.310 | -1.59 | 0.584 |

## Files

- `pooled_trait_on_capability_regression.csv`: standardized pooled coefficients with model fixed effects.
- `trait_capability_pearson.csv`: simple pointwise correlations.
- `per_model_trait_on_capability_regression.csv`: raw within-model OLS coefficients.
- `leave_one_model_out_rmse.csv`: leave-one-model-out predictive RMSE on TRAIT deltas.