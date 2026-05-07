# Alpha SVD Spectrum Statistics

This report is data-first. It summarizes singular-value distribution changes induced by the implemented alpha transform on MLP `up_proj` and `down_proj` weights.

Transform:

```text
s_i' = G * (s_i / G)^(1 + alpha)
G = exp(mean_i log s_i)
```

The transform preserves the geometric mean of singular values. Positive alpha increases spectral inequality; negative alpha smooths it.

## Available Models

| model |
| --- |
| Llama 3.1 8B Instruct |
| Llama 3.2 1B Instruct |
| Llama 3.2 3B Instruct |
| Qwen3 30B-A3B MoE |
| Qwen3 8B |

## Output Tables

- Matrix-level full table: `docs/results/svd_alpha_spectrum/combined_matrix_alpha_svd_stats.csv`
- Layer-level summary: `docs/results/svd_alpha_spectrum/combined_layer_alpha_svd_summary.csv`
- Depth-bucket summary: `docs/results/svd_alpha_spectrum/combined_depth_alpha_svd_summary.csv`
- Selected depth table: `docs/results/svd_alpha_spectrum/table_depth_alpha_svd_summary_selected.csv`
- Top positive-alpha Gini increases: `docs/results/svd_alpha_spectrum/table_top_delta_gini_alpha_pos02.csv`
- Top negative-alpha Gini decreases: `docs/results/svd_alpha_spectrum/table_top_delta_gini_alpha_neg02.csv`
- Top max-singular-value relative increases: `docs/results/svd_alpha_spectrum/table_top_sv_rel_alpha_pos02.csv`
- MoE expert-max Gini increases: `docs/results/svd_alpha_spectrum/table_moe_expert_max_delta_gini_alpha_pos02.csv`
- MoE expert-max top singular value relative increases: `docs/results/svd_alpha_spectrum/table_moe_expert_max_top_sv_rel_alpha_pos02.csv`
- Late-minus-early base spectrum table: `docs/results/svd_alpha_spectrum/table_late_minus_early_base_spectrum.csv`

## Late vs Early Base Spectrum

| model | proj | agg | early Gini | late Gini | late-early Gini | early top/G | late top/G | late-early top/G |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama 3.1 8B Instruct | down_proj | dense | 0.2012 | 0.1989 | -0.0023 | 4.6017 | 3.0415 | -1.5602 |
| Llama 3.1 8B Instruct | up_proj | dense | 0.1852 | 0.1927 | 0.0075 | 2.9769 | 4.7986 | 1.8217 |
| Llama 3.2 1B Instruct | down_proj | dense | 0.1910 | 0.2027 | 0.0116 | 3.8644 | 2.8862 | -0.9783 |
| Llama 3.2 1B Instruct | up_proj | dense | 0.1726 | 0.2093 | 0.0366 | 3.1799 | 4.3885 | 1.2086 |
| Llama 3.2 3B Instruct | down_proj | dense | 0.2227 | 0.2274 | 0.0046 | 4.4082 | 3.2492 | -1.1590 |
| Llama 3.2 3B Instruct | up_proj | dense | 0.2066 | 0.2199 | 0.0132 | 2.9669 | 5.2773 | 2.3104 |
| Qwen3 30B-A3B MoE | down_proj | moe_expert_mean_max | 0.2348 | 0.2269 | -0.0080 | 3.7861 | 3.3018 | -0.4843 |
| Qwen3 30B-A3B MoE | up_proj | moe_expert_mean_max | 0.2328 | 0.2200 | -0.0128 | 2.8082 | 3.0488 | 0.2406 |
| Qwen3 8B | down_proj | dense | 0.2310 | 0.2037 | -0.0273 | 6.8302 | 3.9079 | -2.9224 |
| Qwen3 8B | up_proj | dense | 0.2473 | 0.1994 | -0.0479 | 4.4389 | 4.3108 | -0.1281 |

## Largest Gini Increase at alpha=+0.2

| model | layer | proj | agg | base Gini | delta Gini | top rel | top/G |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3 8B | 2 | up_proj | dense | 0.3910 | 0.0593 | 1.6443 | 12.0206 |
| Qwen3 8B | 0 | up_proj | dense | 0.3512 | 0.0579 | 1.4721 | 6.9126 |
| Qwen3 8B | 1 | up_proj | dense | 0.3669 | 0.0562 | 1.5122 | 7.9082 |
| Qwen3 8B | 2 | down_proj | dense | 0.3513 | 0.0542 | 1.7278 | 15.3986 |
| Qwen3 8B | 1 | down_proj | dense | 0.3209 | 0.0512 | 1.7984 | 18.8111 |
| Qwen3 8B | 3 | up_proj | dense | 0.2962 | 0.0507 | 1.4012 | 5.4004 |
| Qwen3 30B-A3B MoE | 0 | up_proj | moe_expert_mean_max | 0.3371 | 0.0502 | 1.3712 | 5.6961 |
| Llama 3.2 3B Instruct | 27 | up_proj | dense | 0.2506 | 0.0492 | 1.6784 | 13.3201 |
| Qwen3 30B-A3B MoE | 22 | down_proj | moe_expert_mean_max | 0.2408 | 0.0451 | 1.3005 | 3.7923 |
| Qwen3 8B | 3 | down_proj | dense | 0.2611 | 0.0451 | 1.4906 | 7.3577 |
| Llama 3.2 3B Instruct | 24 | down_proj | dense | 0.2451 | 0.0448 | 1.2895 | 3.5657 |
| Qwen3 30B-A3B MoE | 24 | down_proj | moe_expert_mean_max | 0.2395 | 0.0447 | 1.2813 | 3.5511 |

## Largest Gini Decrease at alpha=-0.2

| model | layer | proj | agg | base Gini | delta Gini | top rel | top/G |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3 8B | 2 | up_proj | dense | 0.3910 | -0.0654 | 0.6082 | 12.0206 |
| Qwen3 8B | 0 | up_proj | dense | 0.3512 | -0.0629 | 0.6793 | 6.9126 |
| Qwen3 8B | 1 | up_proj | dense | 0.3669 | -0.0620 | 0.6613 | 7.9082 |
| Qwen3 8B | 2 | down_proj | dense | 0.3513 | -0.0590 | 0.5788 | 15.3986 |
| Qwen3 8B | 1 | down_proj | dense | 0.3209 | -0.0552 | 0.5561 | 18.8111 |
| Qwen3 30B-A3B MoE | 0 | up_proj | moe_expert_mean_max | 0.3371 | -0.0539 | 0.7377 | 5.6961 |
| Qwen3 8B | 3 | up_proj | dense | 0.2962 | -0.0538 | 0.7137 | 5.4004 |
| Llama 3.2 3B Instruct | 27 | up_proj | dense | 0.2506 | -0.0495 | 0.5958 | 13.3201 |
| Qwen3 8B | 3 | down_proj | dense | 0.2611 | -0.0476 | 0.6709 | 7.3577 |
| Qwen3 30B-A3B MoE | 22 | down_proj | moe_expert_mean_max | 0.2408 | -0.0464 | 0.7703 | 3.7923 |
| Llama 3.2 3B Instruct | 24 | down_proj | dense | 0.2451 | -0.0464 | 0.7755 | 3.5657 |
| Qwen3 30B-A3B MoE | 24 | down_proj | moe_expert_mean_max | 0.2395 | -0.0461 | 0.7825 | 3.5511 |

## Largest Top Singular Value Relative Increase at alpha=+0.2

| model | layer | proj | agg | base Gini | delta Gini | top rel | top abs delta | top/G |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3 8B | 1 | down_proj | dense | 0.3209 | 0.0512 | 1.7984 | 19.6545 | 18.8111 |
| Qwen3 8B | 2 | down_proj | dense | 0.3513 | 0.0542 | 1.7278 | 13.3616 | 15.3986 |
| Llama 3.2 3B Instruct | 27 | up_proj | dense | 0.2506 | 0.0492 | 1.6784 | 13.2868 | 13.3201 |
| Llama 3.1 8B Instruct | 31 | up_proj | dense | 0.2143 | 0.0427 | 1.6535 | 12.3583 | 12.3603 |
| Qwen3 8B | 2 | up_proj | dense | 0.3910 | 0.0593 | 1.6443 | 8.2783 | 12.0206 |
| Llama 3.2 1B Instruct | 15 | up_proj | dense | 0.2224 | 0.0443 | 1.5488 | 7.4058 | 8.9111 |
| Qwen3 8B | 35 | up_proj | dense | 0.1956 | 0.0368 | 1.5409 | 13.0995 | 8.6867 |
| Llama 3.2 3B Instruct | 26 | up_proj | dense | 0.2263 | 0.0417 | 1.5278 | 6.2346 | 8.3250 |
| Qwen3 8B | 1 | up_proj | dense | 0.3669 | 0.0562 | 1.5122 | 4.2561 | 7.9082 |
| Qwen3 8B | 34 | down_proj | dense | 0.1931 | 0.0359 | 1.5104 | 10.9520 | 7.8602 |
| Llama 3.1 8B Instruct | 30 | up_proj | dense | 0.1958 | 0.0371 | 1.5033 | 5.3734 | 7.6784 |
| Qwen3 8B | 3 | down_proj | dense | 0.2611 | 0.0451 | 1.4906 | 6.1539 | 7.3577 |

## MoE Expert-Max Gini Increase at alpha=+0.2

| model | layer | proj | n experts | mean base Gini | mean delta Gini | max delta Gini | mean top rel | max top rel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3 30B-A3B MoE | 1 | down_proj | 128 | 0.2373 | 0.0437 | 0.0822 | 1.2916 | 1.5986 |
| Qwen3 30B-A3B MoE | 0 | up_proj | 128 | 0.3371 | 0.0502 | 0.0679 | 1.3712 | 2.0313 |
| Qwen3 30B-A3B MoE | 2 | up_proj | 128 | 0.2308 | 0.0424 | 0.0665 | 1.2222 | 1.5355 |
| Qwen3 30B-A3B MoE | 2 | down_proj | 128 | 0.2370 | 0.0444 | 0.0658 | 1.3118 | 1.5337 |
| Qwen3 30B-A3B MoE | 22 | down_proj | 128 | 0.2408 | 0.0451 | 0.0647 | 1.3005 | 1.5791 |
| Qwen3 30B-A3B MoE | 24 | down_proj | 128 | 0.2395 | 0.0447 | 0.0641 | 1.2813 | 1.5255 |
| Qwen3 30B-A3B MoE | 25 | down_proj | 128 | 0.2358 | 0.0438 | 0.0634 | 1.2714 | 1.5398 |
| Qwen3 30B-A3B MoE | 36 | down_proj | 128 | 0.2332 | 0.0436 | 0.0627 | 1.2744 | 1.5313 |
| Qwen3 30B-A3B MoE | 1 | up_proj | 128 | 0.2464 | 0.0440 | 0.0627 | 1.2350 | 1.8115 |
| Qwen3 30B-A3B MoE | 34 | down_proj | 128 | 0.2353 | 0.0440 | 0.0615 | 1.2961 | 1.5339 |
| Qwen3 30B-A3B MoE | 39 | down_proj | 128 | 0.2324 | 0.0433 | 0.0604 | 1.2739 | 1.5164 |
| Qwen3 30B-A3B MoE | 38 | down_proj | 128 | 0.2274 | 0.0422 | 0.0603 | 1.2681 | 1.5196 |

## MoE Expert-Max Top Singular Value Relative Increase at alpha=+0.2

| model | layer | proj | n experts | mean delta Gini | max delta Gini | mean top rel | max top rel | max top/G |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3 30B-A3B MoE | 0 | down_proj | 128 | 0.0434 | 0.0539 | 1.4319 | 2.1770 | 48.8976 |
| Qwen3 30B-A3B MoE | 0 | up_proj | 128 | 0.0502 | 0.0679 | 1.3712 | 2.0313 | 34.5798 |
| Qwen3 30B-A3B MoE | 1 | up_proj | 128 | 0.0440 | 0.0627 | 1.2350 | 1.8115 | 19.5060 |
| Qwen3 30B-A3B MoE | 1 | down_proj | 128 | 0.0437 | 0.0822 | 1.2916 | 1.5986 | 10.4409 |
| Qwen3 30B-A3B MoE | 22 | down_proj | 128 | 0.0451 | 0.0647 | 1.3005 | 1.5791 | 9.8181 |
| Qwen3 30B-A3B MoE | 47 | up_proj | 128 | 0.0406 | 0.0564 | 1.3418 | 1.5581 | 9.1821 |
| Qwen3 30B-A3B MoE | 7 | down_proj | 128 | 0.0422 | 0.0480 | 1.2879 | 1.5422 | 8.7236 |
| Qwen3 30B-A3B MoE | 25 | down_proj | 128 | 0.0438 | 0.0634 | 1.2714 | 1.5398 | 8.6571 |
| Qwen3 30B-A3B MoE | 2 | up_proj | 128 | 0.0424 | 0.0665 | 1.2222 | 1.5355 | 8.5367 |
| Qwen3 30B-A3B MoE | 34 | down_proj | 128 | 0.0440 | 0.0615 | 1.2961 | 1.5339 | 8.4921 |
| Qwen3 30B-A3B MoE | 2 | down_proj | 128 | 0.0444 | 0.0658 | 1.3118 | 1.5337 | 8.4846 |
| Qwen3 30B-A3B MoE | 36 | down_proj | 128 | 0.0436 | 0.0627 | 1.2744 | 1.5313 | 8.4195 |

## Interpretation

The tables show that alpha has a stable sign effect but a non-uniform layer effect. Positive alpha consistently raises Gini and top singular values; negative alpha consistently lowers them. However, the amount of movement depends on each layer's original spectrum, especially `top_sv / geometric_mean_sv`.

The late-vs-early table should therefore be read per model and projection. In the current dense results, Llama up-projections often show higher late-layer Gini than early-layer Gini, while Qwen3-8B has stronger early-layer spectral concentration. This means a global alpha is not a uniform perturbation across depth: it is filtered through each layer's existing spectral imbalance.

For MoE models, `moe_expert_mean_max` rows aggregate routed experts by layer/projection. Mean captures typical expert behavior; max captures the most spectrally concentrated expert in that layer. Shared experts are retained separately when available.
