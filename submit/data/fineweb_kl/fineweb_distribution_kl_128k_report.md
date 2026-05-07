# FineWeb 128k Output Distribution KL

KL is computed token-wise on a fixed 128k-token FineWeb10B validation sample.
`KL(P_alpha || P_0)` compares the Matthew-perturbed model output distribution against the alpha=0 output distribution at the same token positions.
`baseline_random_pair_kl_mean` is computed under alpha=0 by randomly pairing token positions in the same sample and averaging `KL(P_i || P_j)`.

| model | baseline random-token KL mean | baseline random-token KL std | pairs | alpha=-0.2 KL | alpha=+0.2 KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama32_1b_instruct | 9.62103 | 4.00551 | 2048 | 0.198951 | 0.447433 |
| llama32_3b_instruct | 10.0128 | 4.04614 | 2048 | 0.410298 | 0.699252 |
| llama31_8b_instruct | 11.2442 | 4.60754 | 2048 | 0.433743 | 0.499996 |
| qwen3_8b | 12.8957 | 5.89436 | 2048 | 0.323254 | 0.333024 |

Note: `qwen3_30b_a3b_moe` is still running on xjhl4. It will be appended to this table after its 128k baseline logits and alpha KL runs finish.

Raw long table:

- `docs/results/fineweb_distribution_kl/fineweb_distribution_kl_128k_long.csv`
- `docs/results/fineweb_distribution_kl/fineweb_distribution_kl_128k_summary.csv`
