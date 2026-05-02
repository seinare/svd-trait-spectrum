# Experiment Result Tables

Generated from remote run artifacts on xjhl4 and xjha6. CSV files in this directory contain the full long-form tables.

## Coverage Check

| model | alpha | missing_module |
| --- | --- | --- |
| qwen3_8b | -0.4 | standard_cot |
| qwen3_8b | -0.4 | lm_eval |
| qwen3_8b | -0.3 | lm_eval |
| qwen3_8b | -0.2 | lm_eval |
| qwen3_8b | -0.1 | lm_eval |
| qwen3_8b | 0.0 | standard_cot |
| qwen3_8b | 0.0 | lm_eval |
| qwen3_8b | 0.1 | lm_eval |
| qwen3_8b | 0.2 | lm_eval |
| qwen3_8b | 0.3 | standard_cot |
| qwen3_8b | 0.3 | lm_eval |
| qwen3_8b | 0.4 | lm_eval |

Qwen residual TP=2 lm-eval / CoT was still running when parsed; missing Qwen rows should be refreshed after `results_*.json` and `standard_cot/*.json` are present.


## Main Module Summary

| model | alpha | GSM8K | MATH | ARC | DROP | BFCL_retry | TruthfulQA | HaluEval | AdvBench | GSM8K_cot | MATH_cot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama1b | -0.3 | 0.150 | 0.160 | 0.230 | 0.120 | 0.010 | 0.490 | 0.500 | 2.700 | 0.150 | 0.160 |
| llama1b | -0.2 | 0.200 | 0.210 | 0.350 | 0.250 | 0.085 | 0.580 | 0.890 | 2.740 | 0.310 | 0.190 |
| llama1b | -0.1 | 0.170 | 0.280 | 0.430 | 0.240 | 0.090 | 0.560 | 1.210 | 2.830 | 0.330 | 0.280 |
| llama1b | 0.0 | 0.190 | 0.190 | 0.480 | 0.230 | 0.010 | 0.700 | 1.360 | 2.830 | 0.370 | 0.310 |
| llama1b | 0.1 | 0.080 | 0.120 | 0.490 | 0.230 | 0.045 | 0.710 | 1.350 | 2.750 | 0.280 | 0.370 |
| llama1b | 0.2 | 0.010 | 0.120 | 0.430 | 0.130 | 0.005 | 0.620 | 1.110 | 2.790 | 0.200 | 0.140 |
| llama1b | 0.3 | 0.010 | 0.100 | 0.080 | 0.110 | 0.000 | 0.620 | 0.780 | 2.760 | 0.030 | 0.020 |
| llama3b | -0.3 | 0.010 | 0.040 | 0.000 | 0.020 | 0.000 | 0.200 | 0.040 | 2.870 | 0.000 | 0.000 |
| llama3b | -0.2 | 0.200 | 0.190 | 0.360 | 0.210 | 0.345 | 0.760 | 1.230 | 2.850 | 0.490 | 0.190 |
| llama3b | -0.1 | 0.120 | 0.200 | 0.670 | 0.490 | 0.380 | 0.940 | 1.620 | 2.910 | 0.700 | 0.420 |
| llama3b | 0.0 | 0.200 | 0.210 | 0.690 | 0.460 | 0.095 | 1.000 | 1.660 | 2.970 | 0.720 | 0.460 |
| llama3b | 0.1 | 0.150 | 0.240 | 0.700 | 0.460 | 0.040 | 1.150 | 1.690 | 2.950 | 0.670 | 0.450 |
| llama3b | 0.2 | 0.150 | 0.230 | 0.590 | 0.490 | 0.040 | 1.010 | 1.680 | 2.960 | 0.570 | 0.340 |
| llama3b | 0.3 | 0.110 | 0.100 | 0.190 | 0.230 | 0.000 | 0.620 | 1.340 | 2.950 | 0.130 | 0.090 |
| qwen3_8b | -0.4 | 0.130 | 0.150 | 0.700 | 0.210 | 0.015 | 0.600 | 0.850 | 2.810 |  |  |
| qwen3_8b | -0.3 | 0.450 | 0.470 | 0.860 | 0.430 | 0.355 | 0.820 | 1.240 | 2.810 | 0.030 | 0.020 |
| qwen3_8b | -0.2 | 0.810 | 0.490 | 0.900 | 0.610 | 0.415 | 0.870 | 1.360 | 2.920 | 0.660 | 0.500 |
| qwen3_8b | -0.1 | 0.840 | 0.450 | 0.900 | 0.630 | 0.395 | 0.860 | 1.390 | 2.890 | 0.920 | 0.720 |
| qwen3_8b | 0.0 | 0.870 | 0.420 | 0.890 | 0.680 | 0.510 | 0.930 | 1.350 | 2.930 |  |  |
| qwen3_8b | 0.1 | 0.800 | 0.380 | 0.910 | 0.730 | 0.580 | 1.000 | 1.410 | 2.860 | 0.970 | 0.800 |
| qwen3_8b | 0.2 | 0.710 | 0.350 | 0.850 | 0.600 | 0.270 | 0.980 | 1.410 | 2.830 | 0.800 | 0.610 |
| qwen3_8b | 0.3 | 0.250 | 0.210 | 0.700 | 0.270 | 0.265 | 0.790 | 1.090 | 2.890 |  |  |
| qwen3_8b | 0.4 | 0.080 | 0.150 | 0.270 | 0.060 | 0.015 | 0.770 | 0.510 | 2.380 | 0.080 | 0.020 |

## lm-eval Subtasks: GPQA / AGIEval / HellaSwag / IFEval

| model | alpha | group | task | metric | value |
| --- | --- | --- | --- | --- | --- |
| llama1b | 0.2 | agieval | agieval_aqua_rat | acc_norm,none | 0.205 |
| llama1b | 0.2 | agieval | agieval_logiqa_en | acc_norm,none | 0.327 |
| llama1b | 0.2 | agieval | agieval_lsat_ar | acc_norm,none | 0.178 |
| llama1b | 0.2 | agieval | agieval_lsat_lr | acc_norm,none | 0.214 |
| llama1b | 0.2 | agieval | agieval_lsat_rc | acc_norm,none | 0.204 |
| llama1b | 0.2 | agieval | agieval_sat_en | acc_norm,none | 0.238 |
| llama1b | 0.2 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.228 |
| llama1b | 0.2 | agieval | agieval_sat_math | acc_norm,none | 0.241 |
| llama1b | 0.2 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.258 |
| llama1b | 0.2 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.266 |
| llama1b | 0.2 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.250 |
| llama1b | 0.2 | hellaswag | hellaswag | acc_norm,none | 0.602 |
| llama1b | 0.2 | ifeval | ifeval | prompt_level_strict_acc,none | 0.368 |
| llama1b | -0.2 | agieval | agieval_aqua_rat | acc_norm,none | 0.264 |
| llama1b | -0.2 | agieval | agieval_logiqa_en | acc_norm,none | 0.269 |
| llama1b | -0.2 | agieval | agieval_lsat_ar | acc_norm,none | 0.187 |
| llama1b | -0.2 | agieval | agieval_lsat_lr | acc_norm,none | 0.233 |
| llama1b | -0.2 | agieval | agieval_lsat_rc | acc_norm,none | 0.212 |
| llama1b | -0.2 | agieval | agieval_sat_en | acc_norm,none | 0.272 |
| llama1b | -0.2 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.233 |
| llama1b | -0.2 | agieval | agieval_sat_math | acc_norm,none | 0.255 |
| llama1b | -0.2 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.247 |
| llama1b | -0.2 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.227 |
| llama1b | -0.2 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.295 |
| llama1b | -0.2 | hellaswag | hellaswag | acc_norm,none | 0.587 |
| llama1b | -0.2 | ifeval | ifeval | prompt_level_strict_acc,none | 0.336 |
| llama1b | -0.3 | agieval | agieval_aqua_rat | acc_norm,none | 0.240 |
| llama1b | -0.3 | agieval | agieval_logiqa_en | acc_norm,none | 0.272 |
| llama1b | -0.3 | agieval | agieval_lsat_ar | acc_norm,none | 0.200 |
| llama1b | -0.3 | agieval | agieval_lsat_lr | acc_norm,none | 0.235 |
| llama1b | -0.3 | agieval | agieval_lsat_rc | acc_norm,none | 0.223 |
| llama1b | -0.3 | agieval | agieval_sat_en | acc_norm,none | 0.223 |
| llama1b | -0.3 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.248 |
| llama1b | -0.3 | agieval | agieval_sat_math | acc_norm,none | 0.264 |
| llama1b | -0.3 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.222 |
| llama1b | -0.3 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.218 |
| llama1b | -0.3 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.310 |
| llama1b | -0.3 | hellaswag | hellaswag | acc_norm,none | 0.560 |
| llama1b | -0.3 | ifeval | ifeval | prompt_level_strict_acc,none | 0.257 |
| llama1b | 0.3 | agieval | agieval_aqua_rat | acc_norm,none | 0.193 |
| llama1b | 0.3 | agieval | agieval_logiqa_en | acc_norm,none | 0.304 |
| llama1b | 0.3 | agieval | agieval_lsat_ar | acc_norm,none | 0.226 |
| llama1b | 0.3 | agieval | agieval_lsat_lr | acc_norm,none | 0.208 |
| llama1b | 0.3 | agieval | agieval_lsat_rc | acc_norm,none | 0.197 |
| llama1b | 0.3 | agieval | agieval_sat_en | acc_norm,none | 0.228 |
| llama1b | 0.3 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.199 |
| llama1b | 0.3 | agieval | agieval_sat_math | acc_norm,none | 0.245 |
| llama1b | 0.3 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.283 |
| llama1b | 0.3 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.267 |
| llama1b | 0.3 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.259 |
| llama1b | 0.3 | hellaswag | hellaswag | acc_norm,none | 0.555 |
| llama1b | 0.3 | ifeval | ifeval | prompt_level_strict_acc,none | 0.274 |
| llama1b | 0.1 | agieval | agieval_aqua_rat | acc_norm,none | 0.217 |
| llama1b | 0.1 | agieval | agieval_logiqa_en | acc_norm,none | 0.298 |
| llama1b | 0.1 | agieval | agieval_lsat_ar | acc_norm,none | 0.139 |
| llama1b | 0.1 | agieval | agieval_lsat_lr | acc_norm,none | 0.233 |
| llama1b | 0.1 | agieval | agieval_lsat_rc | acc_norm,none | 0.197 |
| llama1b | 0.1 | agieval | agieval_sat_en | acc_norm,none | 0.296 |
| llama1b | 0.1 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.252 |
| llama1b | 0.1 | agieval | agieval_sat_math | acc_norm,none | 0.291 |
| llama1b | 0.1 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.263 |
| llama1b | 0.1 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.267 |
| llama1b | 0.1 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.263 |
| llama1b | 0.1 | hellaswag | hellaswag | acc_norm,none | 0.616 |
| llama1b | 0.1 | ifeval | ifeval | prompt_level_strict_acc,none | 0.412 |
| llama1b | -0.1 | agieval | agieval_aqua_rat | acc_norm,none | 0.256 |
| llama1b | -0.1 | agieval | agieval_logiqa_en | acc_norm,none | 0.300 |
| llama1b | -0.1 | agieval | agieval_lsat_ar | acc_norm,none | 0.204 |
| llama1b | -0.1 | agieval | agieval_lsat_lr | acc_norm,none | 0.251 |
| llama1b | -0.1 | agieval | agieval_lsat_rc | acc_norm,none | 0.204 |
| llama1b | -0.1 | agieval | agieval_sat_en | acc_norm,none | 0.257 |
| llama1b | -0.1 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.262 |
| llama1b | -0.1 | agieval | agieval_sat_math | acc_norm,none | 0.255 |
| llama1b | -0.1 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.258 |
| llama1b | -0.1 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.209 |
| llama1b | -0.1 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.268 |
| llama1b | -0.1 | hellaswag | hellaswag | acc_norm,none | 0.604 |
| llama1b | -0.1 | ifeval | ifeval | prompt_level_strict_acc,none | 0.396 |
| llama3b | 0.2 | agieval | agieval_aqua_rat | acc_norm,none | 0.169 |
| llama3b | 0.2 | agieval | agieval_logiqa_en | acc_norm,none | 0.335 |
| llama3b | 0.2 | agieval | agieval_lsat_ar | acc_norm,none | 0.209 |
| llama3b | 0.2 | agieval | agieval_lsat_lr | acc_norm,none | 0.276 |
| llama3b | 0.2 | agieval | agieval_lsat_rc | acc_norm,none | 0.320 |
| llama3b | 0.2 | agieval | agieval_sat_en | acc_norm,none | 0.413 |
| llama3b | 0.2 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.252 |
| llama3b | 0.2 | agieval | agieval_sat_math | acc_norm,none | 0.318 |
| llama3b | 0.2 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.313 |
| llama3b | 0.2 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.319 |
| llama3b | 0.2 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.286 |
| llama3b | 0.2 | hellaswag | hellaswag | acc_norm,none | 0.667 |
| llama3b | 0.2 | ifeval | ifeval | prompt_level_strict_acc,none | 0.455 |
| llama3b | -0.3 | agieval | agieval_aqua_rat | acc_norm,none | 0.205 |
| llama3b | -0.3 | agieval | agieval_logiqa_en | acc_norm,none | 0.267 |
| llama3b | -0.3 | agieval | agieval_lsat_ar | acc_norm,none | 0.196 |
| llama3b | -0.3 | agieval | agieval_lsat_lr | acc_norm,none | 0.241 |
| llama3b | -0.3 | agieval | agieval_lsat_rc | acc_norm,none | 0.182 |
| llama3b | -0.3 | agieval | agieval_sat_en | acc_norm,none | 0.218 |
| llama3b | -0.3 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.257 |
| llama3b | -0.3 | agieval | agieval_sat_math | acc_norm,none | 0.205 |
| llama3b | -0.3 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.237 |
| llama3b | -0.3 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.234 |
| llama3b | -0.3 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.268 |
| llama3b | -0.3 | hellaswag | hellaswag | acc_norm,none | 0.576 |
| llama3b | -0.3 | ifeval | ifeval | prompt_level_strict_acc,none | 0.209 |
| llama3b | -0.1 | agieval | agieval_aqua_rat | acc_norm,none | 0.220 |
| llama3b | -0.1 | agieval | agieval_logiqa_en | acc_norm,none | 0.332 |
| llama3b | -0.1 | agieval | agieval_lsat_ar | acc_norm,none | 0.200 |
| llama3b | -0.1 | agieval | agieval_lsat_lr | acc_norm,none | 0.275 |
| llama3b | -0.1 | agieval | agieval_lsat_rc | acc_norm,none | 0.331 |
| llama3b | -0.1 | agieval | agieval_sat_en | acc_norm,none | 0.422 |
| llama3b | -0.1 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.306 |
| llama3b | -0.1 | agieval | agieval_sat_math | acc_norm,none | 0.323 |
| llama3b | -0.1 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.323 |
| llama3b | -0.1 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.266 |
| llama3b | -0.1 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.321 |
| llama3b | -0.1 | hellaswag | hellaswag | acc_norm,none | 0.709 |
| llama3b | -0.1 | ifeval | ifeval | prompt_level_strict_acc,none | 0.457 |
| llama3b | -0.2 | agieval | agieval_aqua_rat | acc_norm,none | 0.224 |
| llama3b | -0.2 | agieval | agieval_logiqa_en | acc_norm,none | 0.280 |
| llama3b | -0.2 | agieval | agieval_lsat_ar | acc_norm,none | 0.165 |
| llama3b | -0.2 | agieval | agieval_lsat_lr | acc_norm,none | 0.233 |
| llama3b | -0.2 | agieval | agieval_lsat_rc | acc_norm,none | 0.230 |
| llama3b | -0.2 | agieval | agieval_sat_en | acc_norm,none | 0.306 |
| llama3b | -0.2 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.320 |
| llama3b | -0.2 | agieval | agieval_sat_math | acc_norm,none | 0.241 |
| llama3b | -0.2 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.273 |
| llama3b | -0.2 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.238 |
| llama3b | -0.2 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.299 |
| llama3b | -0.2 | hellaswag | hellaswag | acc_norm,none | 0.675 |
| llama3b | -0.2 | ifeval | ifeval | prompt_level_strict_acc,none | 0.383 |
| llama3b | 0.3 | agieval | agieval_aqua_rat | acc_norm,none | 0.146 |
| llama3b | 0.3 | agieval | agieval_logiqa_en | acc_norm,none | 0.310 |
| llama3b | 0.3 | agieval | agieval_lsat_ar | acc_norm,none | 0.226 |
| llama3b | 0.3 | agieval | agieval_lsat_lr | acc_norm,none | 0.218 |
| llama3b | 0.3 | agieval | agieval_lsat_rc | acc_norm,none | 0.178 |
| llama3b | 0.3 | agieval | agieval_sat_en | acc_norm,none | 0.204 |
| llama3b | 0.3 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.180 |
| llama3b | 0.3 | agieval | agieval_sat_math | acc_norm,none | 0.273 |
| llama3b | 0.3 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.298 |
| llama3b | 0.3 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.273 |
| llama3b | 0.3 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.246 |
| llama3b | 0.3 | hellaswag | hellaswag | acc_norm,none | 0.549 |
| llama3b | 0.3 | ifeval | ifeval | prompt_level_strict_acc,none | 0.307 |
| llama3b | 0.1 | agieval | agieval_aqua_rat | acc_norm,none | 0.181 |
| llama3b | 0.1 | agieval | agieval_logiqa_en | acc_norm,none | 0.359 |
| llama3b | 0.1 | agieval | agieval_lsat_ar | acc_norm,none | 0.191 |
| llama3b | 0.1 | agieval | agieval_lsat_lr | acc_norm,none | 0.280 |
| llama3b | 0.1 | agieval | agieval_lsat_rc | acc_norm,none | 0.323 |
| llama3b | 0.1 | agieval | agieval_sat_en | acc_norm,none | 0.432 |
| llama3b | 0.1 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.262 |
| llama3b | 0.1 | agieval | agieval_sat_math | acc_norm,none | 0.345 |
| llama3b | 0.1 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.318 |
| llama3b | 0.1 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.315 |
| llama3b | 0.1 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.317 |
| llama3b | 0.1 | hellaswag | hellaswag | acc_norm,none | 0.707 |
| llama3b | 0.1 | ifeval | ifeval | prompt_level_strict_acc,none | 0.505 |
| llama3b | 0.0 | agieval | agieval_aqua_rat | acc_norm,none | 0.209 |
| llama3b | 0.0 | agieval | agieval_logiqa_en | acc_norm,none | 0.339 |
| llama3b | 0.0 | agieval | agieval_lsat_ar | acc_norm,none | 0.191 |
| llama3b | 0.0 | agieval | agieval_lsat_lr | acc_norm,none | 0.282 |
| llama3b | 0.0 | agieval | agieval_lsat_rc | acc_norm,none | 0.361 |
| llama3b | 0.0 | agieval | agieval_sat_en | acc_norm,none | 0.490 |
| llama3b | 0.0 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.291 |
| llama3b | 0.0 | agieval | agieval_sat_math | acc_norm,none | 0.323 |
| llama3b | 0.0 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.308 |
| llama3b | 0.0 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.300 |
| llama3b | 0.0 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.310 |
| llama3b | 0.0 | hellaswag | hellaswag | acc_norm,none | 0.716 |
| llama3b | 0.0 | ifeval | ifeval | prompt_level_strict_acc,none | 0.519 |
| llama1b | 0.0 | agieval | agieval_aqua_rat | acc_norm,none | 0.240 |
| llama1b | 0.0 | agieval | agieval_logiqa_en | acc_norm,none | 0.293 |
| llama1b | 0.0 | agieval | agieval_lsat_ar | acc_norm,none | 0.157 |
| llama1b | 0.0 | agieval | agieval_lsat_lr | acc_norm,none | 0.261 |
| llama1b | 0.0 | agieval | agieval_lsat_rc | acc_norm,none | 0.197 |
| llama1b | 0.0 | agieval | agieval_sat_en | acc_norm,none | 0.286 |
| llama1b | 0.0 | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.262 |
| llama1b | 0.0 | agieval | agieval_sat_math | acc_norm,none | 0.277 |
| llama1b | 0.0 | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.298 |
| llama1b | 0.0 | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.231 |
| llama1b | 0.0 | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.259 |
| llama1b | 0.0 | hellaswag | hellaswag | acc_norm,none | 0.617 |
| llama1b | 0.0 | ifeval | ifeval | prompt_level_strict_acc,none | 0.442 |

## TRAIT Detailed Counts

| model | alpha | trait | total | high | low | invalid | high_rate | low_rate | invalid_rate | trait_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama1b | -0.3 | Openness | 1000 | 340 | 257 | 403 | 0.340 | 0.257 | 0.403 | 0.139 |
| llama1b | -0.3 | Conscientiousness | 1000 | 422 | 228 | 350 | 0.422 | 0.228 | 0.350 | 0.298 |
| llama1b | -0.3 | Extraversion | 1000 | 224 | 316 | 460 | 0.224 | 0.316 | 0.460 | -0.170 |
| llama1b | -0.3 | Agreeableness | 1000 | 241 | 113 | 646 | 0.241 | 0.113 | 0.646 | 0.362 |
| llama1b | -0.3 | Neuroticism | 1000 | 131 | 274 | 595 | 0.131 | 0.274 | 0.595 | -0.353 |
| llama1b | -0.3 | Machiavellianism | 1000 | 115 | 195 | 690 | 0.115 | 0.195 | 0.690 | -0.258 |
| llama1b | -0.3 | Narcissism | 1000 | 120 | 245 | 635 | 0.120 | 0.245 | 0.635 | -0.342 |
| llama1b | -0.3 | Psychopathy | 1000 | 111 | 312 | 577 | 0.111 | 0.312 | 0.577 | -0.475 |
| llama1b | 0.1 | Openness | 1000 | 509 | 491 | 0 | 0.509 | 0.491 | 0.000 | 0.018 |
| llama1b | 0.1 | Conscientiousness | 1000 | 516 | 484 | 0 | 0.516 | 0.484 | 0.000 | 0.032 |
| llama1b | 0.1 | Extraversion | 1000 | 482 | 518 | 0 | 0.482 | 0.518 | 0.000 | -0.036 |
| llama1b | 0.1 | Agreeableness | 1000 | 513 | 487 | 0 | 0.513 | 0.487 | 0.000 | 0.026 |
| llama1b | 0.1 | Neuroticism | 1000 | 431 | 569 | 0 | 0.431 | 0.569 | 0.000 | -0.138 |
| llama1b | 0.1 | Machiavellianism | 1000 | 480 | 520 | 0 | 0.480 | 0.520 | 0.000 | -0.040 |
| llama1b | 0.1 | Narcissism | 1000 | 429 | 571 | 0 | 0.429 | 0.571 | 0.000 | -0.142 |
| llama1b | 0.1 | Psychopathy | 1000 | 461 | 539 | 0 | 0.461 | 0.539 | 0.000 | -0.078 |
| qwen3_8b | 0.2 | Openness | 1000 | 589 | 411 | 0 | 0.589 | 0.411 | 0.000 | 0.178 |
| qwen3_8b | 0.2 | Conscientiousness | 1000 | 651 | 349 | 0 | 0.651 | 0.349 | 0.000 | 0.302 |
| qwen3_8b | 0.2 | Extraversion | 1000 | 458 | 542 | 0 | 0.458 | 0.542 | 0.000 | -0.084 |
| qwen3_8b | 0.2 | Agreeableness | 1000 | 764 | 236 | 0 | 0.764 | 0.236 | 0.000 | 0.528 |
| qwen3_8b | 0.2 | Neuroticism | 1000 | 228 | 772 | 0 | 0.228 | 0.772 | 0.000 | -0.544 |
| qwen3_8b | 0.2 | Machiavellianism | 1000 | 244 | 756 | 0 | 0.244 | 0.756 | 0.000 | -0.512 |
| qwen3_8b | 0.2 | Narcissism | 1000 | 146 | 854 | 0 | 0.146 | 0.854 | 0.000 | -0.708 |
| qwen3_8b | 0.2 | Psychopathy | 1000 | 147 | 853 | 0 | 0.147 | 0.853 | 0.000 | -0.706 |
| qwen3_8b | -0.2 | Openness | 1000 | 605 | 395 | 0 | 0.605 | 0.395 | 0.000 | 0.210 |
| qwen3_8b | -0.2 | Conscientiousness | 1000 | 754 | 245 | 1 | 0.754 | 0.245 | 0.001 | 0.510 |
| qwen3_8b | -0.2 | Extraversion | 1000 | 377 | 622 | 1 | 0.377 | 0.622 | 0.001 | -0.245 |
| qwen3_8b | -0.2 | Agreeableness | 1000 | 787 | 213 | 0 | 0.787 | 0.213 | 0.000 | 0.574 |
| qwen3_8b | -0.2 | Neuroticism | 1000 | 236 | 764 | 0 | 0.236 | 0.764 | 0.000 | -0.528 |
| qwen3_8b | -0.2 | Machiavellianism | 1000 | 268 | 732 | 0 | 0.268 | 0.732 | 0.000 | -0.464 |
| qwen3_8b | -0.2 | Narcissism | 1000 | 162 | 838 | 0 | 0.162 | 0.838 | 0.000 | -0.676 |
| qwen3_8b | -0.2 | Psychopathy | 1000 | 63 | 937 | 0 | 0.063 | 0.937 | 0.000 | -0.874 |
| llama3b | 0.2 | Openness | 1000 | 623 | 375 | 2 | 0.623 | 0.375 | 0.002 | 0.248 |
| llama3b | 0.2 | Conscientiousness | 1000 | 737 | 262 | 1 | 0.737 | 0.262 | 0.001 | 0.475 |
| llama3b | 0.2 | Extraversion | 1000 | 398 | 602 | 0 | 0.398 | 0.602 | 0.000 | -0.204 |
| llama3b | 0.2 | Agreeableness | 1000 | 730 | 270 | 0 | 0.730 | 0.270 | 0.000 | 0.460 |
| llama3b | 0.2 | Neuroticism | 1000 | 343 | 656 | 1 | 0.343 | 0.656 | 0.001 | -0.313 |
| llama3b | 0.2 | Machiavellianism | 1000 | 198 | 796 | 6 | 0.198 | 0.796 | 0.006 | -0.602 |
| llama3b | 0.2 | Narcissism | 1000 | 126 | 873 | 1 | 0.126 | 0.873 | 0.001 | -0.748 |
| llama3b | 0.2 | Psychopathy | 1000 | 118 | 696 | 186 | 0.118 | 0.696 | 0.186 | -0.710 |
| llama3b | -0.2 | Openness | 1000 | 539 | 461 | 0 | 0.539 | 0.461 | 0.000 | 0.078 |
| llama3b | -0.2 | Conscientiousness | 1000 | 794 | 206 | 0 | 0.794 | 0.206 | 0.000 | 0.588 |
| llama3b | -0.2 | Extraversion | 1000 | 300 | 700 | 0 | 0.300 | 0.700 | 0.000 | -0.400 |
| llama3b | -0.2 | Agreeableness | 1000 | 686 | 314 | 0 | 0.686 | 0.314 | 0.000 | 0.372 |
| llama3b | -0.2 | Neuroticism | 1000 | 282 | 718 | 0 | 0.282 | 0.718 | 0.000 | -0.436 |
| llama3b | -0.2 | Machiavellianism | 1000 | 206 | 794 | 0 | 0.206 | 0.794 | 0.000 | -0.588 |
| llama3b | -0.2 | Narcissism | 1000 | 155 | 845 | 0 | 0.155 | 0.845 | 0.000 | -0.690 |
| llama3b | -0.2 | Psychopathy | 1000 | 103 | 897 | 0 | 0.103 | 0.897 | 0.000 | -0.794 |
| llama1b | -0.1 | Openness | 1000 | 532 | 468 | 0 | 0.532 | 0.468 | 0.000 | 0.064 |
| llama1b | -0.1 | Conscientiousness | 1000 | 569 | 431 | 0 | 0.569 | 0.431 | 0.000 | 0.138 |
| llama1b | -0.1 | Extraversion | 1000 | 445 | 555 | 0 | 0.445 | 0.555 | 0.000 | -0.110 |
| llama1b | -0.1 | Agreeableness | 1000 | 558 | 442 | 0 | 0.558 | 0.442 | 0.000 | 0.116 |
| llama1b | -0.1 | Neuroticism | 1000 | 373 | 627 | 0 | 0.373 | 0.627 | 0.000 | -0.254 |
| llama1b | -0.1 | Machiavellianism | 1000 | 416 | 584 | 0 | 0.416 | 0.584 | 0.000 | -0.168 |
| llama1b | -0.1 | Narcissism | 1000 | 348 | 652 | 0 | 0.348 | 0.652 | 0.000 | -0.304 |
| llama1b | -0.1 | Psychopathy | 1000 | 379 | 621 | 0 | 0.379 | 0.621 | 0.000 | -0.242 |
| llama1b | 0.3 | Openness | 1000 | 552 | 448 | 0 | 0.552 | 0.448 | 0.000 | 0.104 |
| llama1b | 0.3 | Conscientiousness | 1000 | 586 | 414 | 0 | 0.586 | 0.414 | 0.000 | 0.172 |
| llama1b | 0.3 | Extraversion | 1000 | 462 | 538 | 0 | 0.462 | 0.538 | 0.000 | -0.076 |
| llama1b | 0.3 | Agreeableness | 1000 | 582 | 418 | 0 | 0.582 | 0.418 | 0.000 | 0.164 |
| llama1b | 0.3 | Neuroticism | 1000 | 413 | 587 | 0 | 0.413 | 0.587 | 0.000 | -0.174 |
| llama1b | 0.3 | Machiavellianism | 1000 | 375 | 625 | 0 | 0.375 | 0.625 | 0.000 | -0.250 |
| llama1b | 0.3 | Narcissism | 1000 | 359 | 641 | 0 | 0.359 | 0.641 | 0.000 | -0.282 |
| llama1b | 0.3 | Psychopathy | 1000 | 288 | 712 | 0 | 0.288 | 0.712 | 0.000 | -0.424 |
| qwen3_8b | 0.4 | Openness | 1000 | 467 | 350 | 183 | 0.467 | 0.350 | 0.183 | 0.143 |
| qwen3_8b | 0.4 | Conscientiousness | 1000 | 417 | 454 | 129 | 0.417 | 0.454 | 0.129 | -0.042 |
| qwen3_8b | 0.4 | Extraversion | 1000 | 436 | 467 | 97 | 0.436 | 0.467 | 0.097 | -0.034 |
| qwen3_8b | 0.4 | Agreeableness | 1000 | 588 | 339 | 73 | 0.588 | 0.339 | 0.073 | 0.269 |
| qwen3_8b | 0.4 | Neuroticism | 1000 | 203 | 669 | 128 | 0.203 | 0.669 | 0.128 | -0.534 |
| qwen3_8b | 0.4 | Machiavellianism | 1000 | 217 | 528 | 255 | 0.217 | 0.528 | 0.255 | -0.417 |
| qwen3_8b | 0.4 | Narcissism | 1000 | 196 | 636 | 168 | 0.196 | 0.636 | 0.168 | -0.529 |
| qwen3_8b | 0.4 | Psychopathy | 1000 | 210 | 608 | 182 | 0.210 | 0.608 | 0.182 | -0.487 |
| qwen3_8b | -0.4 | Openness | 1000 | 435 | 335 | 230 | 0.435 | 0.335 | 0.230 | 0.130 |
| qwen3_8b | -0.4 | Conscientiousness | 1000 | 557 | 240 | 203 | 0.557 | 0.240 | 0.203 | 0.398 |
| qwen3_8b | -0.4 | Extraversion | 1000 | 293 | 481 | 226 | 0.293 | 0.481 | 0.226 | -0.243 |
| qwen3_8b | -0.4 | Agreeableness | 1000 | 487 | 335 | 178 | 0.487 | 0.335 | 0.178 | 0.185 |
| qwen3_8b | -0.4 | Neuroticism | 1000 | 294 | 535 | 171 | 0.294 | 0.535 | 0.171 | -0.291 |
| qwen3_8b | -0.4 | Machiavellianism | 1000 | 262 | 527 | 211 | 0.262 | 0.527 | 0.211 | -0.336 |
| qwen3_8b | -0.4 | Narcissism | 1000 | 265 | 566 | 169 | 0.265 | 0.566 | 0.169 | -0.362 |
| qwen3_8b | -0.4 | Psychopathy | 1000 | 231 | 644 | 125 | 0.231 | 0.644 | 0.125 | -0.472 |
| qwen3_8b | 0.0 | Openness | 1000 | 592 | 408 | 0 | 0.592 | 0.408 | 0.000 | 0.184 |
| qwen3_8b | 0.0 | Conscientiousness | 1000 | 737 | 263 | 0 | 0.737 | 0.263 | 0.000 | 0.474 |
| qwen3_8b | 0.0 | Extraversion | 1000 | 405 | 595 | 0 | 0.405 | 0.595 | 0.000 | -0.190 |
| qwen3_8b | 0.0 | Agreeableness | 1000 | 800 | 200 | 0 | 0.800 | 0.200 | 0.000 | 0.600 |
| qwen3_8b | 0.0 | Neuroticism | 1000 | 205 | 795 | 0 | 0.205 | 0.795 | 0.000 | -0.590 |
| qwen3_8b | 0.0 | Machiavellianism | 1000 | 242 | 758 | 0 | 0.242 | 0.758 | 0.000 | -0.516 |
| qwen3_8b | 0.0 | Narcissism | 1000 | 148 | 852 | 0 | 0.148 | 0.852 | 0.000 | -0.704 |
| qwen3_8b | 0.0 | Psychopathy | 1000 | 83 | 917 | 0 | 0.083 | 0.917 | 0.000 | -0.834 |
| llama3b | 0.0 | Openness | 1000 | 614 | 385 | 1 | 0.614 | 0.385 | 0.001 | 0.229 |
| llama3b | 0.0 | Conscientiousness | 1000 | 823 | 177 | 0 | 0.823 | 0.177 | 0.000 | 0.646 |
| llama3b | 0.0 | Extraversion | 1000 | 294 | 706 | 0 | 0.294 | 0.706 | 0.000 | -0.412 |
| llama3b | 0.0 | Agreeableness | 1000 | 769 | 231 | 0 | 0.769 | 0.231 | 0.000 | 0.538 |
| llama3b | 0.0 | Neuroticism | 1000 | 306 | 694 | 0 | 0.306 | 0.694 | 0.000 | -0.388 |
| llama3b | 0.0 | Machiavellianism | 1000 | 141 | 850 | 9 | 0.141 | 0.850 | 0.009 | -0.715 |
| llama3b | 0.0 | Narcissism | 1000 | 82 | 916 | 2 | 0.082 | 0.916 | 0.002 | -0.836 |
| llama3b | 0.0 | Psychopathy | 1000 | 92 | 731 | 177 | 0.092 | 0.731 | 0.177 | -0.776 |
| llama1b | 0.2 | Openness | 1000 | 496 | 504 | 0 | 0.496 | 0.504 | 0.000 | -0.008 |
| llama1b | 0.2 | Conscientiousness | 1000 | 507 | 493 | 0 | 0.507 | 0.493 | 0.000 | 0.014 |
| llama1b | 0.2 | Extraversion | 1000 | 482 | 518 | 0 | 0.482 | 0.518 | 0.000 | -0.036 |
| llama1b | 0.2 | Agreeableness | 1000 | 514 | 486 | 0 | 0.514 | 0.486 | 0.000 | 0.028 |
| llama1b | 0.2 | Neuroticism | 1000 | 455 | 545 | 0 | 0.455 | 0.545 | 0.000 | -0.090 |
| llama1b | 0.2 | Machiavellianism | 1000 | 478 | 522 | 0 | 0.478 | 0.522 | 0.000 | -0.044 |
| llama1b | 0.2 | Narcissism | 1000 | 446 | 554 | 0 | 0.446 | 0.554 | 0.000 | -0.108 |
| llama1b | 0.2 | Psychopathy | 1000 | 458 | 542 | 0 | 0.458 | 0.542 | 0.000 | -0.084 |
| llama1b | -0.2 | Openness | 1000 | 523 | 431 | 46 | 0.523 | 0.431 | 0.046 | 0.096 |
| llama1b | -0.2 | Conscientiousness | 1000 | 566 | 388 | 46 | 0.566 | 0.388 | 0.046 | 0.187 |
| llama1b | -0.2 | Extraversion | 1000 | 393 | 517 | 90 | 0.393 | 0.517 | 0.090 | -0.136 |
| llama1b | -0.2 | Agreeableness | 1000 | 507 | 321 | 172 | 0.507 | 0.321 | 0.172 | 0.225 |
| llama1b | -0.2 | Neuroticism | 1000 | 322 | 639 | 39 | 0.322 | 0.639 | 0.039 | -0.330 |
| llama1b | -0.2 | Machiavellianism | 1000 | 263 | 585 | 152 | 0.263 | 0.585 | 0.152 | -0.380 |
| llama1b | -0.2 | Narcissism | 1000 | 238 | 647 | 115 | 0.238 | 0.647 | 0.115 | -0.462 |
| llama1b | -0.2 | Psychopathy | 1000 | 144 | 781 | 75 | 0.144 | 0.781 | 0.075 | -0.689 |
| qwen3_8b | 0.3 | Openness | 1000 | 595 | 405 | 0 | 0.595 | 0.405 | 0.000 | 0.190 |
| qwen3_8b | 0.3 | Conscientiousness | 1000 | 588 | 412 | 0 | 0.588 | 0.412 | 0.000 | 0.176 |
| qwen3_8b | 0.3 | Extraversion | 1000 | 487 | 513 | 0 | 0.487 | 0.513 | 0.000 | -0.026 |
| qwen3_8b | 0.3 | Agreeableness | 1000 | 671 | 329 | 0 | 0.671 | 0.329 | 0.000 | 0.342 |
| qwen3_8b | 0.3 | Neuroticism | 1000 | 259 | 741 | 0 | 0.259 | 0.741 | 0.000 | -0.482 |
| qwen3_8b | 0.3 | Machiavellianism | 1000 | 311 | 689 | 0 | 0.311 | 0.689 | 0.000 | -0.378 |
| qwen3_8b | 0.3 | Narcissism | 1000 | 213 | 787 | 0 | 0.213 | 0.787 | 0.000 | -0.574 |
| qwen3_8b | 0.3 | Psychopathy | 1000 | 241 | 759 | 0 | 0.241 | 0.759 | 0.000 | -0.518 |
| qwen3_8b | -0.1 | Openness | 1000 | 595 | 405 | 0 | 0.595 | 0.405 | 0.000 | 0.190 |
| qwen3_8b | -0.1 | Conscientiousness | 1000 | 748 | 252 | 0 | 0.748 | 0.252 | 0.000 | 0.496 |
| qwen3_8b | -0.1 | Extraversion | 1000 | 384 | 616 | 0 | 0.384 | 0.616 | 0.000 | -0.232 |
| qwen3_8b | -0.1 | Agreeableness | 1000 | 817 | 183 | 0 | 0.817 | 0.183 | 0.000 | 0.634 |
| qwen3_8b | -0.1 | Neuroticism | 1000 | 220 | 780 | 0 | 0.220 | 0.780 | 0.000 | -0.560 |
| qwen3_8b | -0.1 | Machiavellianism | 1000 | 243 | 757 | 0 | 0.243 | 0.757 | 0.000 | -0.514 |
| qwen3_8b | -0.1 | Narcissism | 1000 | 144 | 856 | 0 | 0.144 | 0.856 | 0.000 | -0.712 |
| qwen3_8b | -0.1 | Psychopathy | 1000 | 45 | 955 | 0 | 0.045 | 0.955 | 0.000 | -0.910 |
| llama3b | 0.3 | Openness | 1000 | 573 | 426 | 1 | 0.573 | 0.426 | 0.001 | 0.147 |
| llama3b | 0.3 | Conscientiousness | 1000 | 712 | 287 | 1 | 0.712 | 0.287 | 0.001 | 0.425 |
| llama3b | 0.3 | Extraversion | 1000 | 414 | 584 | 2 | 0.414 | 0.584 | 0.002 | -0.170 |
| llama3b | 0.3 | Agreeableness | 1000 | 692 | 307 | 1 | 0.692 | 0.307 | 0.001 | 0.385 |
| llama3b | 0.3 | Neuroticism | 1000 | 425 | 570 | 5 | 0.425 | 0.570 | 0.005 | -0.146 |
| llama3b | 0.3 | Machiavellianism | 1000 | 278 | 714 | 8 | 0.278 | 0.714 | 0.008 | -0.440 |
| llama3b | 0.3 | Narcissism | 1000 | 253 | 744 | 3 | 0.253 | 0.744 | 0.003 | -0.492 |
| llama3b | 0.3 | Psychopathy | 1000 | 167 | 731 | 102 | 0.167 | 0.731 | 0.102 | -0.628 |
| llama3b | -0.1 | Openness | 1000 | 578 | 421 | 1 | 0.578 | 0.421 | 0.001 | 0.157 |
| llama3b | -0.1 | Conscientiousness | 1000 | 820 | 180 | 0 | 0.820 | 0.180 | 0.000 | 0.640 |
| llama3b | -0.1 | Extraversion | 1000 | 296 | 704 | 0 | 0.296 | 0.704 | 0.000 | -0.408 |
| llama3b | -0.1 | Agreeableness | 1000 | 757 | 243 | 0 | 0.757 | 0.243 | 0.000 | 0.514 |
| llama3b | -0.1 | Neuroticism | 1000 | 300 | 700 | 0 | 0.300 | 0.700 | 0.000 | -0.400 |
| llama3b | -0.1 | Machiavellianism | 1000 | 183 | 816 | 1 | 0.183 | 0.816 | 0.001 | -0.634 |
| llama3b | -0.1 | Narcissism | 1000 | 100 | 899 | 1 | 0.100 | 0.899 | 0.001 | -0.800 |
| llama3b | -0.1 | Psychopathy | 1000 | 60 | 915 | 25 | 0.060 | 0.915 | 0.025 | -0.877 |
| llama1b | 0.0 | Openness | 1000 | 526 | 474 | 0 | 0.526 | 0.474 | 0.000 | 0.052 |
| llama1b | 0.0 | Conscientiousness | 1000 | 534 | 466 | 0 | 0.534 | 0.466 | 0.000 | 0.068 |
| llama1b | 0.0 | Extraversion | 1000 | 466 | 534 | 0 | 0.466 | 0.534 | 0.000 | -0.068 |
| llama1b | 0.0 | Agreeableness | 1000 | 531 | 469 | 0 | 0.531 | 0.469 | 0.000 | 0.062 |
| llama1b | 0.0 | Neuroticism | 1000 | 400 | 600 | 0 | 0.400 | 0.600 | 0.000 | -0.200 |
| llama1b | 0.0 | Machiavellianism | 1000 | 455 | 545 | 0 | 0.455 | 0.545 | 0.000 | -0.090 |
| llama1b | 0.0 | Narcissism | 1000 | 405 | 595 | 0 | 0.405 | 0.595 | 0.000 | -0.190 |
| llama1b | 0.0 | Psychopathy | 1000 | 422 | 578 | 0 | 0.422 | 0.578 | 0.000 | -0.156 |
| qwen3_8b | -0.3 | Openness | 1000 | 568 | 429 | 3 | 0.568 | 0.429 | 0.003 | 0.139 |
| qwen3_8b | -0.3 | Conscientiousness | 1000 | 740 | 253 | 7 | 0.740 | 0.253 | 0.007 | 0.490 |
| qwen3_8b | -0.3 | Extraversion | 1000 | 392 | 605 | 3 | 0.392 | 0.605 | 0.003 | -0.214 |
| qwen3_8b | -0.3 | Agreeableness | 1000 | 737 | 257 | 6 | 0.737 | 0.257 | 0.006 | 0.483 |
| qwen3_8b | -0.3 | Neuroticism | 1000 | 271 | 723 | 6 | 0.271 | 0.723 | 0.006 | -0.455 |
| qwen3_8b | -0.3 | Machiavellianism | 1000 | 295 | 703 | 2 | 0.295 | 0.703 | 0.002 | -0.409 |
| qwen3_8b | -0.3 | Narcissism | 1000 | 211 | 789 | 0 | 0.211 | 0.789 | 0.000 | -0.578 |
| qwen3_8b | -0.3 | Psychopathy | 1000 | 75 | 923 | 2 | 0.075 | 0.923 | 0.002 | -0.850 |
| qwen3_8b | 0.1 | Openness | 1000 | 594 | 406 | 0 | 0.594 | 0.406 | 0.000 | 0.188 |
| qwen3_8b | 0.1 | Conscientiousness | 1000 | 724 | 276 | 0 | 0.724 | 0.276 | 0.000 | 0.448 |
| qwen3_8b | 0.1 | Extraversion | 1000 | 395 | 605 | 0 | 0.395 | 0.605 | 0.000 | -0.210 |
| qwen3_8b | 0.1 | Agreeableness | 1000 | 796 | 204 | 0 | 0.796 | 0.204 | 0.000 | 0.592 |
| qwen3_8b | 0.1 | Neuroticism | 1000 | 200 | 800 | 0 | 0.200 | 0.800 | 0.000 | -0.600 |
| qwen3_8b | 0.1 | Machiavellianism | 1000 | 216 | 784 | 0 | 0.216 | 0.784 | 0.000 | -0.568 |
| qwen3_8b | 0.1 | Narcissism | 1000 | 136 | 864 | 0 | 0.136 | 0.864 | 0.000 | -0.728 |
| qwen3_8b | 0.1 | Psychopathy | 1000 | 89 | 911 | 0 | 0.089 | 0.911 | 0.000 | -0.822 |
| llama3b | 0.1 | Openness | 1000 | 640 | 359 | 1 | 0.640 | 0.359 | 0.001 | 0.281 |
| llama3b | 0.1 | Conscientiousness | 1000 | 788 | 211 | 1 | 0.788 | 0.211 | 0.001 | 0.578 |
| llama3b | 0.1 | Extraversion | 1000 | 328 | 672 | 0 | 0.328 | 0.672 | 0.000 | -0.344 |
| llama3b | 0.1 | Agreeableness | 1000 | 778 | 222 | 0 | 0.778 | 0.222 | 0.000 | 0.556 |
| llama3b | 0.1 | Neuroticism | 1000 | 298 | 701 | 1 | 0.298 | 0.701 | 0.001 | -0.403 |
| llama3b | 0.1 | Machiavellianism | 1000 | 147 | 846 | 7 | 0.147 | 0.846 | 0.007 | -0.704 |
| llama3b | 0.1 | Narcissism | 1000 | 85 | 914 | 1 | 0.085 | 0.914 | 0.001 | -0.830 |
| llama3b | 0.1 | Psychopathy | 1000 | 104 | 676 | 220 | 0.104 | 0.676 | 0.220 | -0.733 |
| llama3b | -0.3 | Openness | 1000 | 86 | 99 | 815 | 0.086 | 0.099 | 0.815 | -0.070 |
| llama3b | -0.3 | Conscientiousness | 1000 | 105 | 87 | 808 | 0.105 | 0.087 | 0.808 | 0.094 |
| llama3b | -0.3 | Extraversion | 1000 | 73 | 87 | 840 | 0.073 | 0.087 | 0.840 | -0.087 |
| llama3b | -0.3 | Agreeableness | 1000 | 86 | 93 | 821 | 0.086 | 0.093 | 0.821 | -0.039 |
| llama3b | -0.3 | Neuroticism | 1000 | 72 | 88 | 840 | 0.072 | 0.088 | 0.840 | -0.100 |
| llama3b | -0.3 | Machiavellianism | 1000 | 126 | 115 | 759 | 0.126 | 0.115 | 0.759 | 0.046 |
| llama3b | -0.3 | Narcissism | 1000 | 72 | 95 | 833 | 0.072 | 0.095 | 0.833 | -0.138 |
| llama3b | -0.3 | Psychopathy | 1000 | 102 | 145 | 753 | 0.102 | 0.145 | 0.753 | -0.174 |