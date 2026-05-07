# Llama/Qwen Result Summary

Generated from synced raw artifacts under `docs/results/raw/remote_sync/`.

Primary CSV outputs:
- `all_module_summary_wide.csv`: one row per model/alpha with standard, BFCL, judge, CoT, HellaSwag, and IFEval metrics.
- `all_module_metrics_long.csv`: long-form standard/BFCL/judge/standard-cot metrics.
- `all_lm_eval_subtasks_long.csv`: long-form MMLU, GPQA, AGIEval, HellaSwag, and IFEval metrics.
- `all_mmlu_subtasks_alpha.csv`, `all_gpqa_subtasks_alpha.csv`, `all_agieval_subtasks_alpha.csv`: requested subtask alpha-gradient tables.

## Coverage

All expected model/alpha/module combinations are present.

## Main Module Summary

| model | alpha | GSM8K | MATH | ARC | DROP | BFCL_retry | TruthfulQA | HaluEval | AdvBench | GSM8K_cot | MATH_cot | HellaSwag | IFEval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama1b | -0.300 | 0.150 | 0.160 | 0.230 | 0.120 | 0.010 | 0.490 | 0.500 | 2.700 | 0.150 | 0.160 | 0.560 | 0.257 |
| llama1b | -0.200 | 0.200 | 0.210 | 0.350 | 0.250 | 0.085 | 0.580 | 0.890 | 2.740 | 0.310 | 0.190 | 0.587 | 0.336 |
| llama1b | -0.100 | 0.170 | 0.280 | 0.430 | 0.240 | 0.090 | 0.560 | 1.210 | 2.830 | 0.330 | 0.280 | 0.604 | 0.396 |
| llama1b | 0.000 | 0.190 | 0.190 | 0.480 | 0.230 | 0.010 | 0.700 | 1.360 | 2.830 | 0.370 | 0.310 | 0.617 | 0.442 |
| llama1b | 0.100 | 0.080 | 0.120 | 0.490 | 0.230 | 0.045 | 0.710 | 1.350 | 2.750 | 0.280 | 0.370 | 0.616 | 0.412 |
| llama1b | 0.200 | 0.010 | 0.120 | 0.430 | 0.130 | 0.005 | 0.620 | 1.110 | 2.790 | 0.200 | 0.140 | 0.602 | 0.368 |
| llama1b | 0.300 | 0.010 | 0.100 | 0.080 | 0.110 | 0.000 | 0.620 | 0.780 | 2.760 | 0.030 | 0.020 | 0.555 | 0.274 |
| llama3b | -0.300 | 0.010 | 0.040 | 0.000 | 0.020 | 0.000 | 0.200 | 0.040 | 2.870 | 0.000 | 0.000 | 0.576 | 0.209 |
| llama3b | -0.200 | 0.200 | 0.190 | 0.360 | 0.210 | 0.345 | 0.760 | 1.230 | 2.850 | 0.490 | 0.190 | 0.675 | 0.383 |
| llama3b | -0.100 | 0.120 | 0.200 | 0.670 | 0.490 | 0.380 | 0.940 | 1.620 | 2.910 | 0.700 | 0.420 | 0.709 | 0.457 |
| llama3b | 0.000 | 0.200 | 0.210 | 0.690 | 0.460 | 0.095 | 1.000 | 1.660 | 2.970 | 0.720 | 0.460 | 0.716 | 0.519 |
| llama3b | 0.100 | 0.150 | 0.240 | 0.700 | 0.460 | 0.040 | 1.150 | 1.690 | 2.950 | 0.670 | 0.450 | 0.707 | 0.505 |
| llama3b | 0.200 | 0.150 | 0.230 | 0.590 | 0.490 | 0.040 | 1.010 | 1.680 | 2.960 | 0.570 | 0.340 | 0.667 | 0.455 |
| llama3b | 0.300 | 0.110 | 0.100 | 0.190 | 0.230 | 0.000 | 0.620 | 1.340 | 2.950 | 0.130 | 0.090 | 0.549 | 0.307 |
| llama8b | -0.300 | 0.020 | 0.010 | 0.000 | 0.010 | 0.000 | 0.480 | 0.270 | 2.110 | 0.000 | 0.010 | 0.592 | 0.139 |
| llama8b | -0.200 | 0.240 | 0.210 | 0.370 | 0.330 | 0.465 | 0.990 | 1.600 | 2.970 | 0.580 | 0.150 | 0.774 | 0.372 |
| llama8b | -0.100 | 0.470 | 0.250 | 0.380 | 0.570 | 0.180 | 1.230 | 1.740 | 2.870 | 0.830 | 0.430 | 0.789 | 0.407 |
| llama8b | 0.000 | 0.320 | 0.260 | 0.370 | 0.680 | 0.150 | 1.280 | 1.800 | 2.860 | 0.770 | 0.510 | 0.795 | 0.464 |
| llama8b | 0.100 | 0.290 | 0.280 | 0.520 | 0.650 | 0.205 | 1.280 | 1.690 | 2.940 | 0.770 | 0.500 | 0.785 | 0.481 |
| llama8b | 0.200 | 0.150 | 0.190 | 0.710 | 0.580 | 0.305 | 1.110 | 1.480 | 2.940 | 0.650 | 0.250 | 0.752 | 0.440 |
| llama8b | 0.300 | 0.000 | 0.070 | 0.010 | 0.050 | 0.000 | 0.710 | 0.500 | 2.870 | 0.000 | 0.060 | 0.519 | 0.201 |
| qwen3_8b | -0.400 | 0.130 | 0.150 | 0.700 | 0.210 | 0.015 | 0.600 | 0.850 | 2.810 | 0.020 | 0.010 | 0.533 | 0.170 |
| qwen3_8b | -0.300 | 0.450 | 0.470 | 0.860 | 0.430 | 0.355 | 0.820 | 1.240 | 2.810 | 0.030 | 0.020 | 0.653 | 0.181 |
| qwen3_8b | -0.200 | 0.810 | 0.490 | 0.900 | 0.610 | 0.415 | 0.870 | 1.360 | 2.920 | 0.660 | 0.500 | 0.723 | 0.270 |
| qwen3_8b | -0.100 | 0.840 | 0.450 | 0.900 | 0.630 | 0.395 | 0.860 | 1.390 | 2.890 | 0.920 | 0.720 | 0.745 | 0.261 |
| qwen3_8b | 0.000 | 0.870 | 0.420 | 0.890 | 0.680 | 0.510 | 0.930 | 1.350 | 2.930 | 0.960 | 0.790 | 0.749 | 0.248 |
| qwen3_8b | 0.100 | 0.800 | 0.380 | 0.910 | 0.730 | 0.580 | 1.000 | 1.410 | 2.860 | 0.970 | 0.800 | 0.738 | 0.327 |
| qwen3_8b | 0.200 | 0.710 | 0.350 | 0.850 | 0.600 | 0.270 | 0.980 | 1.410 | 2.830 | 0.800 | 0.610 | 0.694 | 0.274 |
| qwen3_8b | 0.300 | 0.250 | 0.210 | 0.700 | 0.270 | 0.265 | 0.790 | 1.090 | 2.890 | 0.260 | 0.190 | 0.613 | 0.201 |
| qwen3_8b | 0.400 | 0.080 | 0.150 | 0.270 | 0.060 | 0.015 | 0.770 | 0.510 | 2.380 | 0.080 | 0.020 | 0.520 | 0.165 |

## MMLU Subtask Alpha Table

| model | group | task | metric | alpha_-0.4 | alpha_-0.3 | alpha_-0.2 | alpha_-0.1 | alpha_0 | alpha_0.1 | alpha_0.2 | alpha_0.3 | alpha_0.4 | best_alpha | best_value | delta_best_vs_0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama1b | mmlu | mmlu_abstract_algebra | acc,none |  | 0.330 | 0.340 | 0.340 | 0.280 | 0.230 | 0.300 | 0.260 |  | -0.200 | 0.340 | 0.060 |
| llama1b | mmlu | mmlu_clinical_knowledge | acc,none |  | 0.487 | 0.513 | 0.528 | 0.532 | 0.506 | 0.483 | 0.366 |  | 0.000 | 0.532 | 0.000 |
| llama1b | mmlu | mmlu_college_biology | acc,none |  | 0.347 | 0.403 | 0.472 | 0.507 | 0.500 | 0.465 | 0.361 |  | 0.000 | 0.507 | 0.000 |
| llama1b | mmlu | mmlu_college_chemistry | acc,none |  | 0.420 | 0.430 | 0.400 | 0.390 | 0.300 | 0.260 | 0.170 |  | -0.200 | 0.430 | 0.040 |
| llama1b | mmlu | mmlu_college_computer_science | acc,none |  | 0.320 | 0.340 | 0.370 | 0.340 | 0.370 | 0.260 | 0.260 |  | -0.100 | 0.370 | 0.030 |
| llama1b | mmlu | mmlu_college_mathematics | acc,none |  | 0.320 | 0.330 | 0.320 | 0.340 | 0.320 | 0.280 | 0.250 |  | 0.000 | 0.340 | 0.000 |
| llama1b | mmlu | mmlu_college_physics | acc,none |  | 0.314 | 0.284 | 0.324 | 0.284 | 0.294 | 0.265 | 0.186 |  | -0.100 | 0.324 | 0.039 |
| llama1b | mmlu | mmlu_computer_security | acc,none |  | 0.450 | 0.470 | 0.520 | 0.560 | 0.570 | 0.550 | 0.490 |  | 0.100 | 0.570 | 0.010 |
| llama1b | mmlu | mmlu_elementary_mathematics | acc,none |  | 0.317 | 0.323 | 0.320 | 0.341 | 0.341 | 0.317 | 0.304 |  | 0.000 | 0.341 | 0.000 |
| llama1b | mmlu | mmlu_formal_logic | acc,none |  | 0.389 | 0.381 | 0.341 | 0.341 | 0.365 | 0.294 | 0.183 |  | -0.300 | 0.389 | 0.048 |
| llama1b | mmlu | mmlu_high_school_mathematics | acc,none |  | 0.278 | 0.285 | 0.311 | 0.326 | 0.281 | 0.281 | 0.252 |  | 0.000 | 0.326 | 0.000 |
| llama1b | mmlu | mmlu_high_school_physics | acc,none |  | 0.311 | 0.278 | 0.245 | 0.272 | 0.311 | 0.258 | 0.219 |  | -0.300 | 0.311 | 0.040 |
| llama1b | mmlu | mmlu_international_law | acc,none |  | 0.446 | 0.537 | 0.653 | 0.686 | 0.678 | 0.579 | 0.521 |  | 0.000 | 0.686 | 0.000 |
| llama1b | mmlu | mmlu_logical_fallacies | acc,none |  | 0.399 | 0.442 | 0.460 | 0.509 | 0.460 | 0.436 | 0.374 |  | 0.000 | 0.509 | 0.000 |
| llama1b | mmlu | mmlu_machine_learning | acc,none |  | 0.205 | 0.214 | 0.250 | 0.295 | 0.357 | 0.348 | 0.286 |  | 0.100 | 0.357 | 0.062 |
| llama1b | mmlu | mmlu_miscellaneous | acc,none |  | 0.522 | 0.582 | 0.632 | 0.656 | 0.642 | 0.591 | 0.483 |  | 0.000 | 0.656 | 0.000 |
| llama1b | mmlu | mmlu_moral_disputes | acc,none |  | 0.347 | 0.410 | 0.431 | 0.468 | 0.494 | 0.500 | 0.405 |  | 0.200 | 0.500 | 0.032 |
| llama1b | mmlu | mmlu_philosophy | acc,none |  | 0.457 | 0.502 | 0.540 | 0.514 | 0.492 | 0.457 | 0.424 |  | -0.100 | 0.540 | 0.026 |
| llama1b | mmlu | mmlu_professional_law | acc,none |  | 0.287 | 0.313 | 0.349 | 0.370 | 0.370 | 0.317 | 0.280 |  | 0.000 | 0.370 | 0.000 |
| llama1b | mmlu | mmlu_professional_medicine | acc,none |  | 0.493 | 0.529 | 0.570 | 0.574 | 0.529 | 0.412 | 0.287 |  | 0.000 | 0.574 | 0.000 |
| llama3b | mmlu | mmlu_abstract_algebra | acc,none |  | 0.230 | 0.290 | 0.310 | 0.350 | 0.340 | 0.310 | 0.310 |  | 0.000 | 0.350 | 0.000 |
| llama3b | mmlu | mmlu_clinical_knowledge | acc,none |  | 0.377 | 0.608 | 0.683 | 0.672 | 0.642 | 0.551 | 0.426 |  | -0.100 | 0.683 | 0.011 |
| llama3b | mmlu | mmlu_college_biology | acc,none |  | 0.326 | 0.583 | 0.694 | 0.715 | 0.694 | 0.604 | 0.389 |  | 0.000 | 0.715 | 0.000 |
| llama3b | mmlu | mmlu_college_chemistry | acc,none |  | 0.300 | 0.380 | 0.440 | 0.410 | 0.350 | 0.330 | 0.310 |  | -0.100 | 0.440 | 0.030 |
| llama3b | mmlu | mmlu_college_computer_science | acc,none |  | 0.240 | 0.230 | 0.400 | 0.530 | 0.450 | 0.390 | 0.240 |  | 0.000 | 0.530 | 0.000 |
| llama3b | mmlu | mmlu_college_mathematics | acc,none |  | 0.270 | 0.240 | 0.290 | 0.300 | 0.320 | 0.300 | 0.260 |  | 0.100 | 0.320 | 0.020 |
| llama3b | mmlu | mmlu_college_physics | acc,none |  | 0.275 | 0.284 | 0.324 | 0.382 | 0.343 | 0.343 | 0.314 |  | 0.000 | 0.382 | 0.000 |
| llama3b | mmlu | mmlu_computer_security | acc,none |  | 0.340 | 0.570 | 0.690 | 0.670 | 0.620 | 0.570 | 0.460 |  | -0.100 | 0.690 | 0.020 |
| llama3b | mmlu | mmlu_elementary_mathematics | acc,none |  | 0.278 | 0.339 | 0.407 | 0.437 | 0.413 | 0.384 | 0.307 |  | 0.000 | 0.437 | 0.000 |
| llama3b | mmlu | mmlu_formal_logic | acc,none |  | 0.302 | 0.365 | 0.421 | 0.421 | 0.310 | 0.270 | 0.238 |  | -0.100 | 0.421 | 0.000 |
| llama3b | mmlu | mmlu_high_school_mathematics | acc,none |  | 0.267 | 0.270 | 0.319 | 0.374 | 0.337 | 0.319 | 0.281 |  | 0.000 | 0.374 | 0.000 |
| llama3b | mmlu | mmlu_high_school_physics | acc,none |  | 0.265 | 0.285 | 0.364 | 0.377 | 0.338 | 0.377 | 0.311 |  | 0.000 | 0.377 | 0.000 |
| llama3b | mmlu | mmlu_international_law | acc,none |  | 0.372 | 0.661 | 0.694 | 0.727 | 0.719 | 0.719 | 0.570 |  | 0.000 | 0.727 | 0.000 |
| llama3b | mmlu | mmlu_logical_fallacies | acc,none |  | 0.337 | 0.620 | 0.724 | 0.748 | 0.693 | 0.607 | 0.491 |  | 0.000 | 0.748 | 0.000 |
| llama3b | mmlu | mmlu_machine_learning | acc,none |  | 0.348 | 0.455 | 0.464 | 0.455 | 0.357 | 0.357 | 0.321 |  | -0.100 | 0.464 | 0.009 |
| llama3b | mmlu | mmlu_miscellaneous | acc,none |  | 0.490 | 0.714 | 0.774 | 0.783 | 0.762 | 0.695 | 0.552 |  | 0.000 | 0.783 | 0.000 |
| llama3b | mmlu | mmlu_moral_disputes | acc,none |  | 0.329 | 0.514 | 0.645 | 0.682 | 0.618 | 0.575 | 0.384 |  | 0.000 | 0.682 | 0.000 |
| llama3b | mmlu | mmlu_philosophy | acc,none |  | 0.354 | 0.566 | 0.653 | 0.672 | 0.656 | 0.592 | 0.466 |  | 0.000 | 0.672 | 0.000 |
| llama3b | mmlu | mmlu_professional_law | acc,none |  | 0.260 | 0.361 | 0.426 | 0.471 | 0.435 | 0.395 | 0.321 |  | 0.000 | 0.471 | 0.000 |
| llama3b | mmlu | mmlu_professional_medicine | acc,none |  | 0.349 | 0.555 | 0.761 | 0.765 | 0.732 | 0.640 | 0.419 |  | 0.000 | 0.765 | 0.000 |
| llama8b | mmlu | mmlu_abstract_algebra | acc,none |  | 0.260 | 0.320 | 0.370 | 0.350 | 0.400 | 0.310 | 0.260 |  | 0.100 | 0.400 | 0.050 |
| llama8b | mmlu | mmlu_clinical_knowledge | acc,none |  | 0.347 | 0.721 | 0.777 | 0.781 | 0.758 | 0.683 | 0.283 |  | 0.000 | 0.781 | 0.000 |
| llama8b | mmlu | mmlu_college_biology | acc,none |  | 0.285 | 0.674 | 0.799 | 0.806 | 0.806 | 0.722 | 0.368 |  | 0.000 | 0.806 | 0.000 |
| llama8b | mmlu | mmlu_college_chemistry | acc,none |  | 0.210 | 0.470 | 0.480 | 0.470 | 0.430 | 0.380 | 0.240 |  | -0.100 | 0.480 | 0.010 |
| llama8b | mmlu | mmlu_college_computer_science | acc,none |  | 0.300 | 0.460 | 0.460 | 0.560 | 0.560 | 0.460 | 0.230 |  | 0.000 | 0.560 | 0.000 |
| llama8b | mmlu | mmlu_college_mathematics | acc,none |  | 0.280 | 0.400 | 0.370 | 0.340 | 0.390 | 0.350 | 0.290 |  | -0.200 | 0.400 | 0.060 |
| llama8b | mmlu | mmlu_college_physics | acc,none |  | 0.235 | 0.402 | 0.451 | 0.451 | 0.441 | 0.402 | 0.294 |  | -0.100 | 0.451 | 0.000 |
| llama8b | mmlu | mmlu_computer_security | acc,none |  | 0.410 | 0.710 | 0.760 | 0.780 | 0.750 | 0.710 | 0.270 |  | 0.000 | 0.780 | 0.000 |
| llama8b | mmlu | mmlu_elementary_mathematics | acc,none |  | 0.251 | 0.450 | 0.484 | 0.492 | 0.474 | 0.407 | 0.296 |  | 0.000 | 0.492 | 0.000 |
| llama8b | mmlu | mmlu_formal_logic | acc,none |  | 0.246 | 0.341 | 0.452 | 0.492 | 0.516 | 0.365 | 0.286 |  | 0.100 | 0.516 | 0.024 |
| llama8b | mmlu | mmlu_high_school_mathematics | acc,none |  | 0.267 | 0.311 | 0.381 | 0.419 | 0.463 | 0.344 | 0.274 |  | 0.100 | 0.463 | 0.044 |
| llama8b | mmlu | mmlu_high_school_physics | acc,none |  | 0.219 | 0.344 | 0.417 | 0.450 | 0.444 | 0.384 | 0.325 |  | 0.000 | 0.450 | 0.000 |
| llama8b | mmlu | mmlu_international_law | acc,none |  | 0.372 | 0.802 | 0.835 | 0.793 | 0.785 | 0.727 | 0.207 |  | -0.100 | 0.835 | 0.041 |
| llama8b | mmlu | mmlu_logical_fallacies | acc,none |  | 0.331 | 0.755 | 0.791 | 0.816 | 0.773 | 0.687 | 0.221 |  | 0.000 | 0.816 | 0.000 |
| llama8b | mmlu | mmlu_machine_learning | acc,none |  | 0.268 | 0.518 | 0.589 | 0.500 | 0.473 | 0.348 | 0.241 |  | -0.100 | 0.589 | 0.089 |
| llama8b | mmlu | mmlu_miscellaneous | acc,none |  | 0.434 | 0.791 | 0.835 | 0.845 | 0.839 | 0.780 | 0.345 |  | 0.000 | 0.845 | 0.000 |
| llama8b | mmlu | mmlu_moral_disputes | acc,none |  | 0.298 | 0.650 | 0.725 | 0.734 | 0.714 | 0.575 | 0.263 |  | 0.000 | 0.734 | 0.000 |
| llama8b | mmlu | mmlu_philosophy | acc,none |  | 0.344 | 0.682 | 0.723 | 0.733 | 0.701 | 0.675 | 0.222 |  | 0.000 | 0.733 | 0.000 |
| llama8b | mmlu | mmlu_professional_law | acc,none |  | 0.259 | 0.434 | 0.499 | 0.505 | 0.480 | 0.404 | 0.258 |  | 0.000 | 0.505 | 0.000 |
| llama8b | mmlu | mmlu_professional_medicine | acc,none |  | 0.210 | 0.629 | 0.757 | 0.783 | 0.750 | 0.699 | 0.375 |  | 0.000 | 0.783 | 0.000 |
| qwen3_8b | mmlu | mmlu_abstract_algebra | acc,none | 0.260 | 0.490 | 0.530 | 0.580 | 0.570 | 0.570 | 0.540 | 0.400 | 0.300 | -0.100 | 0.580 | 0.010 |
| qwen3_8b | mmlu | mmlu_clinical_knowledge | acc,none | 0.298 | 0.691 | 0.766 | 0.777 | 0.789 | 0.781 | 0.766 | 0.683 | 0.445 | 0.000 | 0.789 | 0.000 |
| qwen3_8b | mmlu | mmlu_college_biology | acc,none | 0.375 | 0.819 | 0.854 | 0.875 | 0.854 | 0.854 | 0.819 | 0.667 | 0.389 | -0.100 | 0.875 | 0.021 |
| qwen3_8b | mmlu | mmlu_college_chemistry | acc,none | 0.340 | 0.480 | 0.560 | 0.570 | 0.590 | 0.570 | 0.550 | 0.520 | 0.320 | 0.000 | 0.590 | 0.000 |
| qwen3_8b | mmlu | mmlu_college_computer_science | acc,none | 0.270 | 0.540 | 0.660 | 0.670 | 0.730 | 0.720 | 0.660 | 0.530 | 0.380 | 0.000 | 0.730 | 0.000 |
| qwen3_8b | mmlu | mmlu_college_mathematics | acc,none | 0.330 | 0.450 | 0.560 | 0.600 | 0.590 | 0.590 | 0.560 | 0.430 | 0.340 | -0.100 | 0.600 | 0.010 |
| qwen3_8b | mmlu | mmlu_college_physics | acc,none | 0.333 | 0.510 | 0.520 | 0.569 | 0.559 | 0.598 | 0.520 | 0.441 | 0.333 | 0.100 | 0.598 | 0.039 |
| qwen3_8b | mmlu | mmlu_computer_security | acc,none | 0.360 | 0.750 | 0.820 | 0.840 | 0.820 | 0.820 | 0.800 | 0.740 | 0.550 | -0.100 | 0.840 | 0.020 |
| qwen3_8b | mmlu | mmlu_elementary_mathematics | acc,none | 0.280 | 0.601 | 0.677 | 0.706 | 0.696 | 0.709 | 0.646 | 0.540 | 0.362 | 0.100 | 0.709 | 0.013 |
| qwen3_8b | mmlu | mmlu_formal_logic | acc,none | 0.373 | 0.611 | 0.643 | 0.651 | 0.611 | 0.587 | 0.548 | 0.444 | 0.254 | -0.100 | 0.651 | 0.040 |
| qwen3_8b | mmlu | mmlu_high_school_mathematics | acc,none | 0.315 | 0.496 | 0.511 | 0.504 | 0.515 | 0.541 | 0.504 | 0.452 | 0.322 | 0.100 | 0.541 | 0.026 |
| qwen3_8b | mmlu | mmlu_high_school_physics | acc,none | 0.404 | 0.530 | 0.649 | 0.709 | 0.702 | 0.682 | 0.570 | 0.464 | 0.344 | -0.100 | 0.709 | 0.007 |
| qwen3_8b | mmlu | mmlu_international_law | acc,none | 0.248 | 0.727 | 0.785 | 0.826 | 0.826 | 0.777 | 0.760 | 0.686 | 0.537 | -0.100 | 0.826 | 0.000 |
| qwen3_8b | mmlu | mmlu_logical_fallacies | acc,none | 0.282 | 0.773 | 0.822 | 0.847 | 0.840 | 0.822 | 0.779 | 0.613 | 0.405 | -0.100 | 0.847 | 0.006 |
| qwen3_8b | mmlu | mmlu_machine_learning | acc,none | 0.259 | 0.455 | 0.607 | 0.625 | 0.589 | 0.571 | 0.518 | 0.393 | 0.259 | -0.100 | 0.625 | 0.036 |
| qwen3_8b | mmlu | mmlu_miscellaneous | acc,none | 0.327 | 0.764 | 0.842 | 0.852 | 0.856 | 0.840 | 0.812 | 0.708 | 0.476 | 0.000 | 0.856 | 0.000 |
| qwen3_8b | mmlu | mmlu_moral_disputes | acc,none | 0.277 | 0.659 | 0.697 | 0.728 | 0.740 | 0.723 | 0.699 | 0.633 | 0.483 | 0.000 | 0.740 | 0.000 |
| qwen3_8b | mmlu | mmlu_philosophy | acc,none | 0.270 | 0.672 | 0.740 | 0.788 | 0.785 | 0.762 | 0.707 | 0.653 | 0.457 | -0.100 | 0.788 | 0.003 |
| qwen3_8b | mmlu | mmlu_professional_law | acc,none | 0.321 | 0.439 | 0.484 | 0.510 | 0.515 | 0.505 | 0.462 | 0.370 | 0.287 | 0.000 | 0.515 | 0.000 |
| qwen3_8b | mmlu | mmlu_professional_medicine | acc,none | 0.357 | 0.676 | 0.772 | 0.783 | 0.816 | 0.827 | 0.724 | 0.577 | 0.301 | 0.100 | 0.827 | 0.011 |

## GPQA Subtask Alpha Table

| model | group | task | metric | alpha_-0.4 | alpha_-0.3 | alpha_-0.2 | alpha_-0.1 | alpha_0 | alpha_0.1 | alpha_0.2 | alpha_0.3 | alpha_0.4 | best_alpha | best_value | delta_best_vs_0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama1b | gpqa | gpqa_diamond_zeroshot | acc_norm,none |  | 0.222 | 0.247 | 0.258 | 0.298 | 0.263 | 0.258 | 0.283 |  | 0.000 | 0.298 | 0.000 |
| llama1b | gpqa | gpqa_extended_zeroshot | acc_norm,none |  | 0.218 | 0.227 | 0.209 | 0.231 | 0.267 | 0.266 | 0.267 |  | 0.100 | 0.267 | 0.037 |
| llama1b | gpqa | gpqa_main_zeroshot | acc_norm,none |  | 0.310 | 0.295 | 0.268 | 0.259 | 0.263 | 0.250 | 0.259 |  | -0.300 | 0.310 | 0.051 |
| llama3b | gpqa | gpqa_diamond_zeroshot | acc_norm,none |  | 0.237 | 0.273 | 0.323 | 0.308 | 0.318 | 0.313 | 0.298 |  | -0.100 | 0.323 | 0.015 |
| llama3b | gpqa | gpqa_extended_zeroshot | acc_norm,none |  | 0.234 | 0.238 | 0.266 | 0.300 | 0.315 | 0.319 | 0.273 |  | 0.200 | 0.319 | 0.018 |
| llama3b | gpqa | gpqa_main_zeroshot | acc_norm,none |  | 0.268 | 0.299 | 0.321 | 0.310 | 0.317 | 0.286 | 0.246 |  | -0.100 | 0.321 | 0.011 |
| llama8b | gpqa | gpqa_diamond_zeroshot | acc_norm,none |  | 0.268 | 0.258 | 0.328 | 0.308 | 0.328 | 0.258 | 0.298 |  | -0.100 | 0.328 | 0.020 |
| llama8b | gpqa | gpqa_extended_zeroshot | acc_norm,none |  | 0.267 | 0.295 | 0.304 | 0.289 | 0.291 | 0.278 | 0.236 |  | -0.100 | 0.304 | 0.015 |
| llama8b | gpqa | gpqa_main_zeroshot | acc_norm,none |  | 0.243 | 0.281 | 0.346 | 0.344 | 0.321 | 0.288 | 0.243 |  | -0.100 | 0.346 | 0.002 |
| qwen3_8b | gpqa | gpqa_diamond_zeroshot | acc_norm,none | 0.268 | 0.338 | 0.364 | 0.389 | 0.409 | 0.394 | 0.333 | 0.288 | 0.283 | 0.000 | 0.409 | 0.000 |
| qwen3_8b | gpqa | gpqa_extended_zeroshot | acc_norm,none | 0.293 | 0.330 | 0.359 | 0.364 | 0.368 | 0.372 | 0.324 | 0.277 | 0.267 | 0.100 | 0.372 | 0.004 |
| qwen3_8b | gpqa | gpqa_main_zeroshot | acc_norm,none | 0.257 | 0.321 | 0.342 | 0.350 | 0.353 | 0.373 | 0.346 | 0.283 | 0.268 | 0.100 | 0.373 | 0.020 |

## AGIEval Subtask Alpha Table

| model | group | task | metric | alpha_-0.4 | alpha_-0.3 | alpha_-0.2 | alpha_-0.1 | alpha_0 | alpha_0.1 | alpha_0.2 | alpha_0.3 | alpha_0.4 | best_alpha | best_value | delta_best_vs_0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama1b | agieval | agieval_aqua_rat | acc_norm,none |  | 0.240 | 0.264 | 0.256 | 0.240 | 0.217 | 0.205 | 0.193 |  | -0.200 | 0.264 | 0.024 |
| llama1b | agieval | agieval_logiqa_en | acc_norm,none |  | 0.272 | 0.269 | 0.300 | 0.293 | 0.298 | 0.327 | 0.304 |  | 0.200 | 0.327 | 0.034 |
| llama1b | agieval | agieval_lsat_ar | acc_norm,none |  | 0.200 | 0.187 | 0.204 | 0.157 | 0.139 | 0.178 | 0.226 |  | 0.300 | 0.226 | 0.070 |
| llama1b | agieval | agieval_lsat_lr | acc_norm,none |  | 0.235 | 0.233 | 0.251 | 0.261 | 0.233 | 0.214 | 0.208 |  | 0.000 | 0.261 | 0.000 |
| llama1b | agieval | agieval_lsat_rc | acc_norm,none |  | 0.223 | 0.212 | 0.204 | 0.197 | 0.197 | 0.204 | 0.197 |  | -0.300 | 0.223 | 0.026 |
| llama1b | agieval | agieval_sat_en | acc_norm,none |  | 0.223 | 0.272 | 0.257 | 0.286 | 0.296 | 0.238 | 0.228 |  | 0.100 | 0.296 | 0.010 |
| llama1b | agieval | agieval_sat_en_without_passage | acc_norm,none |  | 0.248 | 0.233 | 0.262 | 0.262 | 0.252 | 0.228 | 0.199 |  | -0.100 | 0.262 | 0.000 |
| llama1b | agieval | agieval_sat_math | acc_norm,none |  | 0.264 | 0.255 | 0.255 | 0.277 | 0.291 | 0.241 | 0.245 |  | 0.100 | 0.291 | 0.014 |
| llama3b | agieval | agieval_aqua_rat | acc_norm,none |  | 0.205 | 0.224 | 0.220 | 0.209 | 0.181 | 0.169 | 0.146 |  | -0.200 | 0.224 | 0.016 |
| llama3b | agieval | agieval_logiqa_en | acc_norm,none |  | 0.267 | 0.280 | 0.332 | 0.339 | 0.359 | 0.335 | 0.310 |  | 0.100 | 0.359 | 0.020 |
| llama3b | agieval | agieval_lsat_ar | acc_norm,none |  | 0.196 | 0.165 | 0.200 | 0.191 | 0.191 | 0.209 | 0.226 |  | 0.300 | 0.226 | 0.035 |
| llama3b | agieval | agieval_lsat_lr | acc_norm,none |  | 0.241 | 0.233 | 0.275 | 0.282 | 0.280 | 0.276 | 0.218 |  | 0.000 | 0.282 | 0.000 |
| llama3b | agieval | agieval_lsat_rc | acc_norm,none |  | 0.182 | 0.230 | 0.331 | 0.361 | 0.323 | 0.320 | 0.178 |  | 0.000 | 0.361 | 0.000 |
| llama3b | agieval | agieval_sat_en | acc_norm,none |  | 0.218 | 0.306 | 0.422 | 0.490 | 0.432 | 0.413 | 0.204 |  | 0.000 | 0.490 | 0.000 |
| llama3b | agieval | agieval_sat_en_without_passage | acc_norm,none |  | 0.257 | 0.320 | 0.306 | 0.291 | 0.262 | 0.252 | 0.180 |  | -0.200 | 0.320 | 0.029 |
| llama3b | agieval | agieval_sat_math | acc_norm,none |  | 0.205 | 0.241 | 0.323 | 0.323 | 0.345 | 0.318 | 0.273 |  | 0.100 | 0.345 | 0.023 |
| llama8b | agieval | agieval_aqua_rat | acc_norm,none |  | 0.248 | 0.209 | 0.228 | 0.244 | 0.209 | 0.205 | 0.256 |  | 0.300 | 0.256 | 0.012 |
| llama8b | agieval | agieval_logiqa_en | acc_norm,none |  | 0.250 | 0.359 | 0.376 | 0.375 | 0.376 | 0.332 | 0.272 |  | -0.100 | 0.376 | 0.002 |
| llama8b | agieval | agieval_lsat_ar | acc_norm,none |  | 0.209 | 0.183 | 0.200 | 0.196 | 0.204 | 0.174 | 0.239 |  | 0.300 | 0.239 | 0.043 |
| llama8b | agieval | agieval_lsat_lr | acc_norm,none |  | 0.204 | 0.410 | 0.416 | 0.424 | 0.357 | 0.294 | 0.184 |  | 0.000 | 0.424 | 0.000 |
| llama8b | agieval | agieval_lsat_rc | acc_norm,none |  | 0.167 | 0.323 | 0.546 | 0.535 | 0.506 | 0.401 | 0.171 |  | -0.100 | 0.546 | 0.011 |
| llama8b | agieval | agieval_sat_en | acc_norm,none |  | 0.194 | 0.277 | 0.660 | 0.684 | 0.694 | 0.549 | 0.228 |  | 0.100 | 0.694 | 0.010 |
| llama8b | agieval | agieval_sat_en_without_passage | acc_norm,none |  | 0.228 | 0.403 | 0.379 | 0.354 | 0.282 | 0.262 | 0.252 |  | -0.200 | 0.403 | 0.049 |
| llama8b | agieval | agieval_sat_math | acc_norm,none |  | 0.218 | 0.282 | 0.336 | 0.332 | 0.341 | 0.295 | 0.232 |  | 0.100 | 0.341 | 0.009 |
| qwen3_8b | agieval | agieval_aqua_rat | acc_norm,none | 0.228 | 0.319 | 0.335 | 0.366 | 0.394 | 0.421 | 0.346 | 0.307 | 0.240 | 0.100 | 0.421 | 0.028 |
| qwen3_8b | agieval | agieval_logiqa_en | acc_norm,none | 0.320 | 0.413 | 0.507 | 0.525 | 0.524 | 0.508 | 0.439 | 0.310 | 0.267 | -0.100 | 0.525 | 0.002 |
| qwen3_8b | agieval | agieval_lsat_ar | acc_norm,none | 0.226 | 0.252 | 0.287 | 0.265 | 0.270 | 0.274 | 0.209 | 0.226 | 0.191 | -0.200 | 0.287 | 0.017 |
| qwen3_8b | agieval | agieval_lsat_lr | acc_norm,none | 0.312 | 0.555 | 0.722 | 0.745 | 0.727 | 0.684 | 0.555 | 0.304 | 0.235 | -0.100 | 0.745 | 0.018 |
| qwen3_8b | agieval | agieval_lsat_rc | acc_norm,none | 0.242 | 0.483 | 0.669 | 0.706 | 0.710 | 0.732 | 0.621 | 0.349 | 0.271 | 0.100 | 0.732 | 0.022 |
| qwen3_8b | agieval | agieval_sat_en | acc_norm,none | 0.320 | 0.646 | 0.845 | 0.850 | 0.864 | 0.845 | 0.791 | 0.437 | 0.267 | 0.000 | 0.864 | 0.000 |
| qwen3_8b | agieval | agieval_sat_en_without_passage | acc_norm,none | 0.320 | 0.374 | 0.485 | 0.558 | 0.563 | 0.529 | 0.485 | 0.393 | 0.354 | 0.000 | 0.563 | 0.000 |
| qwen3_8b | agieval | agieval_sat_math | acc_norm,none | 0.318 | 0.518 | 0.541 | 0.550 | 0.545 | 0.523 | 0.536 | 0.450 | 0.300 | -0.100 | 0.550 | 0.005 |
