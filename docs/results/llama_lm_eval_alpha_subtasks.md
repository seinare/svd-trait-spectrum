# Llama 1B/3B LM-Eval Alpha Subtask Comparison

Source: `docs/results/lm_eval_subtasks_long.csv`. Values are percentages; `best-0` is percentage-point improvement over alpha=0. Metrics are `acc` for MMLU and normalized accuracy for GPQA/AGIEval.

## Group Averages

| model | group | tasks | -0.3 | -0.2 | -0.1 | 0 | 0.1 | 0.2 | 0.3 | best alpha | best | best-0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama1b | mmlu | 20 | 37.2 | 39.5 | 41.9 | 42.9 | 42.1 | 38.3 | 31.8 | 0 | 42.9 | 0.0 |
| llama1b | gpqa | 3 | 25.0 | 25.6 | 24.5 | 26.3 | 26.4 | 25.8 | 27.0 | 0.3 | 27.0 | 0.7 |
| llama1b | agieval | 8 | 23.8 | 24.1 | 24.9 | 24.7 | 24.0 | 22.9 | 22.5 | -0.1 | 24.9 | 0.2 |
| llama3b | mmlu | 20 | 31.5 | 44.5 | 52.4 | 54.7 | 50.6 | 46.6 | 36.9 | 0 | 54.7 | 0.0 |
| llama3b | gpqa | 3 | 24.7 | 27.0 | 30.3 | 30.6 | 31.7 | 30.6 | 27.2 | 0.1 | 31.7 | 1.0 |
| llama3b | agieval | 8 | 22.1 | 25.0 | 30.1 | 31.1 | 29.7 | 28.7 | 21.7 | 0 | 31.1 | 0.0 |

## llama1b

### MMLU

| task | metric | -0.3 | -0.2 | -0.1 | 0 | 0.1 | 0.2 | 0.3 | best alpha | best | best-0 | 0.3-(-0.3) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu_abstract_algebra | `acc,none` | 33.0 | 34.0 | 34.0 | 28.0 | 23.0 | 30.0 | 26.0 | -0.2 | 34.0 | 6.0 | -7.0 |
| mmlu_clinical_knowledge | `acc,none` | 48.7 | 51.3 | 52.8 | 53.2 | 50.6 | 48.3 | 36.6 | 0 | 53.2 | 0.0 | -12.1 |
| mmlu_college_biology | `acc,none` | 34.7 | 40.3 | 47.2 | 50.7 | 50.0 | 46.5 | 36.1 | 0 | 50.7 | 0.0 | 1.4 |
| mmlu_college_chemistry | `acc,none` | 42.0 | 43.0 | 40.0 | 39.0 | 30.0 | 26.0 | 17.0 | -0.2 | 43.0 | 4.0 | -25.0 |
| mmlu_college_computer_science | `acc,none` | 32.0 | 34.0 | 37.0 | 34.0 | 37.0 | 26.0 | 26.0 | -0.1 | 37.0 | 3.0 | -6.0 |
| mmlu_college_mathematics | `acc,none` | 32.0 | 33.0 | 32.0 | 34.0 | 32.0 | 28.0 | 25.0 | 0 | 34.0 | 0.0 | -7.0 |
| mmlu_college_physics | `acc,none` | 31.4 | 28.4 | 32.4 | 28.4 | 29.4 | 26.5 | 18.6 | -0.1 | 32.4 | 3.9 | -12.7 |
| mmlu_computer_security | `acc,none` | 45.0 | 47.0 | 52.0 | 56.0 | 57.0 | 55.0 | 49.0 | 0.1 | 57.0 | 1.0 | 4.0 |
| mmlu_elementary_mathematics | `acc,none` | 31.7 | 32.3 | 32.0 | 34.1 | 34.1 | 31.7 | 30.4 | 0 | 34.1 | 0.0 | -1.3 |
| mmlu_formal_logic | `acc,none` | 38.9 | 38.1 | 34.1 | 34.1 | 36.5 | 29.4 | 18.3 | -0.3 | 38.9 | 4.8 | -20.6 |
| mmlu_high_school_mathematics | `acc,none` | 27.8 | 28.5 | 31.1 | 32.6 | 28.1 | 28.1 | 25.2 | 0 | 32.6 | 0.0 | -2.6 |
| mmlu_high_school_physics | `acc,none` | 31.1 | 27.8 | 24.5 | 27.2 | 31.1 | 25.8 | 21.9 | -0.3 | 31.1 | 4.0 | -9.3 |
| mmlu_international_law | `acc,none` | 44.6 | 53.7 | 65.3 | 68.6 | 67.8 | 57.9 | 52.1 | 0 | 68.6 | 0.0 | 7.4 |
| mmlu_logical_fallacies | `acc,none` | 39.9 | 44.2 | 46.0 | 50.9 | 46.0 | 43.6 | 37.4 | 0 | 50.9 | 0.0 | -2.5 |
| mmlu_machine_learning | `acc,none` | 20.5 | 21.4 | 25.0 | 29.5 | 35.7 | 34.8 | 28.6 | 0.1 | 35.7 | 6.2 | 8.0 |
| mmlu_miscellaneous | `acc,none` | 52.2 | 58.2 | 63.2 | 65.6 | 64.2 | 59.1 | 48.3 | 0 | 65.6 | 0.0 | -4.0 |
| mmlu_moral_disputes | `acc,none` | 34.7 | 41.0 | 43.1 | 46.8 | 49.4 | 50.0 | 40.5 | 0.2 | 50.0 | 3.2 | 5.8 |
| mmlu_philosophy | `acc,none` | 45.7 | 50.2 | 54.0 | 51.4 | 49.2 | 45.7 | 42.4 | -0.1 | 54.0 | 2.6 | -3.2 |
| mmlu_professional_law | `acc,none` | 28.7 | 31.3 | 34.9 | 37.0 | 37.0 | 31.7 | 28.0 | 0 | 37.0 | 0.0 | -0.7 |
| mmlu_professional_medicine | `acc,none` | 49.3 | 52.9 | 57.0 | 57.4 | 52.9 | 41.2 | 28.7 | 0 | 57.4 | 0.0 | -20.6 |

### GPQA

| task | metric | -0.3 | -0.2 | -0.1 | 0 | 0.1 | 0.2 | 0.3 | best alpha | best | best-0 | 0.3-(-0.3) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpqa_diamond_zeroshot | `acc_norm,none` | 22.2 | 24.7 | 25.8 | 29.8 | 26.3 | 25.8 | 28.3 | 0 | 29.8 | 0.0 | 6.1 |
| gpqa_extended_zeroshot | `acc_norm,none` | 21.8 | 22.7 | 20.9 | 23.1 | 26.7 | 26.6 | 26.7 | 0.1 | 26.7 | 3.7 | 4.9 |
| gpqa_main_zeroshot | `acc_norm,none` | 31.0 | 29.5 | 26.8 | 25.9 | 26.3 | 25.0 | 25.9 | -0.3 | 31.0 | 5.1 | -5.1 |

### AGIEVAL

| task | metric | -0.3 | -0.2 | -0.1 | 0 | 0.1 | 0.2 | 0.3 | best alpha | best | best-0 | 0.3-(-0.3) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| agieval_aqua_rat | `acc_norm,none` | 24.0 | 26.4 | 25.6 | 24.0 | 21.7 | 20.5 | 19.3 | -0.2 | 26.4 | 2.4 | -4.7 |
| agieval_logiqa_en | `acc_norm,none` | 27.2 | 26.9 | 30.0 | 29.3 | 29.8 | 32.7 | 30.4 | 0.2 | 32.7 | 3.4 | 3.2 |
| agieval_lsat_ar | `acc_norm,none` | 20.0 | 18.7 | 20.4 | 15.7 | 13.9 | 17.8 | 22.6 | 0.3 | 22.6 | 7.0 | 2.6 |
| agieval_lsat_lr | `acc_norm,none` | 23.5 | 23.3 | 25.1 | 26.1 | 23.3 | 21.4 | 20.8 | 0 | 26.1 | 0.0 | -2.7 |
| agieval_lsat_rc | `acc_norm,none` | 22.3 | 21.2 | 20.4 | 19.7 | 19.7 | 20.4 | 19.7 | -0.3 | 22.3 | 2.6 | -2.6 |
| agieval_sat_en | `acc_norm,none` | 22.3 | 27.2 | 25.7 | 28.6 | 29.6 | 23.8 | 22.8 | 0.1 | 29.6 | 1.0 | 0.5 |
| agieval_sat_en_without_passage | `acc_norm,none` | 24.8 | 23.3 | 26.2 | 26.2 | 25.2 | 22.8 | 19.9 | -0.1 | 26.2 | 0.0 | -4.9 |
| agieval_sat_math | `acc_norm,none` | 26.4 | 25.5 | 25.5 | 27.7 | 29.1 | 24.1 | 24.5 | 0.1 | 29.1 | 1.4 | -1.8 |

## llama3b

### MMLU

| task | metric | -0.3 | -0.2 | -0.1 | 0 | 0.1 | 0.2 | 0.3 | best alpha | best | best-0 | 0.3-(-0.3) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu_abstract_algebra | `acc,none` | 23.0 | 29.0 | 31.0 | 35.0 | 34.0 | 31.0 | 31.0 | 0 | 35.0 | 0.0 | 8.0 |
| mmlu_clinical_knowledge | `acc,none` | 37.7 | 60.8 | 68.3 | 67.2 | 64.2 | 55.1 | 42.6 | -0.1 | 68.3 | 1.1 | 4.9 |
| mmlu_college_biology | `acc,none` | 32.6 | 58.3 | 69.4 | 71.5 | 69.4 | 60.4 | 38.9 | 0 | 71.5 | 0.0 | 6.2 |
| mmlu_college_chemistry | `acc,none` | 30.0 | 38.0 | 44.0 | 41.0 | 35.0 | 33.0 | 31.0 | -0.1 | 44.0 | 3.0 | 1.0 |
| mmlu_college_computer_science | `acc,none` | 24.0 | 23.0 | 40.0 | 53.0 | 45.0 | 39.0 | 24.0 | 0 | 53.0 | 0.0 | 0.0 |
| mmlu_college_mathematics | `acc,none` | 27.0 | 24.0 | 29.0 | 30.0 | 32.0 | 30.0 | 26.0 | 0.1 | 32.0 | 2.0 | -1.0 |
| mmlu_college_physics | `acc,none` | 27.5 | 28.4 | 32.4 | 38.2 | 34.3 | 34.3 | 31.4 | 0 | 38.2 | 0.0 | 3.9 |
| mmlu_computer_security | `acc,none` | 34.0 | 57.0 | 69.0 | 67.0 | 62.0 | 57.0 | 46.0 | -0.1 | 69.0 | 2.0 | 12.0 |
| mmlu_elementary_mathematics | `acc,none` | 27.8 | 33.9 | 40.7 | 43.7 | 41.3 | 38.4 | 30.7 | 0 | 43.7 | 0.0 | 2.9 |
| mmlu_formal_logic | `acc,none` | 30.2 | 36.5 | 42.1 | 42.1 | 31.0 | 27.0 | 23.8 | -0.1 | 42.1 | 0.0 | -6.3 |
| mmlu_high_school_mathematics | `acc,none` | 26.7 | 27.0 | 31.9 | 37.4 | 33.7 | 31.9 | 28.1 | 0 | 37.4 | 0.0 | 1.5 |
| mmlu_high_school_physics | `acc,none` | 26.5 | 28.5 | 36.4 | 37.7 | 33.8 | 37.7 | 31.1 | 0 | 37.7 | 0.0 | 4.6 |
| mmlu_international_law | `acc,none` | 37.2 | 66.1 | 69.4 | 72.7 | 71.9 | 71.9 | 57.0 | 0 | 72.7 | 0.0 | 19.8 |
| mmlu_logical_fallacies | `acc,none` | 33.7 | 62.0 | 72.4 | 74.8 | 69.3 | 60.7 | 49.1 | 0 | 74.8 | 0.0 | 15.3 |
| mmlu_machine_learning | `acc,none` | 34.8 | 45.5 | 46.4 | 45.5 | 35.7 | 35.7 | 32.1 | -0.1 | 46.4 | 0.9 | -2.7 |
| mmlu_miscellaneous | `acc,none` | 49.0 | 71.4 | 77.4 | 78.3 | 76.2 | 69.5 | 55.2 | 0 | 78.3 | 0.0 | 6.1 |
| mmlu_moral_disputes | `acc,none` | 32.9 | 51.4 | 64.5 | 68.2 | 61.8 | 57.5 | 38.4 | 0 | 68.2 | 0.0 | 5.5 |
| mmlu_philosophy | `acc,none` | 35.4 | 56.6 | 65.3 | 67.2 | 65.6 | 59.2 | 46.6 | 0 | 67.2 | 0.0 | 11.3 |
| mmlu_professional_law | `acc,none` | 26.0 | 36.1 | 42.6 | 47.1 | 43.5 | 39.5 | 32.1 | 0 | 47.1 | 0.0 | 6.1 |
| mmlu_professional_medicine | `acc,none` | 34.9 | 55.5 | 76.1 | 76.5 | 73.2 | 64.0 | 41.9 | 0 | 76.5 | 0.0 | 7.0 |

### GPQA

| task | metric | -0.3 | -0.2 | -0.1 | 0 | 0.1 | 0.2 | 0.3 | best alpha | best | best-0 | 0.3-(-0.3) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpqa_diamond_zeroshot | `acc_norm,none` | 23.7 | 27.3 | 32.3 | 30.8 | 31.8 | 31.3 | 29.8 | -0.1 | 32.3 | 1.5 | 6.1 |
| gpqa_extended_zeroshot | `acc_norm,none` | 23.4 | 23.8 | 26.6 | 30.0 | 31.5 | 31.9 | 27.3 | 0.2 | 31.9 | 1.8 | 3.8 |
| gpqa_main_zeroshot | `acc_norm,none` | 26.8 | 29.9 | 32.1 | 31.0 | 31.7 | 28.6 | 24.6 | -0.1 | 32.1 | 1.1 | -2.2 |

### AGIEVAL

| task | metric | -0.3 | -0.2 | -0.1 | 0 | 0.1 | 0.2 | 0.3 | best alpha | best | best-0 | 0.3-(-0.3) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| agieval_aqua_rat | `acc_norm,none` | 20.5 | 22.4 | 22.0 | 20.9 | 18.1 | 16.9 | 14.6 | -0.2 | 22.4 | 1.6 | -5.9 |
| agieval_logiqa_en | `acc_norm,none` | 26.7 | 28.0 | 33.2 | 33.9 | 35.9 | 33.5 | 31.0 | 0.1 | 35.9 | 2.0 | 4.3 |
| agieval_lsat_ar | `acc_norm,none` | 19.6 | 16.5 | 20.0 | 19.1 | 19.1 | 20.9 | 22.6 | 0.3 | 22.6 | 3.5 | 3.0 |
| agieval_lsat_lr | `acc_norm,none` | 24.1 | 23.3 | 27.5 | 28.2 | 28.0 | 27.6 | 21.8 | 0 | 28.2 | 0.0 | -2.4 |
| agieval_lsat_rc | `acc_norm,none` | 18.2 | 23.0 | 33.1 | 36.1 | 32.3 | 32.0 | 17.8 | 0 | 36.1 | 0.0 | -0.4 |
| agieval_sat_en | `acc_norm,none` | 21.8 | 30.6 | 42.2 | 49.0 | 43.2 | 41.3 | 20.4 | 0 | 49.0 | 0.0 | -1.5 |
| agieval_sat_en_without_passage | `acc_norm,none` | 25.7 | 32.0 | 30.6 | 29.1 | 26.2 | 25.2 | 18.0 | -0.2 | 32.0 | 2.9 | -7.8 |
| agieval_sat_math | `acc_norm,none` | 20.5 | 24.1 | 32.3 | 32.3 | 34.5 | 31.8 | 27.3 | 0.1 | 34.5 | 2.3 | 6.8 |
