# Alpha Response, Capability Demand, and TRAIT Association Analysis

## Data and preprocessing

This figure summarizes the relationship among SVD perturbation strength, latent capability responses, TRAIT personality scores, and task-level capability demand. The analysis uses the four dense models with complete alpha grids: Llama 3.2 1B Instruct, Llama 3.2 3B Instruct, Llama 3.1 8B Instruct, and Qwen3 8B.

For capability responses, the source is `capability_alpha_mle_errorbar_points.csv`. Each task subitem has DeepSeek-judged capability demand over three retained dimensions: Factual Knowledge, Language Understanding, and Deductive Reasoning. The model-level alpha score changes are projected into these dimensions using the previously fitted maximum-likelihood capability-response model.

For TRAIT responses, the source is the alpha-9 TRAIT tables. Each score is centered at alpha = 0 within the same model, so plotted values are perturbation-induced changes rather than raw personality scores.

## Panel a: local alpha-response slopes

Panel a estimates local linear response slopes over four alpha intervals: `[-0.2,-0.1]`, `[-0.1,0]`, `[0,0.1]`, and `[0.1,0.2]`. For each interval, the slope is fitted as `d(delta) / d(alpha)` using the three alpha points in that interval. Bars show the mean across the four dense models, and error bars show SEM across models.

This panel keeps the sign as the true local derivative with respect to alpha. Thus a negative slope in a negative-alpha interval means that moving alpha upward within that interval decreases the centered score; it is not the same convention as using `|alpha|` as the independent variable.

## Panel b: capability-TRAIT Pearson correlation

Panel b uses simple Pearson correlation, not standardized multivariate regression coefficients. Each cell correlates a capability response dimension with a TRAIT response across nonzero-alpha model points. Dark borders mark `|r| >= 0.30`.

The strongest positive association is between Factual Knowledge and Agreeableness (`r ≈ 0.53`). Deductive Reasoning shows a negative association with Machiavellianism (`r ≈ -0.35`) and a positive association with Narcissism (`r ≈ 0.33`). These should be interpreted as association patterns over perturbation-response samples, not causal effects.

## Panel c: response association network

Panel c visualizes the same quantities as a path-style network. The left layer contains positive and negative alpha perturbation directions. Alpha-response edges are dashed and colored by the sign of the aggregated response slope: red for positive response, blue for negative response. Weak alpha-to-capability and alpha-to-TRAIT responses are filtered to reduce clutter.

The middle layer contains the three capability dimensions and eight TRAIT dimensions. Capability labels use semantic colors: Factual Knowledge is green, Language Understanding is blue, and Deductive Reasoning is purple. The dark-triad TRAIT dimensions are shown in dark red; other TRAIT dimensions are shown in light blue.

Solid red/blue curves between capabilities and TRAIT dimensions show Pearson r from panel b. Curves with larger vertical distance are drawn wider and darker, while nearby associations are shorter, shallower, and lighter. The right layer shows selected task subitems. Gray solid lines show the dominant DeepSeek-judged capability demand for each selected subtask; pale dashed alpha-to-subtask lines provide weak contextual response signals and are intentionally kept in the background.

## Summary of observed pattern

Across the complete dense-model grid, capability responses are asymmetric and locally nonlinear over alpha. The largest local slopes occur in the negative-alpha intervals for Factual Knowledge and Language Understanding, while TRAIT responses are generally smaller in magnitude but show structured associations with the capability dimensions.

The Pearson view suggests that Factual Knowledge is the main positive axis linked to Agreeableness and Openness, while Deductive Reasoning separates Machiavellianism and Narcissism in opposite directions. The network view supports the same interpretation while also showing which task families are primarily driven by DeepSeek-judged capability demand.

## Generated files

- Combined figure PNG: `docs/results/trait_capability_main_figure/trait_capability_abc_combined.png`
- Combined figure PDF: `docs/results/trait_capability_main_figure/trait_capability_abc_combined.pdf`
- Panel a/b PNG: `docs/results/trait_capability_main_figure/trait_capability_ab_horizontal.png`
- Panel c PNG: `docs/results/trait_capability_main_figure/trait_capability_panel_c_standalone.png`