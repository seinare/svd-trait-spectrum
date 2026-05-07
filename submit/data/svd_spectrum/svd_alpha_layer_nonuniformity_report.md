# Layer-wise Non-uniformity of Alpha SVD Perturbations

This report summarizes the spectral evidence that the same global alpha perturbation does not produce a uniform update strength across model depth. The core reason is that each layer starts from a different singular-value distribution. Since the current perturbation rescales singular values according to their distance from the layer-wise geometric mean, layers with more unequal spectra receive stronger effective changes than layers with flatter spectra.

## Perturbation Definition

For each MLP `up_proj` and `down_proj` matrix, the alpha perturbation is applied to singular values as:

```text
s_i' = G * (s_i / G)^(1 + alpha)
G = exp(mean_i log s_i)
```

This preserves the geometric mean of the singular values. Positive alpha makes large singular values larger relative to the geometric mean and small singular values smaller, increasing spectral inequality. Negative alpha has the opposite effect and smooths the spectrum.

Therefore alpha has a stable global direction, but not a stable layer-wise magnitude. The effective update depends on the original distribution of singular values in each layer.

## Figures

### 1. Gini Across Layers

![Gini by layer](figures/svd_gini_by_layer_2x5.png)

The Gini plot directly shows that spectral inequality differs by layer, model, and projection. Dense Llama models show clear depth-dependent changes, especially in `up_proj`; Qwen3-8B has unusually concentrated early layers; Qwen3-30B-A3B MoE additionally has expert-level variation, where the maximum expert can deviate substantially from the layer mean.

This means that a uniform alpha is not equivalent to a uniform layer-wise perturbation. Layers that already have high Gini are more sensitive to positive alpha and more strongly smoothed by negative alpha.

### 2. Relative Change of the Top Singular Value

![Top singular value relative change](figures/svd_top_sv_relative_by_layer_2x5.png)

The relative top-singular-value plot shows where alpha causes the largest multiplicative amplification or suppression. The effect is especially strong in layers whose top singular value is already far above the geometric mean.

At alpha `+0.2`, examples of large relative increases include:

| model | layer | proj | base Gini | top singular value relative multiplier |
| --- | ---: | --- | ---: | ---: |
| Qwen3 8B | 1 | down_proj | 0.3209 | 1.7984 |
| Qwen3 8B | 2 | down_proj | 0.3513 | 1.7278 |
| Llama 3.2 3B Instruct | 27 | up_proj | 0.2506 | 1.6784 |
| Llama 3.1 8B Instruct | 31 | up_proj | 0.2143 | 1.6535 |
| Qwen3 8B | 2 | up_proj | 0.3910 | 1.6443 |

These are not merely small smooth rescalings. In sensitive layers, the same alpha can produce a large top-mode amplification.

### 3. Absolute Top Singular Value Across Layers

![Top singular value absolute value](figures/svd_top_sv_absolute_value_by_layer_2x5.png)

The absolute-value plot adds scale. Some layers have large absolute top singular values even before perturbation. If such layers also have high spectral inequality, positive alpha can concentrate even more energy into already dominant directions.

This is the main over-update risk: a global alpha can be moderate on average while still being aggressive in a subset of layers. The risk is stronger when a layer combines:

- high base Gini;
- large `top singular value / geometric mean`;
- large absolute top singular value;
- projection- or expert-specific concentration.

### 4. Llama 3.1 8B Head/Tail Singular Values

![Llama 3.1 8B selected singular values](figures/llama31_8b_up_down_head_tail50_barplane3d_2panel.png)

The 3D bar plot shows selected Llama 3.1 8B layers: bottom, middle, and top. The head and tail of the singular-value spectrum are shown separately. The layer axis makes the depth difference visible: the top singular values and the lower tail are not shaped identically across layers.

This supports the same conclusion from the aggregate plots. The perturbation formula is global, but the spectrum it acts on is layer-specific. Since alpha amplifies or smooths singular values according to their relative position against the geometric mean, different layers receive different effective deformation.

## Main Interpretation

The data support three points.

First, the alpha transform has a consistent sign effect. Positive alpha increases Gini and top singular values; negative alpha decreases them. This is expected from the formula.

Second, the magnitude of this effect is highly non-uniform across layers. A layer with a more unequal singular-value distribution is more affected by the same alpha than a flatter layer. The strongest affected layers are not always the same across model families or projections.

Third, MoE models add another axis of non-uniformity. The expert mean describes the typical routed expert, while the expert maximum reveals that some individual experts can be much more spectrally concentrated than the layer average. A single global alpha can therefore over-perturb a subset of experts even if the layer mean looks acceptable.

## Implication for Experiments

Using one global alpha is useful as a controlled intervention, but it should be interpreted as a spectrum-dependent perturbation rather than a uniform update. The same alpha can be mild in one layer and strong in another. This can explain why alpha sweeps sometimes produce selective task improvements or degradations rather than smooth global quality changes.

For future runs, this suggests several safer variants:

- layer-normalized alpha, where the target Gini shift is bounded per layer;
- projection-specific alpha for `up_proj` and `down_proj`;
- MoE expert clipping, where expert-level top singular value amplification is capped;
- reporting alpha together with realized spectral statistics, not alpha alone.

The current figures therefore justify treating alpha as an experimental control variable, but not as a uniform physical update size across the network.

