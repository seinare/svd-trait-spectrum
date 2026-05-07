# References

`svd_trait_spectrum_refs.bib` contains the BibTeX entries used by the introduction and related-work draft for the SVD trait spectrum paper.

The citation keys are:

- `svd_transformer_interpretable`
- `beyond_components_svd_circuits`
- `spectral_filters_dark_signals`
- `small_singular_values_matter`
- `laser`
- `pissa`
- `milora`
- `kasa`
- `geva_kv_memory`
- `rome`
- `memit`
- `knowledge_neurons`

Validation command used:

```bash
pdflatex -interaction=nonstopmode test.tex
bibtex test
pdflatex -interaction=nonstopmode test.tex
pdflatex -interaction=nonstopmode test.tex
```

The generated `.blg` reports `warning$ -- 0` and `missing$ -- 0`.
