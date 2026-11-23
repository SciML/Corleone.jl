# Boehm STAT5 data provenance

The three TSV files are unmodified copies of the PEtab 2 encoding of
`Boehm_JProteomeRes2014` in the
[PEtab benchmark collection](https://github.com/PEtab-dev/petab-benchmark-problems/tree/49900951a5594a757ad5e651684c7f0a59272881/problems/Boehm_JProteomeRes2014/v2),
revision `49900951a5594a757ad5e651684c7f0a59272881`.
The collection's BSD 3-Clause license is retained in `LICENSE`. The collection
notes that different terms may apply to individual models and datasets; the
original experimental source is:

Boehm ME, Adlung L, Schilling M, Roth S, Klingmüller U, Lehmann WD (2014).
Identification of Isoform-Specific Dynamics in Phosphorylation-Dependent STAT5
Dimerization by Quantitative Mass Spectrometry and Mathematical Modeling.
*Journal of Proteome Research* 13(12):5685–5694.
[doi:10.1021/pr5006923](https://doi.org/10.1021/pr5006923).

The accompanying Julia model transcribes the reactions, compartments, initial
assignments, and time-dependent Epo input from the
[SBML model at the same revision](https://github.com/PEtab-dev/petab-benchmark-problems/blob/49900951a5594a757ad5e651684c7f0a59272881/problems/Boehm_JProteomeRes2014/v2/model.xml).
Observable callbacks transcribe `observables.tsv`. This example fixes the three
noise standard deviations at their published nominal values and fits the six
kinetic log10 parameters within the published bounds. It is a conditional weighted
least-squares fit, rather than the full nine-parameter PEtab likelihood problem.
