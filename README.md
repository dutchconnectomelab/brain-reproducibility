# Reproducibility of neuroimaging studies of brain disorders 

This repository contains the code for the evaluation of reproducibility of neuroimaging findings in clinical neuroimaging case-control studies:
- Power calculation by means of Monte-Carlo simulation based on effect sizes reported by [ENIGMA](https://enigma-toolbox.readthedocs.io/en/latest/) meta-review (>4,000 patients);
- Analysis of empirical brain imaging data (>7,000 subjects) to assess the statistical power and reproducibility rates.

These analyses were performed as part of the work by Libedinsky, I. et al., "Reproducibility of neuroimaging studies of brain disorders with hundreds -not thousands- of participants".

## :hammer: Installation

The analysis was performed using [Python](https://www.python.org/) 3.9.

The scripts and all dependencies can be installed as follows:
```
git clone https://github.com/dutchconnectomelab/brain-reproducibility
cd brain-reproducibility
python3 -m pip install -e .
```

## :game_die: Simulations of statistical power

Figure 1a was generated using:

```
python3 brain_reproducibility/power.py --iterations 1000 --progress
```

The script relies on summary statistics from ENIGMA.
These were downloaded separately using `brain_reproducibility/load_engima_effect_sizes.py`.

## 🧠 Empirical analysis of brain imaging data

Figure 1b was generated using:

```
python3 brain_reproduciblity/replication.py \
  --iterations 10000 \
  --datapath /path/to/data/ \  # remove this to use simulated data
  --progress
```

The empirical analysis of replication rates is based on MRI data of measurements of cortical thickness (see Datasets below). 
Preprocessed data is available upon request to researchers.

If no empirical data is available, the script can alternatively use simulated data that is generated based on the region-wise and dataset-wise means and standard deviations matching the empirical data.
This is done automatically when removing the `--datapath` argument in the command above.

## :heart: Open datasets & Acknowledgements

We thank the following instititions for the data and resources they have made available:
- Enhancing Neuro Imaging Genetics Through Meta Analysis (ENIGMA) Consortium; [Enigma toolbox](https://enigma-toolbox.readthedocs.io/en/latest/pages/01.install/index.html) also allows to further explore other effect-sizes in other disorders
- Northwestern University Schizophrenia Data and Software Tool (NUSDAST)
- MIND Clinical Imaging Consortium (MCIC)
- Centre for Biomedical Research Excellence (COBRE)
- BrainGluSchi
- Neuromorphometry by Computer Algorithm Chicago (NMorphCH)
- Consortium for Neuropsychiatric Phenomics (CNP)
- Phenomenology and Classification of Schizophrenia (Iowa Longitudinal Study)
- Bipolar & Schizophrenia Consortium for Parsing Intermediate Phenotypes (B-SNIP)
- Japanese Strategic Research Program for the Promotion of Brain Science (SRPBS)
- Alzheimer's Disease Neuroimaging Initiative (ADNI)
- Open Access Series of Imaging Studies (OASIS)
- National Alzheimer’s Coordinating Center (NACC)
- Italian Alzheimer's Disease Neuroimaging Initiative (I-ADNI)
- Alzheimer's Disease Repository Without Borders (ARWIBO)
- European Diffusion Tensor Imaging Study in Dementia (EDSD)
