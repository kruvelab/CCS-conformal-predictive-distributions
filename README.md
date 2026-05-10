# Candidate Structure Prioritization in Non-Target Screening with Predicted Collision Cross-Sections and Uncertainty Quantification


**Authors:** Lucas Ferrando Plo and Anneli Kruve

This repository contains the workflow used to obtain **probabilities** and **Continuous Ranked Probability Scores (CRPSs)** for candidate structures in **non-target screening (NTS)** using predicted collision cross-sections.

## Overview

The pipeline combines:

- [myopic-mces](https://github.com/AlBi-HHU/myopic-mces) for myopic MCES distance calculations
- [GraphCCS](https://github.com/tingxiecsu/GraphCCS) for collision cross-section prediction
- the `OODDS_CPS_GraphCCS` workflow for conformal predictive scoring (and out-of-domain detection)

## Setup

Before running the workflow, install the environments required by the two external tools:

1. Install the environment from [myopic-mces](https://github.com/AlBi-HHU/myopic-mces)  
   *(used for myopic MCES distance calculations)*

2. Install the environment from [GraphCCS](https://github.com/tingxiecsu/GraphCCS)  
   *(used for CCS prediction)*

3. Download the `OODDS_CPS_GraphCCS` folder.

## Input File

Place a file named `input.tsv` inside the `OODDS_CPS_GraphCCS` folder.

The file must contain the following columns:

- `SMILES` and/or `InChI`  
  Molecular identifier of the candidate structure
- `Ion Type`  
  Detected ion type of the candidate structure
- `CCS`  
  Measured CCS of the NTS feature corresponding to the candidate structure

### Supported ion types

Currently, the workflow supports:

- `[M+H]+`
- `[M+Na]+`
- `[M-H]-`

## Running the Workflow

Open and run the Jupyter notebook `OODDS_CPS_GraphCCS` located in the same folder.

## Output File

After execution, a file named `outputMETLINCCS.tsv` or `outputEnvCCS.tsv`, depending on the calibrated chemical space selected, will be generated in the same folder.

This file contains all columns from `input.tsv`, plus the following additional outputs:

- `P^OOD`  
  Out-of-domain degree for the candidate structure 
- `CCShat`  
  Predicted CCS for the candidate structure
- `P^CCS`  
  Probability that the measured CCS of the corresponding NTS feature originates from the predictive distribution of the candidate structure
- `CRPS`  
  Continuous Ranked Probability Score, quantifying how close and how tight the predictive distribution of a candidate structure is around the measured CCS of the corresponding NTS feature
