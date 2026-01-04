# CMAPSS Dataset

This folder contains the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset
used for Remaining Useful Life (RUL) prediction.

## Dataset Files

For each subset (FD001–FD004), the following files are provided:

- `train_FD00x.txt`  
  Training data containing full run-to-failure trajectories.

- `test_FD00x.txt`  
  Test data containing truncated trajectories.

- `RUL_FD00x.txt`  
  True Remaining Useful Life values for the test data.

## Subsets Description

- **FD001**: Single operating condition, single fault mode  
- **FD002**: Multiple operating conditions, single fault mode  
- **FD003**: Single operating condition, multiple fault modes  
- **FD004**: Multiple operating conditions, multiple fault modes  

## Data Format

Each row in the training and test files represents one time cycle of an engine and contains:
- Engine ID
- Cycle number
- Operating conditions
- Sensor measurements

The RUL files contain one value per engine in the test set.

## Source

NASA Ames Prognostics Data Repository  
CMAPSS Turbofan Engine Degradation Simulation Dataset
