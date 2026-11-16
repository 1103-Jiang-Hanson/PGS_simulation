# PGS Simulation

A small Python simulation that generates synthetic gene counts and binary condition labels, then compares a correlation-based score with a least-squares linear regression model.

## Requirements

- Python 3
- NumPy

Install NumPy with `python3 -m pip install -r requirements.txt`.

## Run

Run `python3 PGS_simulation.py` and enter the number of genes and simulated people when prompted.

## Important note

This is an educational toy simulation, not a validated polygenic score or medical tool. The current comparison computes model fits and accuracy on the same simulated biobank, so its accuracy is in-sample and should not be interpreted as predictive performance on new people. The simulation also uses randomly generated effects and may generate highly imbalanced labels.
