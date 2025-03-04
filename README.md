# Mixing Indices

This repository contains the source code associated with the open-access article **"Mixing indices in up-scaled simulations"**, published in *Powder Technology* ([doi.org/10.1016/j.powtec.2025.120775](https://doi.org/10.1016/j.powtec.2025.120775)).

The main focus of this repository is the [mixingIndices.py](mixingIndices.py) script, which implements 16 different mixing indices. A detailed description of these indices can be found in the aforementioned open-access article. This script simplifies the evaluation and comparison of mixing indices on any particulate dataset that can be read into NumPy arrays, as all the indices are already implemented.

## Using the Mixing Indices Script

The script can be used to evaluate mixing on any particulate dataset stored as a NumPy array. The file [mixingIndices.py](mixingIndices.py) is a self-contained script designed for mixing evaluation. Key features include:

- Each mixing index is implemented as a separate class, inheriting methods from the `MixingIndex` base class.
- To use an index, import its corresponding class and initialise it with the constructor. Some indices have default constructor parameters, which can be modified if needed.
- **Grid-based methods**: If grid data is unavailable, use the `ParticleGridMapping` class to map particle positions onto a grid, then pass this grid data to the index.
- **Contact-based methods**: If a contact list is not available, use the `ParticleNeighborList` class to compute the contact list from particle positions, then pass it to the index.
- **Distance-based methods**: These indices can use particle positions directly.
- Call the `iterate` method for each timestep.
- At the end, call the `get_M` method to retrieve the mixing index values for each iteration as a NumPy array.

For further details, check [wrappers.py](comparisonMI/wrappers.py), which demonstrates how these indices were evaluated in the article.

## Reproducing Figures and Data from the Article

To recreate the data and figures presented in our open-access article in *Powder Technology* ([doi.org/10.1016/j.powtec.2025.120775](https://doi.org/10.1016/j.powtec.2025.120775)), follow these general steps:

1. Ensure the folder containing the Python scripts is included in the Python PATH.
2. Run the relevant script to generate and save the lattice data, process the data, and produce the figures.
3. When multiple scripts rely on the same dataset, execute all necessary scripts.

To generate the figures in the sections run the following scripts:

- **Section 4.1.1, Figures 3-4**: [compareNumberOfParticles.py](comparisonMI/compareNumberOfParticles.py)
- **Section 4.1.2, Figures 5-6**: [compareSolidFraction.py](comparisonMI/compareSolidFraction.py)
- **Section 4.1.3, Figures 7-8**: [compareInhomogeneity.py](comparisonMI/compareInhomogeneity.py)
- **Section 4.2.1, Figure 9**: [numberOfSamples.py](comparisonMI/numberOfSamples.py)
- **Section 4.2.1, Figures 10-13 (and Supplementary Material)**: [compareNumberOfCells.py](comparisonMI/compareNumberOfCells.py)
- **Section 4.2.2, Figure 16**: [compareReversibility.py](comparisonMI/compareReversibility.py)

- **Tables 1 and 2**: Data saved as text files by [compareNumberOfParticles.py](comparisonMI/compareNumberOfParticles.py) and [compareSolidFraction.py](comparisonMI/compareSolidFraction.py), respectively.
- _Figures 1-2 and 14-15: Illustrations created using Inkscape; therefore, they are not included in this repository._

## Required Python Packages

Ensure the following Python libraries are installed:

- `numpy`
- `scikit-learn` _(only used for some indices)_
- `scipy` _(only used for some indices)_
- `matplotlib` _(only required for generating figures from the article)_

## Citation

If you use the Python code, please cite our article:

Balázs Füvesi, Christoph Goniva, Stefan Luding, Vanessa Magnanimo, **Mixing indices in up-scaled simulations**, *Powder Technology*, 2025, [doi.org/10.1016/j.powtec.2025.120775](https://doi.org/10.1016/j.powtec.2025.120775).

