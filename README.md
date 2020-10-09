# Near-Orthogonal Monte Carlo (NOMC)

This is a python implementation of the opt-NOMC algorithm, as well as sliced Wasserstein distances and kernel approximation applications accompany the publication "Demystifying Orthogonal Monte Carlo and Beyond" by Han Lin, Haoxian Chen, Tianyi Zhang, Clement Laroche, and Krzysztof Choromanski.


# Requirements

NOMC relies on Python 3.7 as well as some commonly-used libraries as mentioned in requirements.txt. The specific python and libraries versions should not matter too much in the implementation of our algorithm. 

# Running the code
 `opt_NOMC.py` implements opt-NOMC algorithm, with parameters set in `config_NOMC.json`.
 `QMC.py` implements QMC algorithm, with parameters set in `config_QMC.json`.
 `SWD.py` implements sliced Wasserstein distance experiments with multivariate gaussian as an example, and the parameters can be set in `config_SWD.json`.
 
 Besides, we also includes some already calculated and optimized NOMC and QMC samples under the folder `data`, which can be used directly to replicate our experiments using NOMC and QMC algirhms in the paper.
