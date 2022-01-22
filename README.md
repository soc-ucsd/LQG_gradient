# Landscape of LQG problems

This repository contains the MATLAB scripts for reproducing the experiments in our paper

1) Yang Zheng*, Yujie Tang*, and Na Li (2020). Analysis of the Optimization Landscape of Linear Quadratic Gaussian (LQG) Control. under preparation. A preliminary version is here: [LINK](https://zhengy09.github.io/papers/LQG_landscape.pdf)  (*Equal contribution)
 

## Instructions
The gradient descent algorithms are implemented in 
* LQG_gd_cano.m (partial gradient over the controllable canonical form) 
* LQG_gd_full.m (full gradient)


Run Example_Doyle.m to see some performance; more examples are included in the *Examples* folder.

# Landscape of dLQR problems (LQR using dynamical output feedback)

The "dLQR" folder contains the Python scripts for reproducing the experiments in our paper

2) Jingliang Duan, Wenhan Cao, Yang Zheng, Lin Zhao (2022). On the Optimization Landscape of Dynamical Output Linear Quadratic Control. 

Run dLQR.py to see the learning curves of three different dynamics; all six examples shown in our paper are included.
