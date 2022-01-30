# Landscape of LQG problems

This repository contains the MATLAB scripts for reproducing the experiments in our paper

1) Yujie Tang*, Yang Zheng* and Na Li (2020). [Analysis of the Optimization Landscape of Linear Quadratic Gaussian (LQG) Control](https://arxiv.org/abs/2102.04393). Mathematical Programming, under review, 2021 (A [short version](http://proceedings.mlr.press/v144/tang21a/tang21a.pdf) was accepted at L4DC, 2021) (*Equal contribution)
 

## Instructions
The gradient descent algorithms are implemented in 
* LQG_gd_cano.m (partial gradient over the controllable canonical form) 
* LQG_gd_full.m (full gradient)


Run Example_Doyle.m to see some performance; more examples are included in the *Examples* folder.

## Landscape of dLQR problems (LQR using dynamical output feedback)

The "dLQR" folder contains the Python scripts for reproducing the experiments in our paper

2) Jingliang Duan, Wenhan Cao, Yang Zheng, Lin Zhao (2022). [On the Optimization Landscape of Dynamical Output Linear Quadratic Control](https://arxiv.org/abs/2201.09598). preprint

Run example_1.py to see the learning curves of example 1 in our paper; similar for other five examples.
