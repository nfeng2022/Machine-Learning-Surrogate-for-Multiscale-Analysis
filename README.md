# Machine Learning Surrogates for Finite Strain Multiscale Analysis
A deep neural network (DNN) model is trained for predicting the material mapping from macroscopic strain tensor to stress and material tangent tensors.

## Dependency
- Matlab

## Benchmark
The DNN can be used as a materail subrountine for FE<sup>2</sup> analysis of multiscale structures composed of periodic hyperelastic microstructures. The below example show the defromed shapes of ar arch example from DNN and homogenization:

![](./arch_problem.png)

![](./arch_results.png)

**To test DNN model, run**
```
matlab dnn_material_test.py
```

## Citation
To cite, please use the following information:
```
@article{FENG2022106742,
title = {Finite strain FE2 analysis with data-driven homogenization using deep neural networks},
journal = {Computers & Structures},
volume = {263},
pages = {106742},
year = {2022},
issn = {0045-7949},
doi = {https://doi.org/10.1016/j.compstruc.2022.106742},
url = {https://www.sciencedirect.com/science/article/pii/S0045794922000025},
author = {Nan Feng and Guodong Zhang and Kapil Khandelwal}
}
```
