# Optimal Decision Tree

<p align="center"><img width="100%" src="fig/dt.jpg" /></p>
 
### Introduction

This project aims to better understand the computational and prediction performance of MIP-based classification tree learning approaches proposed by recent research papers. In particular, the OCT, binOCT, and flowOCT formulations are implemented and evaluated on publicly available classification datasets. Moreover, we propose a stable classification tree formulation incorporating the training and validation split decision into the model fitting processes. Computational results suggest flowOCT has the most consistent performance across datasets and tree complexity. OCT is competitive on small datasets and shallow trees, while binOCT yields the best performance on large datasets and deep trees. The proposed stable classification tree achieves stronger out-of-sample performance than flowOCT under random training and validation set splits.

### Dependencies

* [Python 3.7](https://www.python.org/)
* [Gurobi 9.1](https://www.gurobi.com/)
