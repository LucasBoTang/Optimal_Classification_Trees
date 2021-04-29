# Optimal Decision Tree

<p align="center"><img width="100%" src="fig/dt.jpg" /></p>
 
### Introduction

This project aims to better understand the computational and prediction performance of MIP-based classification tree learning approaches proposed by recent research papers. In particular, the OCT, binOCT, and flowOCT formulations are implemented and evaluated on publicly available classification datasets. Moreover, we propose a stable classification tree formulation incorporating the training and validation split decision into the model fitting processes. Computational results suggest flowOCT has the most consistent performance across datasets and tree complexity. OCT is competitive on small datasets and shallow trees, while binOCT yields the best performance on large datasets and deep trees. The proposed stable classification tree achieves stronger out-of-sample performance than flowOCT under random training and validation set splits.

### Dependencies

* [Python 3.7](https://www.python.org/)
* [Gurobi 9.1](https://www.gurobi.com/)

### MIP Models

- [Optimal Classification Trees (OCT)](https://link.springer.com/article/10.1007/s10994-017-5633-9) - Bertsimas, D., & Dunn, J. (2017). Optimal classification trees. Machine Learning, 106(7), 1039-1082.
- [Optimal Classification Tree with Binary Encoding (binOCT)](https://ojs.aaai.org//index.php/AAAI/article/view/3978) - Verwer, S., & Zhang, Y. (2019). Learning optimal classification trees using a binary linear program formulation. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 1625-1632).
- [Max-Flow Optimal Classification Trees (flowOCT)](http://www.optimization-online.org/DB_HTML/2021/01/8220.html) - Aghaei, S., Gómez, A., & Vayanos, P. (2021). Strong Optimal Classification Trees. arXiv preprint arXiv:2103.15965.

### Stable Classification Tree

With the ideas come from Bertsimas and Paskov (2020), we also implemented a stable classification tree (SOCT) formulation incorporating the training and validation set split decision into the decision tree learning processes. The model can be regardded as training the decision tree on the "hardest" sub-set of the training dataset. The SOCT is solved using a robust optimization method and a cutting plane method.

- Bertsimas, D. and Paskov, I. (2020).  Stable regression:  On the power of optimization over randomizationin training regression problems.Journal of Machine Learning Research, 21:1–25

### Data

- [Balance Scale](https://archive.ics.uci.edu/ml/datasets/balance+scale)
- [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/breast+cancer)
- [Car Evaluation](https://archive.ics.uci.edu/ml/datasets/car+evaluation)
- [Hayes-Roth](https://archive.ics.uci.edu/ml/datasets/Hayes-Roth)
- [House Votes 84](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)
- [Soybean (small)](https://archive.ics.uci.edu/ml/datasets/soybean+(small))
- [SPECT Heart](https://archive.ics.uci.edu/ml/datasets/spect+heart)
- [Tic-Tac-Toe Endgame](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame)
- [MONK's Problems](https://archive.ics.uci.edu/ml/datasets/MONK's+Problems)

### Report

Details about the project report is [here](https://github.com/LucasBoTang/MIP_Decision_Tree/blob/main/Report.pdf).

**Attention: The experiment results in the report is old version.**

### Results

All the tests are conducted on Intel(R) Core(TM) CPU i7-7700HQ @ 2.80GHz and a memory of 16 GB. Algorithms are implemented in [Python 3.7](https://www.python.org/) with [Gurobi 9.1](https://www.gurobi.com/) as an optimization solver. The time limit is set to **600** seconds.

#### Out-of-Sample Prediction Performance for OCT, binOCT, flowOCT, and CART
| instance       | depth | OCT    | binOCT | flowOCT | CART   |
|----------------|-------|--------|--------|---------|--------|
| balance-scale  | 2     | 0.6879 | 0.6497 | 0.6709  | 0.6285 |
| balance-scale  | 3     | 0.6985 | 0.6603 | 0.6964  | 0.7197 |
| balance-scale  | 4     | 0.7070 | 0.6624 | 0.7197  | 0.7707 |
| balance-scale  | 5     | 0.6815 | 0.6157 | 0.7134  | 0.7707 |
| breast-cancer  | 2     | 0.6952 | 0.6857 | 0.6810  | 0.7095 |
| breast-cancer  | 3     | 0.7143 | 0.7143 | 0.6810  | 0.7286 |
| breast-cancer  | 4     | 0.7238 | 0.7000 | 0.7429  | 0.7190 |
| breast-cancer  | 5     | 0.7286 | 0.6238 | 0.6857  | 0.6952 |
| car-evaluation | 2     | 0.7654 | 0.7654 | 0.7654  | 0.7739 |
| car-evaluation | 3     | 0.7971 | 0.7863 | 0.7932  | 0.7724 |
| car-evaluation | 4     | 0.7361 | 0.8171 | 0.8071  | 0.8364 |
| car-evaluation | 5     | 0.6952 | 0.8233 | 0.7978  | 0.8341 |
| hayes-roth     | 2     | 0.6083 | 0.5500 | 0.6083  | 0.5250 |
| hayes-roth     | 3     | 0.6417 | 0.7000 | 0.7250  | 0.6083 |
| hayes-roth     | 4     | 0.7500 | 0.6417 | 0.7917  | 0.6750 |
| hayes-roth     | 5     | 0.7750 | 0.5417 | 0.7750  | 0.7833 |
| house-votes-84 | 2     | 0.9713 | 0.9655 | 0.9713  | 0.9713 |
| house-votes-84 | 3     | 0.9713 | 0.9713 | 0.9713  | 0.9713 |
| house-votes-84 | 4     | 0.9713 | 0.9253 | 0.9713  | 0.9598 |
| house-votes-84 | 5     | 0.9713 | 0.9598 | 0.9713  | 0.9713 |
| monks-1        | 2     | 0.7506 | 0.7050 | 0.7506  | 0.7314 |
| monks-1        | 3     | 0.8729 | 0.8393 | 0.8585  | 0.7890 |
| monks-1        | 4     | 0.9496 | 0.9041 | 1.0000  | 0.7770 |
| monks-1        | 5     | 0.8129 | 1.0000 | 1.0000  | 0.7506 |
| monks-2        | 2     | 0.6623 | 0.6071 | 0.6623  | 0.6623 |
| monks-2        | 3     | 0.6623 | 0.5850 | 0.6623  | 0.5717 |
| monks-2        | 4     | 0.6623 | 0.5806 | 0.6623  | 0.6026 |
| monks-2        | 5     | 0.6623 | 0.5894 | 0.6623  | 0.6689 |
| monks-3        | 2     | 0.9664 | 0.9664 | 0.9664  | 0.9664 |
| monks-3        | 3     | 0.9904 | 0.9904 | 0.9904  | 0.9784 |
| monks-3        | 4     | 0.9904 | 0.9880 | 0.9904  | 0.9904 |
| monks-3        | 5     | 0.9736 | 0.9832 | 0.9904  | 0.9904 |
| soybean-small  | 2     | 1.0000 | 0.9722 | 1.0000  | 0.7500 |
| soybean-small  | 3     | 0.9444 | 0.7500 | 0.9722  | 0.9722 |
| soybean-small  | 4     | 0.9444 | 0.8333 | 0.9444  | 0.9722 |
| soybean-small  | 5     | 0.9722 | 0.7222 | 0.9722  | 0.9722 |
| spect          | 2     | 0.8358 | 0.7811 | 0.8358  | 0.7811 |
| spect          | 3     | 0.8358 | 0.7562 | 0.8358  | 0.8060 |
| spect          | 4     | 0.8358 | 0.7413 | 0.7910  | 0.7761 |
| spect          | 5     | 0.7910 | 0.7313 | 0.7910  | 0.7811 |
| tic-tac-toe    | 2     | 0.6889 | 0.6889 | 0.6889  | 0.7097 |
| tic-tac-toe    | 3     | 0.7514 | 0.7458 | 0.7625  | 0.7125 |
| tic-tac-toe    | 4     | 0.7333 | 0.8014 | 0.7486  | 0.8000 |
| tic-tac-toe    | 5     | 0.6958 | 0.7958 | 0.7361  | 0.8097 |


#### Optimality Gap for OCT, binOCT, flowOCT
| instance       | depth | OCT     | binOCT  | flowOCT |
|----------------|-------|---------|---------|---------|
| balance-scale  | 2     | 0.00%   | 0.00%   | 0.00%   |
| balance-scale  | 3     | 93.82%  | 84.57%  | 0.00%   |
| balance-scale  | 4     | 95.52%  | 100.00% | 19.92%  |
| balance-scale  | 5     | 97.87%  | 100.00% | 20.64%  |
| breast-cancer  | 2     | 0.00%   | 0.00%   | 0.00%   |
| breast-cancer  | 3     | 89.21%  | 99.96%  | 0.00%   |
| breast-cancer  | 4     | 92.47%  | 100.00% | 20.40%  |
| breast-cancer  | 5     | 93.06%  | 100.00% | 24.14%  |
| car-evaluation | 2     | 99.29%  | 0.00%   | 0.00%   |
| car-evaluation | 3     | 99.62%  | 94.30%  | 17.55%  |
| car-evaluation | 4     | 100.00% | 100.00% | 21.76%  |
| car-evaluation | 5     | 100.00% | 100.00% | 24.51%  |
| hayes-roth     | 2     | 0.00%   | 0.00%   | 0.00%   |
| hayes-roth     | 3     | 69.15%  | 29.71%  | 0.00%   |
| hayes-roth     | 4     | 92.10%  | 100.00% | 2.71%   |
| hayes-roth     | 5     | 100.00% | 100.00% | 10.40%  |
| house-votes-84 | 2     | 0.00%   | 0.00%   | 0.00%   |
| house-votes-84 | 3     | 0.00%   | 66.04%  | 0.00%   |
| house-votes-84 | 4     | 0.00%   | 33.33%  | 0.00%   |
| house-votes-84 | 5     | 0.00%   | 0.00%   | 0.00%   |
| monks-1        | 2     | 0.00%   | 0.00%   | 0.00%   |
| monks-1        | 3     | 100.00% | 24.89%  | 0.00%   |
| monks-1        | 4     | 66.67%  | 66.67%  | 0.00%   |
| monks-1        | 5     | 94.67%  | 0.00%   | 0.00%   |
| monks-2        | 2     | 0.00%   | 0.00%   | 0.00%   |
| monks-2        | 3     | 21.39%  | 99.42%  | 0.00%   |
| monks-2        | 4     | 64.39%  | 100.00% | 0.00%   |
| monks-2        | 5     | 97.43%  | 100.00% | 0.00%   |
| monks-3        | 2     | 0.00%   | 0.00%   | 0.00%   |
| monks-3        | 3     | 33.33%  | 0.00%   | 0.00%   |
| monks-3        | 4     | 100.00% | 100.00% | 0.48%   |
| monks-3        | 5     | 79.38%  | 100.00% | 0.00%   |
| soybean-small  | 2     | 0.00%   | 0.00%   | 0.00%   |
| soybean-small  | 3     | 0.00%   | 0.00%   | 0.00%   |
| soybean-small  | 4     | 0.00%   | 0.00%   | 0.00%   |
| soybean-small  | 5     | 0.00%   | 0.00%   | 0.00%   |
| spect          | 2     | 0.00%   | 0.00%   | 0.00%   |
| spect          | 3     | 0.00%   | 13.76%  | 0.00%   |
| spect          | 4     | 0.00%   | 48.55%  | 0.75%   |
| spect          | 5     | 86.84%  | 53.65%  | 3.62%   |
| tic-tac-toe    | 2     | 20.71%  | 0.00%   | 0.00%   |
| tic-tac-toe    | 3     | 94.79%  | 100.00% | 23.22%  |
| tic-tac-toe    | 4     | 100.00% | 100.00% | 26.86%  |
| tic-tac-toe    | 5     | 100.00% | 100.00% | 31.55%  |

#### Training Time for OCT, binOCT, flowOCT and CART
| instance       | depth | OCT     | binOCT  | flowOCT | CART  |
|----------------|-------|---------|---------|---------|-------|
| balance-scale  | 2     | 324.485 | 6.703   | 1.948   | 0.016 |
| balance-scale  | 3     | 600.985 | 601.105 | 187.126 | 0.001 |
| balance-scale  | 4     | 602.283 | 601.775 | 600.087 | 0.001 |
| balance-scale  | 5     | 606.111 | 603.954 | 600.100 | 0.000 |
| breast-cancer  | 2     | 76.785  | 12.278  | 1.327   | 0.000 |
| breast-cancer  | 3     | 600.662 | 600.958 | 7.482   | 0.000 |
| breast-cancer  | 4     | 601.735 | 601.599 | 600.201 | 0.001 |
| breast-cancer  | 5     | 603.483 | 603.669 | 600.165 | 0.001 |
| car-evaluation | 2     | 601.157 | 21.770  | 37.913  | 0.001 |
| car-evaluation | 3     | 602.927 | 601.778 | 600.066 | 0.001 |
| car-evaluation | 4     | 607.996 | 605.761 | 600.285 | 0.001 |
| car-evaluation | 5     | 618.090 | 613.143 | 600.488 | 0.000 |
| hayes-roth     | 2     | 25.766  | 2.432   | 0.892   | 0.000 |
| hayes-roth     | 3     | 600.384 | 539.534 | 18.473  | 0.000 |
| hayes-roth     | 4     | 600.896 | 600.857 | 447.373 | 0.000 |
| hayes-roth     | 5     | 602.023 | 601.833 | 600.136 | 0.001 |
| house-votes-84 | 2     | 6.927   | 1.323   | 0.731   | 0.000 |
| house-votes-84 | 3     | 3.303   | 514.985 | 0.267   | 0.000 |
| house-votes-84 | 4     | 7.094   | 204.534 | 0.237   | 0.000 |
| house-votes-84 | 5     | 22.034  | 6.601   | 0.357   | 0.000 |
| monks-1        | 2     | 68.489  | 3.624   | 3.230   | 0.001 |
| monks-1        | 3     | 601.394 | 475.691 | 31.504  | 0.001 |
| monks-1        | 4     | 423.765 | 404.370 | 9.361   | 0.001 |
| monks-1        | 5     | 608.339 | 12.078  | 24.010  | 0.001 |
| monks-2        | 2     | 62.953  | 26.734  | 3.040   | 0.000 |
| monks-2        | 3     | 601.407 | 601.485 | 24.141  | 0.001 |
| monks-2        | 4     | 604.056 | 602.460 | 37.813  | 0.001 |
| monks-2        | 5     | 608.826 | 606.907 | 24.869  | 0.001 |
| monks-3        | 2     | 27.279  | 1.292   | 1.432   | 0.000 |
| monks-3        | 3     | 256.932 | 199.872 | 18.712  | 0.000 |
| monks-3        | 4     | 603.237 | 602.149 | 600.129 | 0.001 |
| monks-3        | 5     | 609.401 | 606.728 | 8.107   | 0.001 |
| soybean-small  | 2     | 3.059   | 0.329   | 0.475   | 0.001 |
| soybean-small  | 3     | 1.674   | 0.297   | 0.326   | 0.001 |
| soybean-small  | 4     | 11.630  | 0.749   | 0.763   | 0.001 |
| soybean-small  | 5     | 17.474  | 1.842   | 1.631   | 0.000 |
| spect          | 2     | 3.590   | 8.102   | 0.270   | 0.000 |
| spect          | 3     | 9.966   | 431.826 | 0.360   | 0.000 |
| spect          | 4     | 6.822   | 601.869 | 368.500 | 0.001 |
| spect          | 5     | 605.338 | 605.569 | 600.283 | 0.001 |
| tic-tac-toe    | 2     | 567.332 | 159.264 | 63.977  | 0.001 |
| tic-tac-toe    | 3     | 602.668 | 602.060 | 600.130 | 0.001 |
| tic-tac-toe    | 4     | 606.554 | 605.508 | 600.159 | 0.001 |
| tic-tac-toe    | 5     | 615.277 | 621.808 | 600.446 | 0.001 |

#### Average Solution for flowOCT and SOCT Solved with Robust Optimization and Cutting Plane Methods
| instance       | flowOCT | SOCT_CP | SOCT_RB |
|----------------|---------|---------|---------|
| balance-scale  | 3.30    | 15.97   | 7.17    |
| breast-cancer  | 6.18    | 16.53   | 11.84   |
| car_evaluation | 24.16   | 108.59  | 54.20   |
| hayes-roth     | 0.67    | 4.10    | 1.23    |
| house-votes-84 | 0.58    | 1.31    | 1.20    |
| monk1          | 0.47    | 1.18    | 0.81    |
| monk2          | 0.77    | 5.81    | 2.66    |
| monk3          | 0.32    | 0.79    | 0.62    |
| soybean-small  | 0.32    | 0.62    | 1.28    |
| spect          | 1.80    | 5.67    | 5.14    |
| tic-tac-toe    | 71.50   | 246.28  | 88.11   |

