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

### Results

All the tests are conducted on Intel(R) Core(TM) CPU i7-7700HQ @ 2.80GHz and a memory of 16 GB. Algorithms are implemented in Python 3.7 with Gurobi 9.1 as an optimization solver. The time limit is set to 600 seconds.

#### Out-of-Sample Prediction Performance for OCT, binOCT, flowOCT, and CART
| instance       | depth | OCT    | bOCT   | flowOCT | Sklearn |
|----------------|-------|--------|--------|---------|---------|
| balance-scale  | 2     | 0.6638 | 0.6497 | 0.6667  | 0.6285  |
| balance-scale  | 3     | 0.6667 | 0.6603 | 0.6900  | 0.7197  |
| balance-scale  | 4     | 0.6971 | 0.6624 | 0.6907  | 0.7707  |
| balance-scale  | 5     | 0.6624 | 0.5754 | 0.6858  | 0.7707  |
| breast-cancer  | 2     | 0.6873 | 0.6857 | 0.6825  | 0.7095  |
| breast-cancer  | 3     | 0.7048 | 0.7143 | 0.6952  | 0.7286  |
| breast-cancer  | 4     | 0.6825 | 0.7048 | 0.7095  | 0.7143  |
| breast-cancer  | 5     | 0.6921 | 0.6286 | 0.7016  | 0.6714  |
| car-evaluation | 2     | 0.7418 | 0.7654 | 0.7418  | 0.7739  |
| car-evaluation | 3     | 0.7446 | 0.7863 | 0.7639  | 0.7724  |
| car-evaluation | 4     | 0.7449 | 0.8333 | 0.7711  | 0.8364  |
| car-evaluation | 5     | 0.7022 | 0.8279 | 0.7631  | 0.8341  |
| hayes-roth     | 2     | 0.5806 | 0.5500 | 0.5972  | 0.5333  |
| hayes-roth     | 3     | 0.7028 | 0.7000 | 0.6861  | 0.6083  |
| hayes-roth     | 4     | 0.7417 | 0.6417 | 0.7611  | 0.6917  |
| hayes-roth     | 5     | 0.7556 | 0.6167 | 0.7750  | 0.7833  |
| house-votes-84 | 2     | 0.9713 | 0.9655 | 0.9713  | 0.9713  |
| house-votes-84 | 3     | 0.9674 | 0.9713 | 0.9674  | 0.9713  |
| house-votes-84 | 4     | 0.9617 | 0.9253 | 0.9617  | 0.9598  |
| house-votes-84 | 5     | 0.9693 | 0.9598 | 0.9617  | 0.9598  |
| monks-1        | 2     | 0.7434 | 0.7050 | 0.7442  | 0.7314  |
| monks-1        | 3     | 0.8034 | 0.8393 | 0.8161  | 0.7890  |
| monks-1        | 4     | 0.8457 | 0.9041 | 0.9105  | 0.7770  |
| monks-1        | 5     | 0.8185 | 0.8993 | 0.9105  | 0.7602  |
| monks-2        | 2     | 0.6461 | 0.6137 | 0.6468  | 0.6623  |
| monks-2        | 3     | 0.6328 | 0.5960 | 0.6034  | 0.5717  |
| monks-2        | 4     | 0.6313 | 0.5784 | 0.6181  | 0.6026  |
| monks-2        | 5     | 0.6269 | 0.7660 | 0.6394  | 0.7064  |
| monks-3        | 2     | 0.9384 | 0.8345 | 0.9664  | 0.9664  |
| monks-3        | 3     | 0.9680 | 0.9113 | 0.9824  | 0.9640  |
| monks-3        | 4     | 0.9824 | 0.6739 | 0.9824  | 0.9904  |
| monks-3        | 5     | 0.9824 | 0.6835 | 0.9824  | 0.9904  |
| soybean-small  | 2     | 0.8148 | 0.9722 | 0.9630  | 0.7778  |
| soybean-small  | 3     | 0.9537 | 0.7500 | 0.9444  | 1.0000  |
| soybean-small  | 4     | 0.8704 | 0.8333 | 0.9352  | 0.9167  |
| soybean-small  | 5     | 0.8704 | 0.7222 | 0.9444  | 1.0000  |
| spect          | 2     | 0.7993 | 0.7811 | 0.7993  | 0.7811  |
| spect          | 3     | 0.7910 | 0.7562 | 0.7794  | 0.8060  |
| spect          | 4     | 0.7297 | 0.7413 | 0.7844  | 0.7811  |
| spect          | 5     | 0.7894 | 0.7214 | 0.7811  | 0.7662  |
| tic-tac-toe    | 2     | 0.6861 | 0.6889 | 0.6903  | 0.7097  |
| tic-tac-toe    | 3     | 0.7269 | 0.7458 | 0.7366  | 0.7125  |
| tic-tac-toe    | 4     | 0.7417 | 0.7875 | 0.7287  | 0.8000  |
| tic-tac-toe    | 5     | 0.7079 | 0.7944 | 0.7236  | 0.8097  |

#### Solution Time for OCT, binOCT, flowOCT and CART
| instance       | depth | OCT    | bOCT   | flowOCT | Sklearn |
|----------------|-------|--------|--------|---------|---------|
| balance-scale  | 2     | 199.01 | 5.50   | 1.68    | 0.00    |
| balance-scale  | 3     | 600.82 | 601.13 | 90.03   | 0.00    |
| balance-scale  | 4     | 602.13 | 601.79 | 419.37  | 0.00    |
| balance-scale  | 5     | 604.82 | 603.78 | 496.10  | 0.00    |
| breast-cancer  | 2     | 49.98  | 9.50   | 2.96    | 0.00    |
| breast-cancer  | 3     | 566.47 | 601.07 | 401.79  | 0.00    |
| breast-cancer  | 4     | 598.23 | 601.45 | 401.25  | 0.00    |
| breast-cancer  | 5     | 561.77 | 603.09 | 401.46  | 0.00    |
| car-evaluation | 2     | 546.85 | 13.35  | 24.65   | 0.00    |
| car-evaluation | 3     | 602.16 | 601.48 | 416.04  | 0.00    |
| car-evaluation | 4     | 605.61 | 603.61 | 428.08  | 0.00    |
| car-evaluation | 5     | 616.89 | 610.86 | 443.64  | 0.00    |
| hayes-roth     | 2     | 13.32  | 1.51   | 0.63    | 0.00    |
| hayes-roth     | 3     | 600.26 | 476.06 | 10.12   | 0.00    |
| hayes-roth     | 4     | 600.63 | 600.87 | 274.22  | 0.00    |
| hayes-roth     | 5     | 601.47 | 601.82 | 570.76  | 0.00    |
| house-votes-84 | 2     | 2.99   | 0.55   | 0.31    | 0.00    |
| house-votes-84 | 3     | 39.57  | 383.97 | 6.65    | 0.00    |
| house-votes-84 | 4     | 175.62 | 203.04 | 14.16   | 0.00    |
| house-votes-84 | 5     | 228.76 | 4.77   | 5.55    | 0.00    |
| monks-1        | 2     | 62.27  | 2.26   | 1.64    | 0.00    |
| monks-1        | 3     | 600.74 | 364.60 | 20.16   | 0.00    |
| monks-1        | 4     | 601.94 | 601.16 | 18.56   | 0.00    |
| monks-1        | 5     | 604.41 | 29.07  | 19.90   | 0.00    |
| monks-2        | 2     | 164.98 | 5.55   | 11.45   | 0.00    |
| monks-2        | 3     | 600.79 | 601.06 | 405.49  | 0.00    |
| monks-2        | 4     | 601.97 | 601.56 | 407.86  | 0.00    |
| monks-2        | 5     | 604.79 | 603.65 | 405.16  | 0.00    |
| monks-3        | 2     | 11.89  | 0.69   | 0.79    | 0.02    |
| monks-3        | 3     | 391.99 | 40.54  | 5.36    | 0.00    |
| monks-3        | 4     | 417.42 | 601.01 | 203.52  | 0.00    |
| monks-3        | 5     | 449.67 | 602.93 | 201.92  | 0.00    |
| soybean-small  | 2     | 1.11   | 0.25   | 0.32    | 0.00    |
| soybean-small  | 3     | 2.09   | 0.20   | 0.46    | 0.00    |
| soybean-small  | 4     | 4.46   | 0.48   | 0.43    | 0.00    |
| soybean-small  | 5     | 7.53   | 1.34   | 0.69    | 0.00    |
| spect          | 2     | 5.11   | 4.62   | 1.10    | 0.00    |
| spect          | 3     | 305.43 | 419.86 | 94.97   | 0.00    |
| spect          | 4     | 402.07 | 601.38 | 298.74  | 0.00    |
| spect          | 5     | 403.85 | 604.00 | 400.11  | 0.00    |
| tic-tac-toe    | 2     | 323.67 | 105.23 | 35.48   | 0.00    |
| tic-tac-toe    | 3     | 601.56 | 601.49 | 437.20  | 0.00    |
| tic-tac-toe    | 4     | 604.04 | 603.74 | 432.64  | 0.00    |
| tic-tac-toe    | 5     | 610.13 | 612.51 | 427.99  | 0.00    |

#### Average Out-of-Sample Prediction Accuracy over 10 Trails for flowOCT and SOCT Solved with Robust Optimization and Cutting Plane Methods
| instance       | depth | OCT    | bOCT   | flowOCT | Sklearn |
|----------------|-------|--------|--------|---------|---------|
| balance-scale  | 2     | 0.6638 | 0.6497 | 0.6667  | 0.6285  |
| balance-scale  | 3     | 0.6667 | 0.6603 | 0.6900  | 0.6285  |
| balance-scale  | 4     | 0.6971 | 0.6624 | 0.6907  | 0.6285  |
| balance-scale  | 5     | 0.6624 | 0.5754 | 0.6858  | 0.6285  |
| breast-cancer  | 2     | 0.6873 | 0.6857 | 0.6825  | 0.7095  |
| breast-cancer  | 3     | 0.7048 | 0.7143 | 0.6952  | 0.7095  |
| breast-cancer  | 4     | 0.6825 | 0.7048 | 0.7095  | 0.7095  |
| breast-cancer  | 5     | 0.6921 | 0.6286 | 0.7016  | 0.7095  |
| car-evaluation | 2     | 0.7418 | 0.7654 | 0.7418  | 0.7739  |
| car-evaluation | 3     | 0.7446 | 0.7863 | 0.7639  | 0.7739  |
| car-evaluation | 4     | 0.7449 | 0.8333 | 0.7711  | 0.7739  |
| car-evaluation | 5     | 0.7022 | 0.8279 | 0.7631  | 0.7739  |
| hayes-roth     | 2     | 0.5806 | 0.5500 | 0.5972  | 0.5250  |
| hayes-roth     | 3     | 0.7028 | 0.7000 | 0.6861  | 0.5250  |
| hayes-roth     | 4     | 0.7417 | 0.6417 | 0.7611  | 0.5250  |
| hayes-roth     | 5     | 0.7556 | 0.6167 | 0.7750  | 0.5250  |
| house-votes-84 | 2     | 0.9713 | 0.9655 | 0.9713  | 0.9713  |
| house-votes-84 | 3     | 0.9674 | 0.9713 | 0.9674  | 0.9713  |
| house-votes-84 | 4     | 0.9617 | 0.9253 | 0.9617  | 0.9713  |
| house-votes-84 | 5     | 0.9693 | 0.9598 | 0.9617  | 0.9713  |
| monks-1        | 2     | 0.7434 | 0.7050 | 0.7442  | 0.7314  |
| monks-1        | 3     | 0.8034 | 0.8393 | 0.8161  | 0.7314  |
| monks-1        | 4     | 0.8457 | 0.9041 | 0.9105  | 0.7314  |
| monks-1        | 5     | 0.8185 | 0.8993 | 0.9105  | 0.7314  |
| monks-2        | 2     | 0.6461 | 0.6137 | 0.6468  | 0.6623  |
| monks-2        | 3     | 0.6328 | 0.5960 | 0.6034  | 0.6623  |
| monks-2        | 4     | 0.6313 | 0.5784 | 0.6181  | 0.6623  |
| monks-2        | 5     | 0.6269 | 0.7660 | 0.6394  | 0.6623  |
| monks-3        | 2     | 0.9384 | 0.8345 | 0.9664  | 0.9664  |
| monks-3        | 3     | 0.9680 | 0.9113 | 0.9824  | 0.9664  |
| monks-3        | 4     | 0.9824 | 0.6739 | 0.9824  | 0.9664  |
| monks-3        | 5     | 0.9824 | 0.6835 | 0.9824  | 0.9664  |
| soybean-small  | 2     | 0.8148 | 0.9722 | 0.9630  | 0.7778  |
| soybean-small  | 3     | 0.9537 | 0.7500 | 0.9444  | 0.7500  |
| soybean-small  | 4     | 0.8704 | 0.8333 | 0.9352  | 0.7500  |
| soybean-small  | 5     | 0.8704 | 0.7222 | 0.9444  | 0.7222  |
| spect          | 2     | 0.7993 | 0.7811 | 0.7993  | 0.7811  |
| spect          | 3     | 0.7910 | 0.7562 | 0.7794  | 0.7811  |
| spect          | 4     | 0.7297 | 0.7413 | 0.7844  | 0.7811  |
| spect          | 5     | 0.7894 | 0.7214 | 0.7811  | 0.7811  |
| tic-tac-toe    | 2     | 0.6861 | 0.6889 | 0.6903  | 0.7097  |
| tic-tac-toe    | 3     | 0.7269 | 0.7458 | 0.7366  | 0.7097  |
| tic-tac-toe    | 4     | 0.7417 | 0.7875 | 0.7287  | 0.7097  |
| tic-tac-toe    | 5     | 0.7079 | 0.7944 | 0.7236  | 0.7097  |

#### Average Solution over 10 Trails for flowOCT and SOCT Solved with Robust Optimization and Cutting Plane Methods
| instance       | flowOCT | SOCT_CP | SOCT_R |
|----------------|---------|---------|--------|
| soybean-small  | 0.32    | 0.62    | 1.28   |
| monk3          | 0.32    | 0.79    | 0.62   |
| monk1          | 0.47    | 1.18    | 0.81   |
| hayes-roth     | 0.67    | 4.10    | 1.23   |
| monk2          | 0.77    | 5.81    | 2.66   |
| house-votes-84 | 0.58    | 1.31    | 1.20   |
| spect          | 1.80    | 5.67    | 5.14   |
| breast-cancer  | 6.18    | 16.53   | 11.84  |
| balance-scale  | 3.30    | 15.97   | 7.17   |
| tic-tac-toe    | 71.50   | 246.28  | 88.11  |
| car_evaluation | 24.16   | 108.59  | 54.20  |
