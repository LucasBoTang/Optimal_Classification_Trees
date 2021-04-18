# Optimal Decision Tree

<p align="center"><img width="100%" src="fig/dt.jpg" /></p>
 
### Introduction

This project aims to better understand the computational and prediction performance of MIP-based classification tree learning approaches proposed by recent research papers. In particular, the OCT, binOCT, and flowOCT formulations are implemented and evaluated on publicly available classification datasets. Moreover, we propose a stable classification tree formulation incorporating the training and validation split decision into the model fitting processes. Computational results suggest flowOCT has the most consistent performance across datasets and tree complexity. OCT is competitive on small datasets and shallow trees, while binOCT yields the best performance on large datasets and deep trees. The proposed stable classification tree achieves stronger out-of-sample performance than flowOCT under random training and validation set splits.

### Dependencies

* [Python 3.7](https://www.python.org/)
* [Gurobi 9.1](https://www.gurobi.com/)

### Models

- [Optimal Classification Trees (OCT)](https://link.springer.com/article/10.1007/s10994-017-5633-9) - Bertsimas, D., & Dunn, J. (2017). Optimal classification trees. Machine Learning, 106(7), 1039-1082.
- [Optimal Classification Tree with Binary Encoding (binOCT)](https://ojs.aaai.org//index.php/AAAI/article/view/3978) - Verwer, S., & Zhang, Y. (2019). Learning optimal classification trees using a binary linear program formulation. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 1625-1632).
- [Max-Flow Optimal Classification Trees (flowOCT)](http://www.optimization-online.org/DB_HTML/2021/01/8220.html) - Aghaei, S., Gómez, A., & Vayanos, P. (2021). Strong Optimal Classification Trees. arXiv preprint arXiv:2103.15965.
