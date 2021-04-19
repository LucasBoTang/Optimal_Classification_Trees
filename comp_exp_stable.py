from dataset import dataloader
import pandas as pd
import time
from MaxFlow_OCT import MaxFlow_OCT
import numpy as np
from sklearn.model_selection import train_test_split


seeds = [11, 23, 34, 45, 56, 67, 78, 89, 93, 5]
depths = [3]
datasets = ['balance-scale', 'breast-cancer', 'car_evaluation', 'hayes-roth', 'house-votes-84',
            'soybean-small', 'spect', 'tic-tac-toe', 'monk1', 'monk2', 'monk3']

alphas = [0, 0.01, 0.1]
repeat = [1]

rec_oct, rec_soctr, rec_soctcp = [], [], []

for seed in seeds:
    for depth in depths:
        for dataset in datasets:

            x_train, x_test, y_train, y_test = dataloader(dataset, seed)
            print('\n{} -- Train: {}, Test: {}'.format(dataset, len(y_train), len(y_test)))
            N = int(len(y_train) * 0.666)

            for alpha in alphas:

                x_train_oct, x_vali, y_train_oct, y_vali = train_test_split(x_train, y_train, test_size=0.3333, random_state=r*3)
                model = MaxFlow_OCT(depth, alpha)
                tick = time.time()
                model.fit(x_train_oct, y_train_oct)
                t_oct = time.time() - tick
                oct_train = model.eval(x_train_oct, y_train_oct)
                oct_test = model.eval(x_test, y_test)
                oct_vali = model.eval(x_vali, y_vali)
                print(' OCT Train: {:.2f}, Vali: {:.2f} Test: {:.2f}'.format(oct_train, oct_vali, oct_test))

                rec_oct.append([dataset, depth, alpha, seed, oct_train, oct_vali, oct_test, t_oct])
                df = pd.DataFrame(rec_oct, columns=['instance', 'depth', 'alpha', 'seed',
                                                    'Train', 'Vali', 'Test', 'Time'])
                df.to_csv('./res/flowOCT-3.csv', index=False)

                model = MaxFlow_OCT(depth, alpha)
                tick = time.time()
                model.stable_fit_robust(x_train, y_train, N=N)
                soctr_t = time.time() - tick
                soctr_train = model.eval(x_train, y_train)
                soctr_test = model.eval(x_test, y_test)
                print('SOCTR Train: {:.2f}, Test: {:.2f}'.format(soctr_train, soctr_test))
                rec_soctr.append([dataset, depth, alpha, seed, soctr_train, soctr_test, soctr_t])
                df = pd.DataFrame(rec_soctr, columns=['instance', 'depth', 'alpha', 'seed',
                                                      'Train', 'Test', 'Time'])
                df.to_csv('./res/SOCT_Robust-3.csv', index=False)

                model = MaxFlow_OCT(depth, alpha)
                tick = time.time()
                model.stable_fit_CP(x_train, y_train, N=N)
                soctcp_t = time.time() - tick
                soctcp_train = model.eval(x_train, y_train)
                soctcp_test = model.eval(x_test, y_test)
                print('SOCT Train: {:.2f}, Test: {:.2f}'.format(soctcp_train, soctcp_test))

                rec_soctcp.append([dataset, depth, alpha, seed, soctcp_train, soctcp_test, soctcp_t])
                df = pd.DataFrame(rec_soctcp, columns=['instance', 'depth', 'alpha', 'seed',
                                                    'Train', 'Test', 'Time'])
                df.to_csv('./res/SOCT_CP-3.csv', index=False)