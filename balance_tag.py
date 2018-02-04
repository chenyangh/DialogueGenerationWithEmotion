import numpy as np
import pandas as pd


def balance_data(f_name):
    emo_arr = np.loadtxt('OpenSubData/' + f_name + '.tag', dtype=np.int32)
    emo_idx = {}
    max_count = 99999999999
    for idx in range(9):
        emo_idx[idx] = np.where(emo_arr == idx)[0]
        if len(emo_idx[idx]) < max_count:
            max_count = len(emo_idx[idx])

    balanced_list = []
    if 'text' in f_name:
        max_count = 10000  # size of the test set will be max_count * 9

    import random
    for idx in range(9):
        balanced_list.extend(random.sample(emo_idx[idx].tolist(), max_count))
    balanced_list = sorted(balanced_list)
    np.savetxt('OpenSubData/' + f_name + '_balance.tag', np.asarray(emo_arr[balanced_list], dtype=np.int32), fmt="%d")
    df = pd.read_csv('OpenSubData/' + f_name + '.csv')
    df = df.iloc[balanced_list]
    df.to_csv('OpenSubData/' + f_name + '_balance.csv')


if __name__ == '__main__':
    balance_data('data_2_train')
    balance_data('data_2_test')
    balance_data('data_6_train')
    balance_data('data_6_test')
