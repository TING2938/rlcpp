#%%
import json
import numpy as np
import pandas as pd

#%%
prefix = "/home/x19860/AIECN/data_25G/"
fnm = ['result_25G.log']
head = ['Incast', 'ecn_low_limit', 'ecn_high_limit', 'ecn_discard_prob']

fp = open(prefix + fnm[0])
all_data = json.load(fp)
all_data = pd.DataFrame(all_data)
fp.close()

# %%
keys = list(all_data.keys())

#%%
data = pd.DataFrame()

data['Kmin'] = all_data['ecn_low_limit']
data['Kmax'] = all_data['ecn_high_limit']
data['Pmax'] = all_data['ecn_discard_prob']
data['Incast'] = all_data['Incast']

data['qlen'] = all_data['queue_length_list'].apply(lambda s: np.max([int(i) for i in (s + '0').replace('interface', '').split()]))
data['BW_avg'] = all_data.loc[:, [f'BW_avg{i+1}' for i in range(8)]].sum(axis=1)
data['pause_num'] = all_data.loc[:, [f'client{i+1}_rx_prioX_pause' for i in range(8)]].sum(axis=1)
data['pause_time'] = all_data.loc[:, [f'client{i+1}_rx_prioX_pause_duration' for i in range(8)]].sum(axis=1)

# %%
data.to_csv('data.csv')
# %%
