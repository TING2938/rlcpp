
#%%
import numpy as np
import pandas as pd
import tensorflow as tf

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    return x, y

#%%
data = pd.read_csv('data.csv')

# %%
batchsz = 64
x = data.loc[:, ['Kmin', 'Kmax', 'Pmax', 'Incast', 'BW_avg']]
y = data.loc[:, ['qlen', 'pause_num', 'pause_time']]

x = x.to_numpy()
y = y.to_numpy()

meanx = x.mean(axis=0)
stdx = x.std(axis=0)

meany = y.mean(axis=0)
stdy = y.std(axis=0)

x -= meanx 
x /= stdx

y -= meany 
y /= stdy 

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsz)
# %% for network

network = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3)
])
network.build(input_shape=(None, 5))
network.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.01),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
network.summary()
network.load_weights('router_weight.ckpt')
#%% fit
network.fit(db, epochs=1000)

network.save_weights('router_weight.ckpt')
