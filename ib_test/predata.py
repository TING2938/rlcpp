#%%
import numpy as np
import tensorflow as tf

kmin, kmax, pmax, rateLimit, thrput, RTT, outputAvgSpeed = np.loadtxt(
            'ib_test/incast_5_rate_multi.dat', 
            dtype={
                    'names':   ('kmin', 'kmax', 'pmax', 'rateLimit', 'thrput', 'RTT', 'outputAvgSpeed'), 
                    'formats': ('i4', 'i4', 'i4', 'i4', list, list, 'f4')
                    }, 
            delimiter=';',
            unpack=True
        )

getmean = lambda x: np.mean(eval(x))
thrput = np.vectorize(getmean)(thrput)
RTT = np.vectorize(getmean)(RTT)

#%%

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[test_indices]

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    return x, y

# %%
batchsz = 32
x = np.array((kmin, kmax, rateLimit, thrput)).T
y = np.array((RTT, outputAvgSpeed)).T

meanx = x.mean(axis=0)
stdx = x.std(axis=0)

meany = y.mean(axis=0)
stdy = y.std(axis=0)

#%%
x -= meanx 
x /= stdx

y -= meany 
y /= stdy 

train_x, test_x = split_train_test(x, 0.2)
train_y, test_y = split_train_test(y, 0.2)

db_train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
db_train = db_train.map(preprocess).shuffle(60000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
db_test = db_test.map(preprocess).batch(batchsz)
# %% for network

network = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2)
])
network.build(input_shape=(None, 4))
network.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='MSE',
    metrics=['accuracy'])
network.summary()
# network.load_weights('router_weight.ckpt')
#%% fit
history = network.fit(db_train, epochs=1000, validation_data=db_test, validation_freq=10)

network.save('af1.h5')
network.save_weights('router_weight1.ckpt')

# %%
