import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers, Sequential


class MyAgent:
    def __init__(self, obs_n, act_n, lr):
        self.obs_n = obs_n
        self.act_n = act_n
        self.lr = lr
        self.model = self.creat_model()

    # 创建模型
    def creat_model(self):
        model = keras.Sequential([
            keras.layers.Input((self.obs_n,)),
            keras.layers.Dense(self.act_n*10, activation='tanh'),
            keras.layers.Dense(self.act_n, activation='softmax')
        ])
        return model

    # 根据观测值，采样输出动作，带探索过程
    def sample(self, obs):
        obs = tf.constant(obs, dtype=tf.float32)
        obs = tf.expand_dims(obs, axis=0)
        act_prob = self.model(obs)
        # 根据概率选择动作
        action = tf.random.categorical(tf.math.log(act_prob), 1)[0]
        action = int(action)
        return action

    # 根据输入观测值，预测下一步动作
    def predict(self, obs):
        obs = tf.constant(obs, dtype=tf.float32)
        obs = tf.expand_dims(obs, axis=0)
        act_prob = self.model(obs)
        act_prob = tf.squeeze(act_prob, axis=0)
        action = tf.argmax(act_prob)
        return int(action)

    # 学习过程（即更新Q表格的过程）
    def learn(self, obs, action, reward):
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        with tf.GradientTape() as tape:
            act_prob = self.model(obs)
            log_prob = tf.reduce_sum(-1.0 * 
            tf.math.log(act_prob) * tf.one_hot(action, act_prob.shape[1]), axis=1)
            loss = log_prob * reward
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))