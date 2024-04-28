#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/22/2023 3:31 PM
# @Author  : xiaomanl
# @File    : model_utils.py
# @Software: PyCharm
from sklearn.preprocessing import LabelEncoder , StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.utils import custom_object_scope, Sequence
from tensorflow.keras.layers import Layer
import keras.backend as K
import numpy as np
import tensorflow as tf
import time
import pandas as pd
from collections import OrderedDict
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SimpleDense(Layer):

    def __init__(self, units=32):
        '''Initializes the instance attributes'''
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        # initialize the weights
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="kernel",
            initial_value=w_init(shape=(input_shape[-1], self.units),
                                 dtype='float32'),
            trainable=True)

        # initialize the biases
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias",
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True)

    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        return tf.matmul(inputs, self.w) + self.b

class CustomEmbeddingDataGenerator(Sequence):
    def __init__(self, x_train, y_train, batch_size):
        self.char_var_train , self.numeric_var_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.char_var_train) / float(self.batch_size)))

    def __getitem__(self, idx):
        char_var_batch = self.char_var_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        numeric_var_batch = self.numeric_var_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_batch = self.y_train[idx * self.batch_size:(idx + 1) * self.batch_size]

        input_batch = {'char_var_input': char_var_batch, 'numeric_var_input': numeric_var_batch}

        yield input_batch, y_batch


class NBatchLogger(Callback):
    def __init__(self, display):
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            # you can access loss, accuracy in self.params['metrics']
            print('\n{}/{} - loss ....\n'.format(self.seen, self.params['nb_sample']))

# class AttentionWeightCollector(Callback):
#     def __init__(self, name):
#         super(AttentionWeightCollector, self).__init__()
#         self.attention_weights = []
#         self.layer_name = name
#         self.get_attention_weights_function = K.function(inputs=[self.ncpp.input],
#                                                          outputs=[self.ncpp.get_layer(self.layer_name).output])
#
#
#     def on_batch_end(self, batch, logs=None):
#         attention_layer = self.ncpp.get_layer(self.layer_name)
#         # _, attention_weights_tensor = attention_layer.output
#         # if tf.executing_eagerly():
#         #     attention_weights_tensor = tf.convert_to_tensor(attention_weights_tensor)
#         #     attention_weights_value = attention_weights_tensor.numpy()
#         # else:
#         #     attention_weights_value = tf.keras.backend.get_value(attention_weights_tensor)
#         _, attention_weights_value = self.get_attention_weights_function([self.validation_data])[0]
#         self.attention_weights.append(attention_weights_value)

class LogPrintCallback(tf.keras.callbacks.Callback):

    def __init__(self, interval=50):
        super(LogPrintCallback, self).__init__()
        self.interval = interval

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            self.epoch_start_time = time.time()
        else:
            self.epoch_start_time = float("-inf")
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            print("Epoch {}/{}".format(epoch+1, self.params['epochs']))
            metrics_log = ''
            for k in logs:
                val = logs[k]
                try:
                    val = float(val)
                    if abs(val) > 1e-3:
                        metrics_log += ' - %s: %.4f' % (k, val)
                    else:
                        metrics_log += ' - %s: %.4e' % (k, val)
                except ValueError:
                    metrics_log += ' - %s: %s' % (k, val)
            Cost_time = (time.time() - self.epoch_start_time) * self.interval
            print('{}/{} .... - Epoch time: {:.2f} s {}'.format(epoch+1, self.params['epochs'], Cost_time, metrics_log))



class Dense:
    """
    Implements a fully connected layer (Dense layer) in a neural network.
    """
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, doutputs):
        dinputs = np.dot(doutputs, self.weights.T)
        dweights = np.dot(self.inputs.T, doutputs)
        dbiases = np.sum(doutputs, axis=0, keepdims=True)
        self.weights -= learning_rate * dweights
        self.biases -= learning_rate * dbiases
        return dinputs

class Adam:
    """
    Implements the Adam optimizer for training neural networks.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, weights, gradients):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class CustomDataGenerator(Sequence):
    def __init__(self, x_train, y_train, batch_size):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x_train) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_train[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y


class NorScaler():
    def __init__(self, feature):
        self.mean_value = feature.mean()
        self.std_value = feature.std()

    def normalize(self, value):
        normalized_value = (value - self.mean_value) / (self.std_value + 1e-9)
        return normalized_value

    def denormalize(self, normalized_value):
        value = (normalized_value * (self.std_value + 1e-9)) + self.mean_value
        return value


class MinMaxScaler():
    def __init__(self, feature):
        self.min_value = feature.min()
        self.max_value = feature.max()

    def normalize(self, value):
        normalized_value = (value - self.min_value) / (self.max_value - self.min_value + 1e-9)
        return normalized_value

    def denormalize(self, normalized_value):
        value = (normalized_value * (self.max_value - self.min_value + 1e-9)) + self.min_value
        return value


class OneHotEncoder:
    def __init__(self):
        self.mapping = OrderedDict()

    def fit_transform(self, feature):
        unique_counts = {}
        feature_value_count = {}
        col = feature.columns.to_list()[0]
        unique_counts[col] = feature[col].nunique()
        value_counts = feature[col].value_counts()
        feature_value_count[col] = {value: count for value, count in zip(value_counts.index, value_counts.values)}

        processed_feature = pd.get_dummies(feature, columns=feature.columns, prefix_sep='#', dummy_na=True,
                                           drop_first=True)
        processed_feature = processed_feature.astype(int)

        self.decode = processed_feature.idxmax(axis=1)
        i = 0
        for column in processed_feature.columns:
            category, value = column.split('#')
            if category not in self.mapping:
                i = 0
                self.mapping[category] = OrderedDict()
            self.mapping[category][value] = {"OHE_dimension": unique_counts[category], "OHE_index": i}
            i += 1

        delete_key = list(set(str(x) for x in feature_value_count[col]) - set(self.mapping[col]))[0]
        self.mapping[col][delete_key] = {"OHE_dimension": unique_counts[col], "OHE_index": -1}

        return processed_feature, self.mapping, col

    def transform(self, feature):
        col = feature.columns.to_list()[0]
        onehot_config = self.mapping[col]
        result_list = []
        dimension = list(onehot_config.values())[0]["OHE_dimension"]
        columns = [f"{col}#"] * dimension
        onehot_vector = np.zeros(dimension)

        for feature_value, params in onehot_config.items():
            onehot_index = params["OHE_index"]
            if onehot_index != -1:
                columns[onehot_index] += feature_value

        for index, row in feature.iterrows():
            value_config = onehot_config.get(str(row[feature.columns[0]]), onehot_config["nan"])
            onehot_index = value_config["OHE_index"]
            if onehot_index != -1:
                onehot_vector[onehot_index] = 1
            result_list.append(list(onehot_vector))

        result_df = pd.DataFrame(result_list, columns=columns)
        return result_df

class LabelEncode:
    def __init__(self):
        self.mapping = OrderedDict()

    def fit_transform(self, feature):
        label_encoder = LabelEncoder()
        scaler = StandardScaler()
        feature = feature.astype(str)
        for col in feature.columns:
            categories = list(str(x) for x in feature[col].unique())
            categories.append('unknown')
            # Fit the encoder with the updated list of categories
            label_encoder.fit(categories)

            # Transform the feature column using the fitted encoder
            mask = feature[col].notnull()
            feature.loc[mask, col] = label_encoder.transform(feature.loc[mask, col])
            # feature[col] = label_encoder.fit_transform(feature[col])
            # Create a StandardScaler object
            # Normalize the encoded features
            feature = pd.DataFrame(scaler.fit_transform(feature), columns=[col])
            scaler_mean = scaler.mean_[0]
            scaler_std = scaler.var_[0] if scaler.var_[0] != 0 else 1e-9
            assert scaler_std != 0, f"{col}is the same"
            self.mapping[col] = dict(zip(label_encoder.classes_.tolist(), (
                        (label_encoder.transform(label_encoder.classes_) - scaler_mean) / scaler_std).tolist()))
        return feature, self.mapping

    def transform(self, feature):
        col = feature.columns.to_list()[0]
        le_config = self.mapping[col]
        processed_feature = feature.copy()
        for index, row in feature.iterrows():
            value_mapping = le_config.get(str(row[feature.columns[0]]), le_config["unknown"])
            processed_feature.iloc[index, :] = value_mapping
        return pd.DataFrame(processed_feature, columns=[col])

class TextTokenizer:
    def __init__(self, feature, padding_value = 0):
        self.max_chars_per_feature = feature.apply(lambda x: max([len(str(i)) for i in x])).max()
        self.tokenizer = Tokenizer()
        self.padding_value = padding_value
        self.feature_name = feature.columns.to_list()[0]
        self.columns = [self.feature_name + f"_{i}" for i in range(self.max_chars_per_feature)]

    def fit_transform(self, feature):

        char_var_train = []
        for i in range(feature.shape[0]):
            if len(feature.iloc[i, 0]) == 0:
                char_var_train.append([' '])
            else:
                char_var_train.append(list(feature.iloc[i, 0]))

        self.tokenizer.fit_on_texts(char_var_train)

        feature = self.tokenizer.texts_to_sequences(char_var_train)
        feature = pad_sequences(feature, padding='post', maxlen=self.max_chars_per_feature, value=self.padding_value)

        feature = pd.DataFrame(feature)
        feature.columns = self.columns
        # nan_rows = feature.isnull().any(axis=1)
        # nan_indices = nan_rows[nan_rows].index.tolist()
        # nan_features = [f'feature_{i}' for i in nan_indices]

        return feature

    def transform(self, feature):
        char_var_train = []
        for i in range(feature.shape[0]):
            if len(str(feature.iloc[i, 0])) == 0:
                char_var_train.append([' '])
            else:
                char_var_train.append(list(str(feature.iloc[i, 0])))
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(char_var_train)

        feature = self.tokenizer.texts_to_sequences(char_var_train)
        feature = pad_sequences(feature, padding='post', truncating='post',  maxlen=self.max_chars_per_feature, value = self.padding_value)
        feature = pd.DataFrame(feature)
        nan_rows = feature.isnull().any(axis=1)
        nan_indices = nan_rows[nan_rows].index.tolist()
        nan_features = [f'feature_{i}' for i in nan_indices]
        feature.columns = self.columns

        return feature


# class Self_Attention(Layer):
#     """
#     Attention layer for RNN models
#     """
#
#     def __init__(self, output_dim, return_attention=False, **kwargs):
#         self.supports_masking = True
#         self.return_attention = return_attention
#         self.output_dim = output_dim
#         super(Self_Attention, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # Create a trainable weight for this layer.
#         # input_shape is expected to be (batch_size, time_steps, input_dim)
#         self.WQ = self.add_weight(name='WQ',
#                                   shape=(input_shape[-1], self.output_dim),
#                                   initializer='uniform',
#                                   trainable=True)
#         self.WK = self.add_weight(name='WK',
#                                   shape=(input_shape[-1], self.output_dim),
#                                   initializer='uniform',
#                                   trainable=True)
#         self.WV = self.add_weight(name='WV',
#                                   shape=(input_shape[-1], self.output_dim),
#                                   initializer='uniform',
#                                   trainable=True)
#         super(Self_Attention, self).build(input_shape)  # Be sure to call this at the end
#
#     def call(self, x, mask=None):
#         Q = K.dot(x, self.WQ)
#         K_mat = K.dot(x, self.WK)
#         V = K.dot(x, self.WV)
#
#         attention_scores = K.batch_dot(Q, K.permute_dimensions(K_mat, [0, 2, 1]))
#         attention_scores = attention_scores / (self.output_dim ** 0.5)
#
#         if mask is not None:
#             mask = K.cast(mask[..., None], K.floatx())
#             attention_scores += -1e9 * (1. - mask)
#
#         attention_scores = K.softmax(attention_scores)
#         attention_output = K.batch_dot(attention_scores, V)
#
#         if self.return_attention:
#             return [attention_output, attention_scores]
#         return attention_output
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[1], self.output_dim
#
#     def get_config(self):
#         config = super(Self_Attention, self).get_config()
#         config.update({
#             'output_dim': self.output_dim,
#             'return_attention': self.return_attention
#         })
#         return config


class Self_Attention(Layer):
    """
    Attention layer for RNN models
    """

    def __init__(self, output_dim, return_attention = False,**kwargs):
        self.supports_masking = True
        self.return_attention = return_attention
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight for this layer
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Self_Attention, self).build(input_shape)  # Be sure to call it at the end

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask = None):
        print("x.shape", x.shape)

        # if mask is not None:
        #     mask = K.cast(mask[..., None], K.floatx())
        #     print("mask.shape", mask.shape)
        #     x *= mask
        print("self.kernel[0].shape", self.kernel[0].shape)

        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        # self.kernel("WQ.shape", WQ.shape)
        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)
        print("WV.shape", WV.shape)
        # attention score
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        print("QK.shape", QK.shape)
        QK = QK / (64 ** 0.5)

        if mask is not None:
            mask = K.cast(mask[..., None], K.floatx())
            print("mask.shape", mask.shape)
            QK *= mask
        QK = K.softmax(QK)


        V = K.batch_dot(QK, WV)
        print("V.shape", V.shape)
        if self.return_attention:
            return [V,]
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)
    def get_config(self):
        config = super(Self_Attention, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'return_attention': self.return_attention
        })
        return config

class MultiHeadAtten(Layer):
    """
    Attention layer for RNN models
    """

    def __init__(self, output_dim, nheads, return_attention = False, trainable=True, **kwargs):
        self.supports_masking = True
        self.return_attention = return_attention
        self.output_dim = output_dim
        self.hidden_size = output_dim
        self._n_heads = nheads
        self.trainable = trainable
        # self.name = name
        super(MultiHeadAtten, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight for this layer
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, self._n_heads, input_shape[2], self.hidden_size),
                                      initializer='he_normal',
                                      trainable=self.trainable)
        self.WO = self.add_weight(name='output',
                                      shape=(1, self._n_heads*self.hidden_size,input_shape[2]),
                                      initializer='he_normal',
                                      trainable=self.trainable)
        print("self.kernel.shape", self.kernel.shape)
        # print("self.WO.shape", self.WO.shape)
        super(MultiHeadAtten, self).build(input_shape)  # Be sure to call it at the end

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def split_heads(self, x):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        _, a, _ = x.shape
        x = K.reshape(x, (-1, a, self._n_heads, self.output_dim))

        return x

    def softmax(self, x):
        e_x = tf.exp(x - tf.reduce_max(x, axis=-1, keepdims=True))
        return e_x / tf.reduce_sum(e_x, axis=-1, keepdims=True)

    def call(self, x, mask = None):
        print("x.shape", x.shape)
        # # x = tf.keras.backend.expand_dims(x, axis=3)
        # print("x.expand", x.shape)


        # X = [batch_size, 1, sequence_length], weight = # [batch_size,sequence_length,self.multiheads*self.seq]
        print(self.kernel[0].shape)
        Qi = K.dot(x, self.kernel[0])
        # tf.print(Qi[0,:0,:,:])
        # print(Qi[0,:0,:,:].numpy())
        print("1",self.kernel[0].shape)

        print("qi shape", Qi.shape)
        Ki = K.dot(x, self.kernel[1])
        # Ki = Qi
        Vi = K.dot(x, self.kernel[2])


        or_QKo = tf.matmul(K.permute_dimensions(Qi, [0, 2, 1, 3]), K.permute_dimensions(Ki, [0, 2, 3, 1])) # [batch_size,  HEADS,  q_seq, k_seq]
        # [batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        # QK = K.permute_dimensions(QK, pattern=[0, 3, 2, 1]) #[batch_size, HEADS, k_seq, q_seq, ]

        print("tf QK.shape", or_QKo.shape)
        # print("QK", K.eval(QK))
        QKo1 = or_QKo / self.output_dim / self.hidden_size / self._n_heads

        QKo = self.softmax(QKo1)
        # QKo = K.softmax(QKo1, axis=3)

        # print("QK.shape", QKo.shape)
        # print("Vi.shape", Vi.shape)
        # vi (None, 1, 3, 128)
        Vi = K.permute_dimensions(Vi, [0, 2, 1, 3])
        print("1", self.kernel[0].shape)
        # print("PER Vi.shape", Vi.shape)
        out = tf.matmul(QKo, Vi)
        print("out.shape", out.shape)
        out = K.permute_dimensions(out, [0, 2, 3, 1])
        # print("out permute_dimensions shape", out.shape)
        # out = K.squeeze(out, axis=3)
        print("self.WO.shape", self.WO.shape)
        print("out.shape", out.shape)
        out = K.reshape(out, (-1, out.shape[1], self._n_heads*self.hidden_size))
        print("out.shape1", out.shape)
        out = tf.matmul(out, self.WO)
        print("1", self.kernel[0].shape)
        print("final.shape1", out.shape)
        # out = K.squeeze(out, axis=3)
        # out = K.permute_dimensions(out, [0, 2, 1])
        # print("final.shape2", out.shape)
        if self.return_attention:
            return [out,]
        return out, QKo, or_QKo, Qi, x, self.kernel[0]

    def compute_output_shape(self, input_shape):
        # print("output shape", (input_shape[0], input_shape[1], self.output_dim))
        return (input_shape[0], input_shape[1], self.output_dim)
    def get_config(self):
        config = super(MultiHeadAtten, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'return_attention': self.return_attention,
            "nheads": self._n_heads,
            "trainable": self.trainable,
            # "name": self.name
        })
        return config

# class DynamicEmbeddingLayer(Layer):
#     def __init__(self, input_dim, output_dim):
#         super(DynamicEmbeddingLayer, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#     def build(self, input_shape):
#         self.embeddings = self.add_weight(shape=(self.input_dim, self.output_dim),
#                                          initializer='random_normal',
#                                          trainable=True)
#
#     def call(self, inputs):
#         return tf.nn.embedding_lookup(self.embeddings, inputs)
#return tf.matmul(inputs, self.kernel)
#
class oldMultiHeadAtten(Layer):

    def __init__(self, output_dim, nheads, return_attention = False, trainable=True, **kwargs):
        self.supports_masking = True
        self.return_attention = return_attention
        self.output_dim = output_dim
        self.hidden_size = output_dim
        self._n_heads = nheads
        self.trainable = trainable
        # self.name = name
        super(oldMultiHeadAtten, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight for this layer
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, self._n_heads, input_shape[2], self.hidden_size),
                                      initializer='glorot_uniform',
                                      trainable=self.trainable)
        self.WO = self.add_weight(name='output',
                                      shape=(1, 1, self._n_heads,1),
                                      initializer='glorot_uniform',
                                      trainable=self.trainable)
        # print("self.kernel.shape", self.kernel.shape)
        # print("self.WO.shape", self.WO.shape)
        super(MultiHeadAtten, self).build(input_shape)  # Be sure to call it at the end

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def split_heads(self, x):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        _, a, _ = x.shape
        x = K.reshape(x, (-1, a, self._n_heads, self.output_dim))

        return x

    def call(self, x, mask = None):
        print("x.shape", x.shape)
        # x = tf.keras.backend.expand_dims(x, axis=3)
        print("x.expand", x.shape)

        # if mask is not None:
        #     mask = K.cast(mask[..., None], K.floatx())
        #     print("mask.shape", mask.shape)
        #     x *= mask
        # WO = self.split_heads(self.kernel[4])
        # WO = K.permute_dimensions(WO, [])
        # WK = self.split_heads(self.kernel[1])
        # WV = self.split_heads(self.kernel[2])
        # print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)
        # print("WV.shape", WV.shape)
        # print("WQ.shape", WQ.shape)


        # X = [batch_size, 1, sequence_length], weight = # [batch_size,sequence_length,self.multiheads*self.seq]
        print(self.kernel[0].shape)
        # print("split qi shape", Qi.shape)
        Qi = K.dot(x, self.kernel[0])
        print("qi shape", Qi.shape)
        Ki = K.dot(x, self.kernel[1])
        Vi = K.dot(x, self.kernel[2])

        # print("split qi shape", Qi.shape)
        # # [batch_size, 1, HEADS,  seq]
        #
        # # attention score
        #
        # print("K.permute_dimensions(Qi, [0, 2, 3, 1])", K.permute_dimensions(Qi, [0, 2, 1, 3])) # [batch_size,  HEADS,  q_seq, 1]
        # print("K.permute_dimensions(Ki, [0, 2, 1, 3])", K.permute_dimensions(Ki, [0, 2, 3, 1])) #  # [batch_size,  HEADS,  1, k_seq, ]
        # QK = K.batch_dot(K.permute_dimensions(Qi, [0, 2, 1, 3]), K.permute_dimensions(Ki, [0, 2, 1, 3]), axes=[3, 3])
        # print("QK.shape", QK.shape)

        QKo = tf.matmul(K.permute_dimensions(Qi, [0, 2, 1, 3]), K.permute_dimensions(Ki, [0, 2, 3, 1])) # [batch_size,  HEADS,  q_seq, k_seq]
        # [batch_size,self.multiheads,Q_sequence_length,K_sequence_length]

        # print("tf QK.shape", QKo.shape)
        QK = QKo / (self.output_dim ** 0.5)

        # if mask is not None:
        #     mask = K.cast(mask[..., None], K.floatx())
        #     print("mask.shape", mask.shape)
        #     QK *= mask
        QK = K.softmax(QK, axis=3)

        # print("QK.shape", QK.shape)
        # print("Vi.shape", Vi.shape)
        # vi (None, 1, 3, 128)
        Vi = K.permute_dimensions(Vi, [0, 2, 1, 3])
        # print("PER Vi.shape", Vi.shape)
        out = tf.matmul(QK, Vi)
        # print("out.shape", out.shape)
        out = K.permute_dimensions(out, [0, 2, 3, 1])
        # print("out permute_dimensions shape", out.shape)
        # out = K.squeeze(out, axis=3)
        out = tf.matmul(out, self.WO)
        print("final.shape1", out.shape)
        out = K.squeeze(out, axis=3)
        # out = K.permute_dimensions(out, [0, 2, 1])
        # print("final.shape2", out.shape)
        if self.return_attention:
            return [out,]
        return out, QK,QKo, Qi, x, self.kernel[0]

    def compute_output_shape(self, input_shape):
        # print("output shape", (input_shape[0], input_shape[1], self.output_dim))
        return (input_shape[0], input_shape[1], self.output_dim)
    def get_config(self):
        config = super(MultiHeadAtten, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'return_attention': self.return_attention,
            "nheads": self._n_heads,
            "trainable": self.trainable
            # "name": self.name
        })
        return config

