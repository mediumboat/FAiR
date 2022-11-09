import typing
from abc import ABC
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, BatchNormalization, Embedding
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, Reduction
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import L2
from tensorflow.keras.metrics import Mean


@register_keras_serializable(package='Custom', name='l2')
class L2Regularizer_to_one(tf.keras.regularizers.Regularizer):
    # Unlike traditional L2 regularization that forces variables to be 0, this forces variables to be 1.
    def __init__(self, l2=0.01):
        self.l2 = l2

    def __call__(self, x):
        return self.l2 * tf.math.reduce_sum(tf.math.square(x - 1))

    def get_config(self):
        return {'l2': float(self.l2)}


class Filter(Model, ABC):
    # In the released code, the FiLM is implemented in a slightly diffrent way from the paper to improve the efficiency, which does not significantly affect the main results/conclusions. 
    def __init__(self, flags):
        super(Filter, self).__init__()
        self.filter_units = flags.filter_units
        self.reg_term = flags.reg_term
        self.mlp_layers = []
        self.bn_layers = []
        self.film_alpha_layers = []
        self.film_beta_layers = []
        for i, unit in enumerate(self.filter_units):
            self.mlp_layers.append(
                Dense(unit, activation='relu', name='filter_{}'.format(i)))
            self.film_alpha_layers.append(
                Dense(unit, activation=None, name='alpha_{}'.format(i),
                      kernel_regularizer=L2Regularizer_to_one(self.reg_term), use_bias=False))
            self.film_beta_layers.append(
                Dense(unit, activation=None, name='beta_{}'.format(i),
                      kernel_regularizer=L2(self.reg_term), use_bias=False))
            self.bn_layers.append(BatchNormalization())

    def call(self, inputs, training):
        embedding = inputs
        for i, unit in enumerate(self.filter_units):
            filter_layer = self.mlp_layers[i]
            film_alpha_layer = self.film_alpha_layers[i]
            film_beta_layer = self.film_beta_layers[i]
            bn_layer = self.bn_layers[i]
            embedding = filter_layer(embedding)
            alpha = film_alpha_layer(inputs)
            beta = film_beta_layer(inputs)
            embedding = tf.multiply(embedding, alpha) + beta
            embedding = bn_layer(embedding, training=training)
            embedding = tf.nn.leaky_relu(embedding)
        return embedding


class Discriminator(Model, ABC):
    def __init__(self, flags, model_type):
        super(Discriminator, self).__init__()
        self.reg_term = flags.reg_term
        if model_type == "implicit":
            self.discriminator_units = flags.implicit_layer_units
        elif model_type == "explicit":
            self.discriminator_units = flags.explicit_layer_units
        else:
            raise Exception("Only implicit or explicit discriminator is allowed.")
        self.mlp_layers = []
        self.prediction_layer = Dense(1, activation='sigmoid')
        for i, unit in enumerate(self.discriminator_units):
            self.mlp_layers.append(
                Dense(unit, activation='relu', name='discriminator_{}'.format(i), kernel_regularizer=L2(self.reg_term)))

    def call(self, inputs, training=None):
        embedding = inputs
        for i, unit in enumerate(self.discriminator_units):
            discriminator_layer = self.mlp_layers[i]
            embedding = discriminator_layer(embedding)
        pred = self.prediction_layer(embedding)
        return tf.squeeze(pred)


class BaseRecModel(Model):
    def __init__(self, flags):
        super(BaseRecModel, self).__init__()
        self.reg_term = flags.reg_term
        self.base_units = flags.rec_layer_units
        self.mlp_layers = []
        self.prediction_layer = Dense(1, activation='relu')
        for i, unit in enumerate(self.base_units):
            self.mlp_layers.append(
                Dense(unit, activation='relu', name='rec_{}'.format(i), kernel_regularizer=L2(self.reg_term)))

    def call(self, inputs, training=None):
        embedding = inputs
        for i, unit in enumerate(self.base_units):
            rec_layer = self.mlp_layers[i]
            embedding = rec_layer(embedding)
        pred = self.prediction_layer(embedding)
        return tf.squeeze(pred)


class FAiR(Model):
    def __init__(self, flags, n_users, n_items, user_group, item_group, average_rating, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flags = flags
        self.d_step = flags.d_step
        self.g_step = flags.g_step
        self.reg_term = flags.reg_term
        self.n_users = n_users
        self.n_items = n_items
        self.n_user_samples = flags.n_user_samples
        self.n_item_samples = flags.n_item_samples
        self.l1 = flags.lambda_1
        self.l2 = flags.lambda_2
        self.l3 = flags.lambda_3
        self.user_group = tf.squeeze(tf.convert_to_tensor(user_group, dtype=tf.int32))
        self.item_group = tf.squeeze(tf.convert_to_tensor(item_group, dtype=tf.int32))
        self.average_rating = tf.squeeze(tf.convert_to_tensor(average_rating, dtype=tf.float32))
        self.user_embedding_layer = Embedding(input_dim=self.n_users, output_dim=flags.dimension,
                                              name='user_embedding', input_length=1, trainable=True
                                             )
        self.item_embedding_layer = Embedding(input_dim=self.n_items, output_dim=flags.dimension,
                                              name='item_embedding', input_length=1, trainable=True
                                              )
        self.user_filter = Filter(flags)
        self.item_filter = Filter(flags)
        self.user_explicit_discriminator = Discriminator(flags, "explicit")
        self.item_explicit_discriminator = Discriminator(flags, "explicit")
        self.implicit_discriminator = Discriminator(flags, "implicit")
        self.rec_model = BaseRecModel(flags)
        self.concat = Concatenate()
        self.pretrain_loss = MeanSquaredError()
        self.user_d_loss = BinaryCrossentropy()
        self.item_d_loss = BinaryCrossentropy()
        self.im_d_loss = BinaryCrossentropy()
        self.rec_loss = MeanSquaredError()
        self.pretrain_optimizer = Adam()
        self.d_optimizer = Adam()
        self.g_optimizer = Adam()

    @tf.function()
    def call(self, inputs, training):
        user, item = inputs
        user_emb = self.user_embedding_layer(user)
        item_emb = self.item_embedding_layer(item)
        user_emb = self.user_filter(user_emb, training)
        item_emb = self.item_filter(item_emb, training)
        joint_emb = self.concat([user_emb, item_emb])
        result = self.rec_model(joint_emb)
        return tf.squeeze(result)

    @tf.function()
    def pretrain(self, inputs):
        user, item, y_true = inputs
        with tf.GradientTape() as tape:
            user_emb = self.user_embedding_layer(user)
            item_emb = self.item_embedding_layer(item)
            out = tf.squeeze(tf.matmul(user_emb, item_emb, transpose_b=True))
            loss = self.pretrain_loss(y_true=y_true, y_pred=out)
        train_var = self.user_embedding_layer.trainable_weights + self.item_embedding_layer.trainable_weights
        grads = tape.gradient(loss, train_var)
        self.pretrain_optimizer.apply_gradients(zip(grads, train_var))

    @tf.function()
    def adv_train(self, inputs):
        user, item, y_true = inputs
        user_g = tf.random.uniform(shape=(tf.math.floordiv(self.n_users, 10),), minval=0, maxval=self.n_users,
                                                 dtype=tf.int32)
        item_g = tf.random.uniform(shape=(tf.math.floordiv(self.n_items, 10),), minval=0, maxval=self.n_items,
                                             dtype=tf.int32)
        user_group_true = tf.cast(tf.gather(self.user_group, user_g),tf.int32)
        item_group_true = tf.cast(tf.gather(self.item_group, item_g),tf.int32)
        user_emb = self.user_filter(self.user_embedding_layer(user_g), training=False)
        item_emb = self.item_filter(self.item_embedding_layer(item_g), training=False)

        sampled_user_ids = tf.random.uniform(shape=(self.n_user_samples,), minval=0, maxval=self.n_users,
                                             dtype=tf.int32)
        sampled_item_ids = tf.random.uniform(shape=(self.n_item_samples,), minval=0, maxval=self.n_items,
                                             dtype=tf.int32)
        sampled_users = tf.repeat(sampled_user_ids, [self.n_item_samples])
        sampled_items = tf.tile(sampled_item_ids, [self.n_user_samples])
        ru_vectors = self((sampled_users, sampled_items), training=False)
        ru_vectors = tf.reshape(ru_vectors, [self.n_user_samples, 1, self.n_item_samples])
        ru_labels = tf.zeros((self.n_user_samples,), dtype=tf.float32)
        ro_vectors = tf.reshape(tf.gather(self.average_rating, sampled_item_ids), [1, 1, self.n_item_samples])
        ro_labels = tf.ones((1,), dtype=tf.float32)
        for i in range(self.d_step):
            with tf.GradientTape() as tape:
                user_group_pred = self.user_explicit_discriminator(user_emb)
                item_group_pred = self.item_explicit_discriminator(item_emb)
                ru_pred = self.implicit_discriminator(ru_vectors)
                ro_pred = self.implicit_discriminator(ro_vectors)
                item_d_loss = self.item_d_loss(y_true=item_group_true, y_pred=item_group_pred)
                user_d_loss = self.user_d_loss(y_true=user_group_true, y_pred=user_group_pred)
                ru_loss = self.im_d_loss(y_pred=ru_pred, y_true=ru_labels)
                ro_loss = self.im_d_loss(y_pred=tf.expand_dims(ro_pred, -1), y_true=tf.expand_dims(ro_labels, -1))
                d_loss = item_d_loss + user_d_loss + ru_loss + ro_loss \
                         + tf.add_n(self.user_explicit_discriminator.losses) \
                         + tf.add_n(self.item_explicit_discriminator.losses)\
                         + tf.add_n(self.implicit_discriminator.losses)
            d_train_var = self.user_explicit_discriminator.trainable_weights + \
                          self.item_explicit_discriminator.trainable_weights + \
                          self.implicit_discriminator.weights
            d_grads = tape.gradient(d_loss, d_train_var)
            self.d_optimizer.apply_gradients(zip(d_grads, d_train_var))

        user_group_true = tf.abs(tf.gather(self.user_group, user) - 1)
        item_group_true = tf.abs(tf.gather(self.item_group, item) - 1)
        sampled_user_ids = tf.squeeze(user)
        n_sampled_user = tf.size(user)
        sampled_item_ids = tf.random.uniform(shape=(self.n_item_samples,), minval=0, maxval=self.n_items,
                                             dtype=tf.int32)
        sampled_users = tf.repeat(sampled_user_ids, [self.n_item_samples])
        sampled_items = tf.tile(sampled_item_ids, [n_sampled_user])
        ru_labels = tf.ones((n_sampled_user,), dtype=tf.float32)
        for i in range(self.g_step):
            with tf.GradientTape() as tape:
                user_emb = self.user_filter(self.user_embedding_layer(user), training=True)
                item_emb = self.item_filter(self.item_embedding_layer(item), training=True)
                rec_pred = self((user, item), training=True)
                ru_vectors = self((sampled_users, sampled_items), training=True)
                ru_vectors = tf.reshape(ru_vectors, [n_sampled_user, 1, self.n_item_samples])
                ru_pred = self.implicit_discriminator(ru_vectors)
                ru_g_loss = self.im_d_loss(y_true=ru_labels, y_pred=ru_pred)
                user_group_pred = self.user_explicit_discriminator(user_emb)
                item_group_pred = self.item_explicit_discriminator(item_emb)
                user_g_loss = self.user_d_loss(y_true=user_group_true, y_pred=user_group_pred)
                item_g_loss = self.item_d_loss(y_true=item_group_true, y_pred=item_group_pred)
                rec_loss = self.rec_loss(y_true=y_true, y_pred=rec_pred)
                g_loss = rec_loss + self.l1 * user_g_loss + self.l2 * item_g_loss + self.l3 * ru_g_loss
                g_loss += tf.add_n(self.user_filter.losses)
                g_loss += tf.add_n(self.item_filter.losses)
            g_train_var = self.user_filter.trainable_weights + \
                          self.item_filter.trainable_weights + \
                          self.rec_model.trainable_weights
            g_grads = tape.gradient(g_loss, g_train_var)
            self.g_optimizer.apply_gradients(zip(g_grads, g_train_var))
