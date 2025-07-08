import tensorflow as tf
import numpy as np
import pandas as pd

class GAIN:
    def __init__(self, data_dim, hint_rate=0.9, alpha=100, batch_size=128, epochs=1000):
        self.data_dim = data_dim
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs

    def _build_generator(self, X, M):
        inputs = tf.keras.layers.Concatenate(axis=1)([X, M])
        h1 = tf.keras.layers.Dense(self.data_dim * 2, activation='relu')(inputs)
        h2 = tf.keras.layers.Dense(self.data_dim * 2, activation='relu')(h1)
        output = tf.keras.layers.Dense(self.data_dim, activation='sigmoid')(h2)
        return output

    def _build_discriminator(self, X_hat, H):
        inputs = tf.keras.layers.Concatenate(axis=1)([X_hat, H])
        h1 = tf.keras.layers.Dense(self.data_dim * 2, activation='relu')(inputs)
        h2 = tf.keras.layers.Dense(self.data_dim * 2, activation='relu')(h1)
        output = tf.keras.layers.Dense(self.data_dim, activation='sigmoid')(h2)
        return output

    def fit_transform(self, data_x):
        data_x = data_x.copy().astype(np.float32)

        # Handle all-NaN columns
        valid_cols = ~np.all(np.isnan(data_x), axis=0)
        data_x = data_x[:, valid_cols]

        # Normalization
        min_val = np.nanmin(data_x, axis=0)
        max_val = np.nanmax(data_x, axis=0)
        norm_data = (data_x - min_val) / (max_val - min_val + 1e-6)
        norm_data[np.isnan(norm_data)] = 0

        mask = 1 - np.isnan(data_x).astype(np.float32)

        # Build models
        X_in = tf.keras.Input(shape=(norm_data.shape[1],))
        M_in = tf.keras.Input(shape=(norm_data.shape[1],))
        G_sample = self._build_generator(X_in, M_in)
        generator = tf.keras.Model([X_in, M_in], G_sample)

        X_hat_in = tf.keras.Input(shape=(norm_data.shape[1],))
        H_in = tf.keras.Input(shape=(norm_data.shape[1],))
        D_prob = self._build_discriminator(X_hat_in, H_in)
        discriminator = tf.keras.Model([X_hat_in, H_in], D_prob)

        d_optimizer = tf.keras.optimizers.Adam()
        g_optimizer = tf.keras.optimizers.Adam()

        @tf.function
        def train_step(x_mb, m_mb, h_mb):
            with tf.GradientTape() as tape_d:
                g_sample_mb = generator([x_mb, m_mb])
                x_hat_mb = x_mb * m_mb + g_sample_mb * (1 - m_mb)
                d_prob_mb = discriminator([x_hat_mb, h_mb])
                d_loss = -tf.reduce_mean(
                    m_mb * tf.math.log(d_prob_mb + 1e-8) +
                    (1 - m_mb) * tf.math.log(1. - d_prob_mb + 1e-8)
                )
            grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))

            with tf.GradientTape() as tape_g:
                g_sample_mb = generator([x_mb, m_mb])
                x_hat_mb = x_mb * m_mb + g_sample_mb * (1 - m_mb)
                d_prob_mb = discriminator([x_hat_mb, h_mb])
                mse_loss = tf.reduce_mean((m_mb * x_mb - m_mb * g_sample_mb) ** 2) / tf.reduce_mean(m_mb)
                g_loss = -tf.reduce_mean((1 - m_mb) * tf.math.log(d_prob_mb + 1e-8)) + self.alpha * mse_loss
            grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))

            return d_loss, g_loss

        # Training loop
        n_batches = int(np.ceil(norm_data.shape[0] / self.batch_size))
        for epoch in range(self.epochs):
            for _ in range(n_batches):
                batch_idx = np.random.choice(norm_data.shape[0], self.batch_size, replace=True)
                x_mb = norm_data[batch_idx, :]
                m_mb = mask[batch_idx, :]
                z_mb = np.random.uniform(0, 0.01, size=x_mb.shape).astype(np.float32)
                x_mb = m_mb * x_mb + (1 - m_mb) * z_mb
                h_mb = (np.random.binomial(1, self.hint_rate, size=m_mb.shape) * m_mb).astype(np.float32)

                x_mb = tf.convert_to_tensor(x_mb, dtype=tf.float32)
                m_mb = tf.convert_to_tensor(m_mb, dtype=tf.float32)
                h_mb = tf.convert_to_tensor(h_mb, dtype=tf.float32)

                d_loss, g_loss = train_step(x_mb, m_mb, h_mb)

            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}, D Loss: {d_loss.numpy():.4f}, G Loss: {g_loss.numpy():.4f}')

        # Final imputation
        g_sample = generator.predict([norm_data, mask])
        imputed_data = mask * norm_data + (1 - mask) * g_sample
        imputed_data = imputed_data * (max_val - min_val + 1e-6) + min_val

        # Rebuild full matrix with original columns
        full_imputed = np.zeros((data_x.shape[0], self.data_dim), dtype=np.float32)
        full_imputed[:, valid_cols] = imputed_data

        return full_imputed