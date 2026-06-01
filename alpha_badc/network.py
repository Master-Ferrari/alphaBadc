"""Нейросеть AlphaZero на TensorFlow/Keras.

Общий residual-ствол → две головы:
  * policy — логиты по всем действиям (num_cells*NUM_LETTERS);
  * value  — скаляр в [-1, 1] (tanh): ожидаемый исход с точки зрения ходящего.

Класс :class:`AZNetwork` инкапсулирует модель, предсказание (одиночное и
батчевое для MCTS), один шаг обучения и сохранение/загрузку весов.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import AZConfig


def _conv_block(x, channels: int, l2: float, name: str):
    x = layers.Conv2D(
        channels, 3, padding="same", use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2), name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def _residual_block(x, channels: int, l2: float, name: str):
    shortcut = x
    y = _conv_block(x, channels, l2, f"{name}_a")
    y = layers.Conv2D(
        channels, 3, padding="same", use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2), name=f"{name}_b_conv",
    )(y)
    y = layers.BatchNormalization(name=f"{name}_b_bn")(y)
    y = layers.Add(name=f"{name}_add")([shortcut, y])
    return layers.ReLU(name=f"{name}_out")(y)


def build_model(input_shape: Tuple[int, int, int], action_size: int, cfg: AZConfig) -> keras.Model:
    l2 = cfg.weight_decay
    inputs = keras.Input(shape=input_shape, name="state")
    x = _conv_block(inputs, cfg.channels, l2, "stem")
    for i in range(cfg.blocks):
        x = _residual_block(x, cfg.channels, l2, f"res{i}")

    # policy head
    p = _conv_block(x, 32, l2, "policy_head")
    p = layers.Flatten(name="policy_flatten")(p)
    policy = layers.Dense(
        action_size, kernel_regularizer=keras.regularizers.l2(l2), name="policy_logits"
    )(p)

    # value head
    v = _conv_block(x, 32, l2, "value_head")
    v = layers.Flatten(name="value_flatten")(v)
    v = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(l2), name="value_dense")(v)
    value = layers.Dense(1, activation="tanh", name="value")(v)

    return keras.Model(inputs=inputs, outputs=[policy, value], name="alpha_badc_net")


class AZNetwork:
    def __init__(self, input_shape: Tuple[int, int, int], action_size: int, config: AZConfig):
        self.config = config
        self.action_size = action_size
        self.model = build_model(input_shape, action_size, config)
        self.optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

    # ---- инференс ---------------------------------------------------------
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Одна позиция (rows, cols, C) -> (policy_probs[action_size], value)."""
        logits, value = self.predict_batch(state[None, ...])
        return logits[0], float(value[0])

    def predict_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Батч позиций -> (softmax-probs [N, action_size], values [N])."""
        logits, value = self._forward(tf.convert_to_tensor(states, dtype=tf.float32))
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        return probs, value.numpy().reshape(-1)

    @tf.function(reduce_retracing=True)
    def _forward(self, states):
        return self.model(states, training=False)

    # ---- обучение ---------------------------------------------------------
    @tf.function(reduce_retracing=True)
    def _train_step(self, states, target_pi, target_z):
        with tf.GradientTape() as tape:
            logits, value = self.model(states, training=True)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            policy_loss = -tf.reduce_mean(tf.reduce_sum(target_pi * log_probs, axis=-1))
            value_loss = tf.reduce_mean(tf.square(target_z - tf.squeeze(value, axis=-1)))
            reg_loss = tf.add_n(self.model.losses) if self.model.losses else 0.0
            total = policy_loss + self.config.value_loss_coef * value_loss + reg_loss
        grads = tape.gradient(total, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total, policy_loss, value_loss

    def train_on_batch(self, states, target_pi, target_z) -> Dict[str, float]:
        total, p_loss, v_loss = self._train_step(
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(target_pi, dtype=tf.float32),
            tf.convert_to_tensor(target_z, dtype=tf.float32),
        )
        return {
            "total_loss": float(total),
            "policy_loss": float(p_loss),
            "value_loss": float(v_loss),
        }

    # ---- сохранение -------------------------------------------------------
    def save(self, path: str) -> None:
        self.model.save_weights(path)

    def load(self, path: str) -> None:
        self.model.load_weights(path)

    def clone_weights_to(self, other: "AZNetwork") -> None:
        other.model.set_weights(self.model.get_weights())
