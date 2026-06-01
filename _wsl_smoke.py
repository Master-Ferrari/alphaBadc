"""Временный смоук-тест WSL+GPU+Keras3 для новой 43-канальной версии."""
import numpy as np
import tensorflow as tf

print("TF", tf.__version__)
print("Keras", tf.keras.__version__)
print("GPU", tf.config.list_physical_devices("GPU"))

from alpha_badc.config import GameConfig, AZConfig
from alpha_badc.network import AZNetwork
from alpha_badc.encoding import action_size, num_channels
from alpha_badc.selfplay import play_game

print("channels", num_channels())
gc = GameConfig()
ac = AZConfig()
ac.num_simulations = 8
net = AZNetwork((gc.rows, gc.cols, num_channels()), action_size(gc), ac)

ex, st = play_game(net, gc, ac)
print("selfplay OK: turns", st.turns, "state", ex[0][0].shape, "gap", round(st.mean_policy_gap, 3))

states = np.stack([e[0] for e in ex[:8]]).astype("float32")
pis = np.stack([e[1] for e in ex[:8]]).astype("float32")
zs = np.array([e[2] for e in ex[:8]], dtype="float32")
m = net.train_on_batch(states, pis, zs)
print("train_on_batch OK:", {k: round(v, 3) for k, v in m.items()})

net.save("/tmp/_smoke.weights.h5")
net2 = AZNetwork((gc.rows, gc.cols, num_channels()), action_size(gc), ac)
net2.load("/tmp/_smoke.weights.h5")
print("save/load OK")
print("ALL_OK")
