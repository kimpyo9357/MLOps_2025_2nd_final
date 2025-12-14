import os
import jax
import jax.numpy as jnp

import optax
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

from flax.training.train_state import TrainState

from model.unet import UNet
from dataset import get_datasets
from tools import pred_step

train_ds, test_ds = get_datasets()

rng = jax.random.PRNGKey(0)
model = UNet(num_classes=10)
rng, key = jax.random.split(rng)
variables = model.init(key, jnp.ones((1, 32, 32, 3)))
params = variables['params']

learning_rate = 0.001
tx = optax.adam(learning_rate=learning_rate)

state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
)

train_path = f"{os.path.abspath('.')}/ckpt/"
checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
options = ocp.CheckpointManagerOptions(
    create=False,
    max_to_keep=3,
)

ckpt_mgr = ocp.CheckpointManager(
    train_path,
    checkpointer,
    options
)

step = ckpt_mgr.latest_step()
if step is None:
    print("저장된 체크포인트가 없습니다.")
else:
    initial_state = state 

    state = ckpt_mgr.restore(
        step, 
        initial_state 
    )
    
    print(f"체크포인트 {step} (저장된 save_args 포함) 로드 완료.")
pred = pred_step(state, test_ds['image'][:25])

fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
   ax.imshow(test_ds['image'][i], cmap='gray')
   ax.set_title(f"label={pred[i]}")
   ax.axis('off')
   
save_path = "prediction_grid.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close(fig)