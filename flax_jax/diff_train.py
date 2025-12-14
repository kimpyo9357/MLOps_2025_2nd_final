import os
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from model.unet import UNet
from model.diffusion.gaussian_diffusion import GaussianDiffusion
from tools import train_loop, eval_loop
from dataset import get_datasets


train_ds, test_ds = get_datasets()

rng = jax.random.PRNGKey(0)
model = UNet(num_classes=10)

T = 1000
betas = jnp.linspace(1e-4, 0.02, T, dtype=jnp.float32)
model_mean_type = 'EPSILON'  # 실제 enum/class 정의 확인 필요
model_var_type = 'FIXED_SMALL'  # 실제 enum/class 정의 확인 필요
loss_type = 'MSE'  # L2 loss

# 3. GaussianDiffusion 생성
diffusion = GaussianDiffusion(
    betas=betas,
    model_mean_type=model_mean_type,
    model_var_type=model_var_type,
    loss_type=loss_type,
    rescale_timesteps=True,  # optional
)

rng, key = jax.random.split(rng)
variables = model.init(key, jnp.ones((1, 32, 32, 3)))
params = variables['params']

jax.tree_util.tree_map(jnp.shape, variables)

learning_rate = 0.001
tx = optax.adam(learning_rate=learning_rate)

state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
)

train_path = f"{os.path.abspath('.')}/ckpt/"

handler = ocp.PyTreeCheckpointHandler()
checkpointer = ocp.Checkpointer(handler)
options = ocp.CheckpointManagerOptions(
    create=not os.path.isdir(train_path),
    max_to_keep=3,
)

ckpt_mgr = ocp.CheckpointManager(
    train_path,
    checkpointer,
    options
)

train_epoch = 10
batch_size = 64
eval_batch_size = 100

for epoch in range(train_epoch):
    rng, key = jax.random.split(rng)
    state = train_loop(state, train_ds, batch_size, epoch, rng, diffusion)
    # eval_loop(state, test_ds, eval_batch_size)
    save_args = orbax_utils.save_args_from_target(state)
    ckpt_mgr.save(epoch, state, save_kwargs={"save_args": save_args})