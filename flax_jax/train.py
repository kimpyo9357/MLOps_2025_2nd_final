import os
import jax
import jax.numpy as jnp
import time
import optax
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from model.unet import UNet
from tools import train_loop, eval_loop, get_vram_usage
from dataset import get_datasets
from logger import Logging

train_epoch = 10
learning_rate = 0.001
batch_size = 64
eval_batch_size = 100

logger = Logging('mlops_fin','jax/flax','unet',learning_rate,batch_size,train_epoch,'adam')

start = time.time()
first_time = start

train_ds, test_ds = get_datasets()

rng = jax.random.PRNGKey(0)
model = UNet(num_classes=10)

# T = 1000
# betas = jnp.linspace(1e-4, 0.02, T, dtype=jnp.float32)
# model_mean_type = 'EPSILON'  # 실제 enum/class 정의 확인 필요
# model_var_type = 'FIXED_SMALL'  # 실제 enum/class 정의 확인 필요
# loss_type = 'MSE'  # L2 loss

# diffusion = GaussianDiffusion(
#     betas=betas,
#     model_mean_type=model_mean_type,
#     model_var_type=model_var_type,
#     loss_type=loss_type,
#     rescale_timesteps=True,  # optional
# )

rng, key = jax.random.split(rng)
variables = model.init(key, jnp.ones((1, 32, 32, 3)))
params = variables['params']

jax.tree_util.tree_map(jnp.shape, variables)

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

mem = get_vram_usage()
logger.log('train',-1,0,0,time.time()-first_time,time.time()-first_time,mem)
logger.log('test',-1,0,0,time.time()-first_time,time.time()-first_time,mem)

for epoch in range(train_epoch):
    start = time.time()
    rng, key = jax.random.split(rng)
    state , metrics= train_loop(state, train_ds, batch_size, epoch, rng)
    mem = get_vram_usage()
    logger.log('train',epoch,metrics['loss'],metrics['accuracy'],time.time()-start,time.time()-first_time,mem)
    
    # state = train_loop(state, train_ds, batch_size, epoch, rng, diffusion)
    start = time.time()
    metrics= eval_loop(state, test_ds, eval_batch_size)
    mem = get_vram_usage()
    logger.log('test',epoch,metrics['loss'],metrics['accuracy'],time.time()-start,time.time()-first_time,mem)
    
    save_args = orbax_utils.save_args_from_target(state)
    ckpt_mgr.save(epoch, state, save_kwargs={"save_args": save_args})
