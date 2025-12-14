import jax
import jax.numpy as jnp

import optax
import subprocess

def compute_metrics(logits, labels):
    loss = jnp.mean(optax.softmax_cross_entropy(logits,
                    jax.nn.one_hot(labels, num_classes=10)))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }
    return metrics

# def train_step(state, batch, key, diffusion):
#     """
#     state: TrainState
#     batch: dict, {'image': (B,H,W,C)}
#     key: JAX random key
#     """
#     images = batch['image']

#     def loss_fn(params):
#         # timestep 샘플링
#         t = jax.random.randint(key, (images.shape[0],), 0, diffusion.num_timesteps)

#         # 노이즈 추가 (q_sample)
#         noisy_images, noise = diffusion.q_sample(images, t, key)  

#         # 모델 예측
#         pred_noise = state.apply_fn({'params': params}, noisy_images, t=t)

#         # MSE loss
#         loss = jnp.mean((pred_noise - noise) ** 2)
#         return loss, pred_noise

#     grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#     (loss, pred_noise), grads = grad_fn(state.params)
#     state = state.apply_gradients(grads=grads)

#     metrics = {'loss': loss}
#     return state, metrics


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = jnp.mean(optax.softmax_cross_entropy(
            logits=logits,
            labels=jax.nn.one_hot(batch['label'], num_classes=10))
        )
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)


    metrics = compute_metrics(logits, batch['label'])
    return state, metrics

def train_loop(state, train_ds, batch_size, epoch, rng):
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # Skip an incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        
        state, metrics = train_step(state, batch)
        # rng, subkey = jax.random.split(rng)
        # state, metrics = train_step(state, batch, subkey, diffusion)
        batch_metrics.append(metrics)


    training_batch_metrics = jax.device_get(batch_metrics)
    training_epoch_metrics = {
        k: sum([metrics[k] for metrics in training_batch_metrics])/steps_per_epoch
        for k in training_batch_metrics[0]}
    
    metrics['loss'].block_until_ready()
    metrics['accuracy'].block_until_ready()
    print('EPOCH: %d\nTraining loss: %.4f, accuracy: %.2f' % (epoch, training_epoch_metrics['loss'], training_epoch_metrics['accuracy'] * 100))
    return state, training_epoch_metrics

@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch['image'])
    return compute_metrics(logits, batch['label'])

def eval_loop(state, test_ds, batch_size):
    eval_ds_size = test_ds['image'].shape[0]
    steps_per_epoch = eval_ds_size // batch_size


    batch_metrics = []
    for i in range(steps_per_epoch):
        batch = {k: v[i*batch_size:(i+1)*batch_size, ...] for k, v in test_ds.items()}
        metrics = eval_step(state, batch)
        batch_metrics.append(metrics)


    eval_batch_metrics = jax.device_get(batch_metrics)
    eval_batch_metrics = {
        k: sum([metrics[k] for metrics in eval_batch_metrics])/steps_per_epoch
        for k in eval_batch_metrics[0]}
    print('    Eval loss: %.4f, accuracy: %.2f' % (eval_batch_metrics['loss'], eval_batch_metrics['accuracy'] * 100))
    metrics['loss'].block_until_ready()
    metrics['accuracy'].block_until_ready()
    
    return metrics

@jax.jit
def pred_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch)
    return logits.argmax(axis=1)

def get_vram_usage():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        used_memory = int(result.strip().split("\n")[0])
        return used_memory
    except Exception as e:
        print("nvidia-smi 호출 실패:", e)
        return None

