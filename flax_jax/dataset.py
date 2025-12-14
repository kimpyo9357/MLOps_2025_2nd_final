import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset


def get_datasets():
    """MNIST 데이터셋 로드 및 전처리

    datasets 라이브러리의 최신 버전에서는 직접 인덱싱이 제대로 동작하지 않으므로
    먼저 numpy 배열로 변환한 후 JAX 배열로 변환합니다.
    """
    datasets = load_dataset("cifar10",cache_dir='../dataset')

    # numpy 배열로 먼저 변환 후 처리
    train_images = np.array(datasets["train"]["img"])
    train_labels = np.array(datasets["train"]["label"])
    test_images = np.array(datasets["test"]["img"])
    test_labels = np.array(datasets["test"]["label"])

    # 채널 차원 추가 및 정규화 후 JAX 배열로 변환
    datasets = {
        "train": {
            "image": jnp.array(train_images.astype(np.float32) / 255),
            "label": jnp.array(train_labels),
        },
        "test": {
            "image": jnp.array(test_images.astype(np.float32) / 255),
            "label": jnp.array(test_labels),
        },
    }
    return datasets['train'], datasets['test']