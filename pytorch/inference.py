import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from model.unet import UNet
from dataset import get_datasets
from tools import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, test_dataset = get_datasets()

model = UNet(num_classes=10).to(device)
model.eval() 

infer_path = os.path.abspath("./ckpt")
checkpoint = load_checkpoint(infer_path,device)

model.load_state_dict(checkpoint["model_state_dict"])

images = np.array([test_dataset[i]["image"].numpy() for i in range(25)])
images_tensor = torch.tensor(images, dtype=torch.float32, device=device)

with torch.no_grad():
    logits = model(images_tensor)
    pred = torch.argmax(logits, dim=1).cpu().numpy()

fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    img = images[i].transpose(1, 2, 0)
    ax.imshow(img)
    ax.set_title(f"label={pred[i]}")
    ax.axis('off')

save_path = "prediction_grid.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Prediction grid saved to {save_path}")
