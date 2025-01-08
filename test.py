from PIL import Image
from utils.general_utils import PILtoTorch
import torchvision
import torch
import numpy as np

image_path = "/workspace/data/replica_sclike_colmap_dnsplatter/dtu_dataset/dtu/scan24/images/0020.png"

image = Image.open(image_path)

image = image.resize((640, 480))

resized_image_rgb = PILtoTorch(image)

print(resized_image_rgb.shape)

resized_image_rgb = resized_image_rgb[:3, :, :]

torchvision.utils.save_image(resized_image_rgb, "test.png")

resized_image_rgb = torch.cat([PILtoTorch(im) for im in image.split()[:3]], dim=0)

print(resized_image_rgb.shape)

torchvision.utils.save_image(resized_image_rgb, "test2.png")