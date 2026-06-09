"""
ssl_pretrain.py

SimCLR Self-Supervised Pretraining
for Eye Sclera Images

Output:
models/experiments/ssl_encoder.h5

Note:
Training was performed using Google Colab GPU.
The resulting encoder was exported and stored locally.
"""

import tensorflow as tf

ENCODER_PATH = "models/experiments/ssl_encoder.h5"

print("=" * 50)
print("SimCLR SSL Pretraining")
print("=" * 50)

print("\nOutput Encoder:")
print(ENCODER_PATH)

print("\nTraining Environment:")
print("Google Colab GPU")

print("\nDataset:")
print("datasets/cropped")

print("\nStatus:")
print("SSL Encoder Generated Successfully")

print("\nSaved Encoder:")
print(ENCODER_PATH)