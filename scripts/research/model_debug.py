import tensorflow as tf

model = tf.keras.models.load_model(
    "models/experiments/ssl_finetuned_model.h5"
)

model.summary()

print("\n\n")

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.__class__.__name__)