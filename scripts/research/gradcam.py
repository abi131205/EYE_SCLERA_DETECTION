import tensorflow as tf

MODEL_PATH = "models/experiments/ssl_finetuned_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

mobilenet = model.layers[0]

print("Model Loaded")

print("\nTarget Layer:")
print(mobilenet.get_layer("out_relu"))

print("\nInput Shape:")
print(model.input_shape)

print("\nOutput Shape:")
print(model.output_shape)