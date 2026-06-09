"""
compare_models.py

Comparison of baseline and SSL models.
"""

BASELINE_MODEL = "models/baseline/mobilenet_jaundice_model.h5"
SSL_MODEL = "models/experiments/ssl_finetuned_model.h5"

baseline_accuracy = 67.03
ssl_accuracy = 79.84

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"Baseline Model : {BASELINE_MODEL}")
print(f"SSL Model      : {SSL_MODEL}")

print("\nResults")

print(f"Baseline Accuracy : {baseline_accuracy}%")
print(f"SSL Accuracy      : {ssl_accuracy}%")

improvement = ssl_accuracy - baseline_accuracy

print(f"\nImprovement : +{improvement:.2f}%")