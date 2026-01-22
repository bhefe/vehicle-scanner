#!/usr/bin/env python3
"""Debug script to inspect TFLite model output shape and values"""

import numpy as np
import tensorflow as tf
from pathlib import Path

# Load the TFLite model
model_path = 'assets/models/detectlatest.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=" * 60)
print("INPUT DETAILS:")
print("=" * 60)
for i, detail in enumerate(input_details):
    print(f"Input {i}:")
    print(f"  Name: {detail['name']}")
    print(f"  Shape: {detail['shape']}")
    print(f"  Type: {detail['dtype']}")
    print()

print("=" * 60)
print("OUTPUT DETAILS:")
print("=" * 60)
for i, detail in enumerate(output_details):
    print(f"Output {i}:")
    print(f"  Name: {detail['name']}")
    print(f"  Shape: {detail['shape']}")
    print(f"  Type: {detail['dtype']}")
    print()

# Create dummy input to test
input_shape = input_details[0]['shape']  # e.g., [1, 640, 640, 3]
print("=" * 60)
print("TESTING WITH DUMMY INPUT:")
print("=" * 60)
print(f"Input shape: {input_shape}")

# Create random dummy input in the right range
dummy_input = np.random.rand(*input_shape).astype(np.float32) * 255

# Set input and run inference
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

# Get outputs
print(f"\nOutput shapes and sample values:")
for i, detail in enumerate(output_details):
    output_data = interpreter.get_tensor(detail['index'])
    print(f"\nOutput {i} ({detail['name']}):")
    print(f"  Shape: {output_data.shape}")
    print(f"  Dtype: {output_data.dtype}")
    print(f"  Min: {output_data.min():.6f}, Max: {output_data.max():.6f}, Mean: {output_data.mean():.6f}")
    
    # Print first few values
    if output_data.size > 0:
        flat = output_data.flatten()[:20]
        print(f"  First 20 values: {flat}")
