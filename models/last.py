import torch
import onnx
import onnx_tf
from onnx_tf.backend import prepare
import tensorflow as tf
from tensorflow import lite

# Step 1: Load the PyTorch model
model = torch.load('plant_disease_model.pth')
model.eval()

# Step 2: Export the model to ONNX
input_tensor = torch.randn(1, *input_shape)  # Input shape of your model
torch.onnx.export(model, input_tensor, 'model.onnx')

# Step 3: Convert ONNX to TensorFlow
onnx_model = onnx.load('model.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('model.pb')

# Step 4: Optimize the TensorFlow model (optional)
# Perform optimization techniques as required for your deployment scenario

# Step 5: Convert to TensorFlow Lite
converter = lite.TFLiteConverter.from_saved_model('model.pb')
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
