import pickle
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

# Step 1: Load the Pickle Model
with open('SVMClassifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Step 2: Prepare the Model (if necessary)
# Example: Convert scikit-learn model to TensorFlow model
if isinstance(model, LogisticRegression):
    # Create a new TensorFlow model using tf.keras API
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(model.coef_.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False)
    ])
    
    # Copy weights from scikit-learn model to TensorFlow model
    tf_model.layers[1].set_weights([model.coef_.T])

# Step 3: Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
