import joblib
import tensorflow as tf
from tensorflow import lite

# Load scikit-learn models
decision_tree_model = joblib.load('DecisionTree.pkl')
nb_classifier_model = joblib.load('NBClassifier.pkl')
random_forest_model = joblib.load('RandomForest.pkl')
svm_classifier_model = joblib.load('SVMClassifier.pkl')

# Load TensorFlow model
with tf.keras.utils.custom_object_scope({}):  # Add custom objects if needed
    xgboost_model = tf.keras.models.load_model('XGBoost.pkl')

# Convert scikit-learn models to TensorFlow format
decision_tree_tf = tf.function(decision_tree_model.predict)
nb_classifier_tf = tf.function(nb_classifier_model.predict)
random_forest_tf = tf.function(random_forest_model.predict)
svm_classifier_tf = tf.function(svm_classifier_model.predict)

# Convert TensorFlow model to TensorFlow Lite
converter = lite.TFLiteConverter.from_keras_model(xgboost_model)
xgboost_tflite_model = converter.convert()

# Save TensorFlow Lite models
with open('DecisionTree.tflite', 'wb') as f:
    f.write(tf.compat.v1.make_tensor_proto(decision_tree_tf).SerializeToString())

with open('NBClassifier.tflite', 'wb') as f:
    f.write(tf.compat.v1.make_tensor_proto(nb_classifier_tf).SerializeToString())

with open('RandomForest.tflite', 'wb') as f:
    f.write(tf.compat.v1.make_tensor_proto(random_forest_tf).SerializeToString())

with open('SVMClassifier.tflite', 'wb') as f:
    f.write(tf.compat.v1.make_tensor_proto(svm_classifier_tf).SerializeToString())

with open('XGBoost.tflite', 'wb') as f:
    f.write(xgboost_tflite_model)
