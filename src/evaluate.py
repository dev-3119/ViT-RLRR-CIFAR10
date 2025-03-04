import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

model = load_model("vit_finetuned_rlrr_model.h5")

from tensorflow.keras.datasets import cifar10
(_, _), (x_test, y_test) = cifar10.load_data()
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

def preprocess_image(image, label):
    image = tf.image.resize(tf.cast(image, tf.float32), (224, 224))
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test_cat))
           .map(preprocess_image, num_parallel_calls=AUTOTUNE)
           .batch(64)
           .prefetch(AUTOTUNE))

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=[
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["airplane", "automobile", "bird", "cat", "deer",
                         "dog", "frog", "horse", "ship", "truck"],
            yticklabels=["airplane", "automobile", "bird", "cat", "deer",
                         "dog", "frog", "horse", "ship", "truck"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - CIFAR-10")
plt.show()
