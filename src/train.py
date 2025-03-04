import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import cifar10

class RLRRDense(layers.Layer):
    def __init__(self, units, use_bias=True, kernel_initializer='glorot_uniform', **kwargs):
        super(RLRRDense, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.W_frozen = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=False,
            name="W_frozen"
        )
        self.s_left = self.add_weight(
            shape=(input_dim, 1),
            initializer='zeros',
            trainable=True,
            name="s_left"
        )
        self.s_right = self.add_weight(
            shape=(1, self.units),
            initializer='zeros',
            trainable=True,
            name="s_right"
        )
        self.f = self.add_weight(
            shape=(input_dim, self.units),
            initializer='zeros',
            trainable=True,
            name="f_bias"
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias"
            )
        else:
            self.bias = None
        super(RLRRDense, self).build(input_shape)
    
    def call(self, inputs):
        scaling = 1 + tf.matmul(self.s_left, self.s_right)
        W_reparam = scaling * self.W_frozen + self.f
        output = tf.matmul(inputs, W_reparam)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        return output

vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
vit_layer = hub.KerasLayer(vit_url, trainable=False)

def build_model():
    input_img = tf.keras.Input(shape=(224, 224, 3))
    x = layers.Rescaling(1./255.0)(input_img)
    # Wrap vit_layer in a Lambda to avoid symbolic tensor issues
    features = layers.Lambda(lambda img: vit_layer(img))(x)
    x = RLRRDense(256, use_bias=True)(features)
    x = layers.ReLU()(x)
    output = layers.Dense(10, activation='softmax')(x)
    model = Model(inputs=input_img, outputs=output)
    return model

def preprocess_image(image, label):
    image = tf.image.resize(tf.cast(image, tf.float32), (224, 224))
    return image, label

def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train_cat))
                .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                .batch(64)
                .prefetch(AUTOTUNE))
    test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test_cat))
               .map(preprocess_image, num_parallel_calls=AUTOTUNE)
               .batch(64)
               .prefetch(AUTOTUNE))
    
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=test_ds, epochs=5)
    
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    model.save("vit_finetuned_rlrr_model.h5")
    
if __name__ == '__main__':
    main()
