import tensorflow as tf
alphabets = ['Ka', 'Kha', 'Ga', 'Gha', 'Nga', 'Cha', 'Chha', 'Ja', 'Jha', 'Yan', 'Ta', 'Tha', 'Da', 'Dha', 'Na',
             'Taa', 'Thaa', 'Daa', 'Dhaa', 'Naa', 'Pa', 'Pha', 'Ba', 'Bha', 'Ma', 'Ya', 'Ra', 'La', 'Wa', 'T_Sha', 
             'M_Sha', 'D_Sha', 'Ha', 'Ksha', 'Tra', 'Gya']

# Custom Layers
class CustomConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation, **kwargs):
        super(CustomConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.kernel_size, int(input_shape[-1]), self.filters),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(self.filters,),
            initializer='zeros',
            trainable=True,
        )

    def call(self, inputs):
        conv = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='SAME')
        conv = tf.nn.bias_add(conv, self.bias)
        return self.activation(conv)

class CustomMaxPooling1D(tf.keras.layers.Layer):
    def __init__(self, pool_size, **kwargs):
        super(CustomMaxPooling1D, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        return tf.nn.pool(inputs, window_shape=[self.pool_size], pooling_type='MAX', padding='SAME')

class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units, activation, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(int(input_shape[-1]), self.units),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )

    def call(self, inputs):
        dense = tf.matmul(inputs, self.kernel) + self.bias
        return self.activation(dense)


# Custom Functions
def custom_relu(x):
    return tf.maximum(0.0, x)

def custom_softmax(x):
    e_x = tf.exp(x - tf.reduce_max(x, axis=-1, keepdims=True))  # For numerical stability
    return e_x / tf.reduce_sum(e_x, axis=-1, keepdims=True)

def custom_categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

def custom_accuracy(y_true, y_pred):
    correct_predictions = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

@tf.keras.utils.register_keras_serializable()
class NSLPredictionModel(tf.keras.Model):
    def __init__(self):
        super(NSLPredictionModel, self).__init__()
        self.conv1 = CustomConv1D(64, kernel_size=3, activation=custom_relu)
        self.pool1 = CustomMaxPooling1D(pool_size=2)
        self.conv2 = CustomConv1D(128, kernel_size=3, activation=custom_relu)
        self.pool2 = CustomMaxPooling1D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = CustomDense(128, activation=custom_relu)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = CustomDense(len(alphabets), activation=custom_softmax)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)
    
    def compute_output_shape(self, input_shape):
        shape = self.conv1.compute_output_shape(input_shape)
        shape = self.pool1.compute_output_shape(shape)
        shape = self.conv2.compute_output_shape(shape)
        shape = self.pool2.compute_output_shape(shape)
        shape = (shape[0], shape[1] * shape[2])
        shape = (shape[0], self.dense1.units)
        shape = (shape[0], len(alphabets))
        return shape

    def get_config(self):
        # Return configuration for serialization
        return super(NSLPredictionModel, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls()
