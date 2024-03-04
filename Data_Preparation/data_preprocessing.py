import tensorflow as tf
from tensorflow.keras.utils import to_categorical

class DataPreprocessing:
    def __init__(self, strategy, load_data_func, batch_size=32):
        self.strategy = strategy
        self.load_data_func = load_data_func  # Pass the dataset loading function here
        self.batch_size = batch_size * strategy.num_replicas_in_sync
        self.AUTO = tf.data.experimental.AUTOTUNE
    
    def load_and_preprocess_data(self):
        (x_train, y_train), (x_test, y_test) = self.load_data_func()
        
        x_train, x_test = [self._reshape_and_normalize(x) for x in [x_train, x_test]]
        y_train, y_test = [to_categorical(y, 10) for y in [y_train, y_test]]
        
        return self._prepare_datasets(x_train, y_train, x_test, y_test)
    
    def _reshape_and_normalize(self, x):
        return x.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    def _prepare_datasets(self, x_train, y_train, x_test, y_test):
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(self.batch_size).prefetch(self.AUTO)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.batch_size).cache().prefetch(self.AUTO)
        return train_dataset, test_dataset
