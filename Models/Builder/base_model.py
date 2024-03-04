import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, MaxPooling2D

class BaseModel:
    def __init__(self, strategy):
        self.strategy = strategy

    def build_base(self, input_shape):
        with self.strategy.scope():
            inputs = Input(shape=input_shape)
            
            # First Block
            x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='valid')(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.35)(x)
            x = Conv2D(128, (3, 3), activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.35)(x)
            x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='valid')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            
            # Second Block
            x = Dropout(0.35)(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='valid')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.35)(x)
            x = Conv2D(1024, (3, 3), activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.35)(x)
            x = Conv2D(2000, (3, 3), activation='relu', padding='valid')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.35)(x)

            return inputs, x
