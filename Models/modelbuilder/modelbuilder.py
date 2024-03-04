from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

class ModelBuilder:
    def __init__(self, input_shape=(28, 28, 1)):
        self.input_shape = input_shape
    
    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        # Add your model layers...
        x = Flatten()(x)
        outputs = Dense(10, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile_and_train_model(self, model, train_dataset, val_dataset, epochs=10):
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
        return history
