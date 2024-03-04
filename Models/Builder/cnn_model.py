from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from .base_model import BaseModel
from .utils import DropConnectDense  # Assuming DropConnectDense is defined in utils.py

class CNNModel(BaseModel):
    def build(self, input_shape):
        inputs, x = self.build_base(input_shape)
        CNN = x
        CNN = Flatten()(CNN)
        CNN = Dense(512, activation='relu')(CNN)
        CNN = BatchNormalization()(CNN)
        CNN = Dropout(0.5)(CNN)
        CNN = DropConnectDense(512, activation='relu', prob=0.5)(CNN)
        outputs = Dense(10, activation='softmax')(CNN)
        model = Model(inputs=inputs, outputs=outputs)
        return model
