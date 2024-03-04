import tensorflow as tf
import tensorflow_addons as tfa
from .cnn_model import CNNModel
from .subnet_model import SubnetModel
from .tpu_config import TPUConfig  # Assuming TPUConfig is defined in a module named tpu_config.py

class FullModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.tpu_config = TPUConfig()
        # Instantiate CNNModel and SubnetModel with the TPU strategy
        self.cnn_model = CNNModel(self.tpu_config.strategy)
        self.subnet_model = SubnetModel(self.tpu_config.strategy)

    def build(self):
        # Build the base model and the CNN part
        cnn_inputs, cnn_base_output = self.cnn_model.build_base(self.input_shape)
        cnn_output = self.cnn_model.build_cnn(cnn_base_output)
        
        # Build the subnet part based on the base model output
        subnet_outputs = self.subnet_model.create_subnets(cnn_base_output)
        # Append the CNN output to the subnet outputs
        subnet_outputs.append(cnn_output)

        with self.tpu_config.strategy.scope():
            # Create the full model with both CNN and subnet outputs
            full_model = tf.keras.Model(inputs=cnn_inputs, outputs=subnet_outputs)
            
            # Define the optimizer and compile the model
            adamw_optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0)
            full_model.compile(loss='categorical_crossentropy', optimizer=adamw_optimizer, metrics=['accuracy'])
            
            return full_model
