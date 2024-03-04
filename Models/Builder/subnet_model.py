# subnet_model.py

from tensorflow.keras.layers import Lambda, Reshape, Dense, BatchNormalization, Dropout
from .base_model import BaseModel
from .utils import DropConnectDense  # Assuming you have defined DropConnectDense in a utils module

class SubNetModel(BaseModel):
    def build_subnet(self, input_tensor, num_subnets):
        subnet_outputs = []
        shape = K.int_shape(input_tensor)
        num_feature_maps = shape[-1]
        subnet_feature_maps = num_feature_maps // num_subnets

        for i in range(num_subnets):
            subnet_input = Lambda(lambda z: z[:, :, :, i * subnet_feature_maps:(i + 1) * subnet_feature_maps])(input_tensor)
            subnet_input = Reshape((shape[1] * shape[2] * subnet_feature_maps,))(subnet_input)

            # Subnetwork layers
            fc = Dense(512, activation='relu')(subnet_input)
            # ... (other layers) ...

            subnet_output = Dense(10, activation='softmax')(fc)
            subnet_outputs.append(subnet_output)
        
        return subnet_outputs

