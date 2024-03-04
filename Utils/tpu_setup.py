# tpu_utils.py

import tensorflow as tf

class TPUConfig:
    def __init__(self):
        self.strategy = self.detect_tpu()

    def detect_tpu(self):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.master())
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
        except ValueError:
            print('TPU not found')
            strategy = tf.distribute.get_strategy()
        return strategy

    @property
    def replicas(self):
        return self.strategy.num_replicas_in_sync