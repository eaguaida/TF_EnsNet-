# base_model.py
import tensorflow as tf

class BaseModel:
    def __init__(self, strategy):
        self.strategy = strategy
    
    def build(self):
        raise NotImplementedError("Build method must be implemented by the subclass.")
