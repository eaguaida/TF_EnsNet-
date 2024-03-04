
import tensorflow as tf
import math

class DataAugmentation:
    def __init__(self, rotation_range=10, shear_range=3, zoom_range=0.08, shift_range=8):
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.shift_range = shift_range

    def get_mat(self, rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
        # returns 3x3 transform matrix which transforms indicies
        
        rotation = math.pi * rotation / 180.
        
        # Rotation Matrix
        c1 = tf.math.cos(rotation)
        s1 = tf.math.sin(rotation)
        rotation_matrix = tf.reshape(tf.concat([c1, s1, tf.zeros([1]), -s1, c1, tf.zeros([1]), tf.zeros([2]), tf.ones([1])], axis=0), [3, 3])
        
        # Shear Matrix
        c2 = tf.math.cos(shear)
        s2 = tf.math.sin(shear)
        shear_matrix = tf.reshape(tf.concat([tf.ones([1]), s2, tf.zeros([1]), tf.zeros([1]), c2, tf.zeros([1]), tf.zeros([2]), tf.ones([1])], axis=0), [3, 3])    

        # Zoom Matrix
        zoom_matrix = tf.reshape(tf.concat([tf.ones([1])/height_zoom, tf.zeros([1]), tf.zeros([1]), tf.zeros([1]), tf.ones([1])/width_zoom, tf.zeros([1]), tf.zeros([2]), tf.ones([1])], axis=0), [3, 3])
        
        # Shift Matrix
        shift_matrix = tf.reshape(tf.concat([tf.ones([1]), tf.zeros([1]), height_shift, tf.zeros([1]), tf.ones([1]), width_shift, tf.zeros([2]), tf.ones([1])], axis=0), [3, 3])

        return tf.linalg.matmul(tf.linalg.matmul(rotation_matrix, shear_matrix), tf.linalg.matmul(zoom_matrix, shift_matrix))

    def transform(self, image, label):
        DIM = image.shape[0]
        XDIM = DIM % 2
        
        rot = self.rotation_range * tf.random.normal([1], dtype='float32')
        shr = self.shear_range * tf.random.normal([1], dtype='float32') 
        h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / self.zoom_range
        w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / self.zoom_range
        h_shift = self.shift_range * tf.random.normal([1], dtype='float32') 
        w_shift = self.shift_range * tf.random.normal([1], dtype='float32') 
  
        m = self.get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift) 

        x = tf.repeat(tf.range(DIM//2, -DIM//2, -1), DIM)
        y = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
        z = tf.ones([DIM*DIM], dtype='int32')
        idx = tf.stack([x, y, z])
        
        idx2 = tf.keras.backend.dot(m, tf.cast(idx, dtype='float32'))
        idx2 = tf.keras.backend.cast(idx2, dtype='int32')
        idx2 = tf.keras.backend.clip(idx2, -DIM//2+XDIM+1, DIM//2)
        
        idx3 = tf.stack([DIM//2 - idx2[0, ], DIM//2 - 1 + idx2[1, ]])
        d = tf.gather_nd(image, tf.transpose(idx3))
        
        return tf.reshape(d, [DIM, DIM, 3]), label