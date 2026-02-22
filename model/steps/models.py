import os
import tensorflow as tf
from tensorflow.keras import layers, applications, optimizers, models

import config as cfg

# Идея - сделать сеть
# которая будет иметь свой инпут, передавать его на MobileNetV3
# у которой будет 3 головы - кол-во, форма и тип закраски
# возможно она должна быть чб чтобы поэткономить место
# и отдельно брать свой инпут и оценивать цвет (hinge loss)???

class SetgameClassificationModel(tf.keras.Model):
    def __init__(self):
        super(SetgameClassificationModel, self).__init__()
        
        self.input_spec_layer = layers.InputLayer(shape=(cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3))

        # based on this model
        self.base_model = applications.MobileNetV3Large(
            input_shape=(cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3),
            include_top=False,
            weights='imagenet'
        )
        # frozen by default
        self.set_fine_tuning(False)
        
        self.pooling = layers.GlobalAveragePooling2D()
        
        # TODO: flow B - color
        # self.custom_branch = models.Sequential([
        #     layers.Conv2D(32, (3, 3), strides=2, activation='relu'),
        #      layers.GlobalAveragePooling2D(name='color_avg_pool'),
        #      layers.Dense(16, activation='relu'),
        #      layers.Dense(1, activation='linear', name='hinge_output')
        # ], name="color")

        # create 4 heads (Dense layers) with 3 one-hot
        self.heads = [layers.Dense(3, activation='softmax', name=f'{name}_output') 
                      for i, name in enumerate(["count", "color", "fill", "shape"])]

    def call(self, inputs, training=False):
        # common input (color (flow B) in future may take this directly)
        x_input = inputs 
        
        # flow A - base model 
        x_mobile = self.base_model(x_input, training=training)
        x_mobile = self.pooling(x_mobile)

        # TODO: flow B - color
        # x_custom = self.custom_branch(x_input)

        # combine output of 3 heads
        # outputs = [head(x_mobile) for head in self.heads]
        # TODO: add flow B
        # outputs.append(x_custom)
        #return outputs
        # Return a dictionary matching your dataset keys
        return {
            "count_output": self.heads[0](x_mobile),
            "color_output": self.heads[1](x_mobile),
            "fill_output": self.heads[2](x_mobile),
            "shape_output": self.heads[3](x_mobile)
        }

    # --- Fine-tuning phase ---
    def set_fine_tuning(self, enabled=True):
        print(f"--- Fine-tuning: {enabled} ---")
        self.base_model.trainable = enabled

    def load(self, checkpoint_path):        
        if os.path.exists(checkpoint_path):
            print(f"--- Found checkpoint - Loading weights from: {checkpoint_path} ---")
            # For Subclassing model we need to do 1 fake pass, 
            # to build vars, or call build()
            self.build((None, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3))
            self.load_weights(checkpoint_path)
        else:
            print("--- No checkpoints. Using ImageNet ---")
            raise FileNotFoundError(f"couldn't find {checkpoint_path}")
        
    def save(self, checkpoint_path):
        self.save_weights(checkpoint_path)
        
