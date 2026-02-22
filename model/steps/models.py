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
        self.heads = [layers.Dense(n, activation='softmax', name=f'head_{i}') 
                      for i, n in enumerate([3, 3, 3, 3])]

    def call(self, inputs, training=False):
        # common input (color (flow B) in future may take this directly)
        x_input = inputs 
        
        # flow A - base model 
        x_mobile = self.base_model(x_input, training=training)
        x_mobile = self.pooling(x_mobile)

        # TODO: flow B - color
        # x_custom = self.custom_branch(x_input)

        # combine output of 3 heads
        outputs = [head(x_mobile) for head in self.heads]
        # TODO: add flow B
        # outputs.append(x_custom)
        return outputs

    # Not absolutely necessary, but it's better to have this helpers here
    # loss_fns depend heaviliy on the model structure, so lets keep this code as a static fn
    def get_loss_fns(self):
        loss_fns = [tf.keras.losses.SparseCategoricalCrossentropy()]
        #loss_fns.append(tf.keras.losses.Hinge())
        loss_fns.append(tf.keras.losses.SparseCategoricalCrossentropy())
        loss_fns.append(tf.keras.losses.SparseCategoricalCrossentropy())
        loss_fns.append(tf.keras.losses.SparseCategoricalCrossentropy())
        return loss_fns

    # --- Fine-tuning phase ---
    def set_fine_tuning(self, enabled=True):
        print(f"--- Fine-tuning: {enabled} ---")
        self.base_model.trainable = enabled

    def load(self, checkpoint_path=cfg.CKPT_PATH):        
        if os.path.exists(checkpoint_path):
            print(f"--- Found checkpoint - Loading weights from: {checkpoint_path} ---")
            # For Subclassing model we need to do 1 fake pass, 
            # to build vars, or call build()
            self.build((None, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3))
            self.load_weights(checkpoint_path)
        else:
            print("--- No checkpoints. Using ImageNet ---")
            raise FileNotFoundError(f"couldn't find {checkpoint_path}")
        
    def save(self, checkpoint_path=cfg.CKPT_PATH):
        self.save_weights(checkpoint_path)
        
