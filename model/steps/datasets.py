import os
import random
import tensorflow as tf

import config as cfg

def _load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]])
    return img

# helper to make classification dataset based on the structure of data we have
class ClassificationDataset:
    COLORS = ['green', 'purple', 'red']
    FILLS  = ['empty', 'striped', 'solid']
    SHAPES_MAP = {'diamond': 0, 'diamonds': 0, 'oval': 1, 'ovals': 1, 'squiggle': 2, 'squiggles': 2}

    @classmethod
    def _parse_folder_name(cls, folder_name):
        """Разбирает строку вида '1-green-empty-diamonds'"""
        parts = folder_name.lower().split('-')
        if len(parts) < 4: return None
        
        try:
            # 1-3 -> 0-2
            count_idx = int(parts[0]) - 1
            color_idx = cls.COLORS.index(parts[1])
            fill_idx  = cls.FILLS.index(parts[2])
            # shapes mappings (can be singular or plural)
            shape_idx = cls.SHAPES_MAP[parts[3]]
            
            return (count_idx, color_idx, fill_idx, shape_idx)
        except (ValueError, KeyError, IndexError):
            return None

    @classmethod
    def create(cls, root_dir, batch_size=32, split_ratio=0.8, seed=None):
        # slices: same index -> same picture/label
        image_paths = []
        labels_count = []
        labels_color = []
        labels_fill = []
        labels_shape = []

        # TODO: # another approach to consider (requires different data org)
        # random.seed(42)
        # random.shuffle(all_paths)
        # # splitting
        # split_idx = int(len(all_paths) * split_ratio)
        # train_paths = all_paths[:split_idx]
        # val_paths = all_paths[split_idx:]

        # build slices
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                l_count, l_color, l_fill, l_shape = cls._parse_folder_name(folder)
                
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(folder_path, img_name))
                        labels_count.append(l_count)
                        labels_color.append(l_color)
                        labels_fill.append(l_fill)
                        labels_shape.append(l_shape)

        # build datsets
        path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        img_ds = path_ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

        label_ds = tf.data.Dataset.from_tensor_slices((
            labels_count, labels_color, labels_fill, labels_shape
        ))

        # combine img/labels datasets and labels
        ds = tf.data.Dataset.zip((img_ds, label_ds))
        
        full_ds = ds.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=False)
        
        # split train/validate
        train_size = int(split_ratio * len(image_paths))
        train_ds = full_ds.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = full_ds.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return train_ds, val_ds 
