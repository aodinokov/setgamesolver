import os
import tensorflow as tf

import config as cfg

def parse_folder_name(folder_name):
    """Разбирает строку вида '1-green-empty-diamonds'"""
    parts = folder_name.lower().split('-')
    if len(parts) < 4: return None
    
    try:
        # 1-3 -> 0-2
        count_idx = int(parts[0]) - 1
        color_idx = cfg.COLORS.index(parts[1])
        fill_idx  = cfg.FILLS.index(parts[2])
        # Маппинг для shapes (единственное/множественное)
        shape_idx = cfg.SHAPES_MAP[parts[3]]
        
        return (count_idx, color_idx, fill_idx, shape_idx)
    except (ValueError, KeyError, IndexError):
        return None

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]])
    return img

def create(root_dir, batch_size=32):
    image_paths = []
    labels_count = []
    labels_color = []
    labels_fill = []
    labels_shape = []

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            l_count, l_color, l_fill, l_shape = parse_folder_name(folder)
            
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(folder_path, img_name))
                    labels_count.append(l_count)
                    labels_color.append(l_color)
                    labels_fill.append(l_fill)
                    labels_shape.append(l_shape)

    # Создаем TF Dataset
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    img_ds = path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Создаем Dataset для меток (кортеж из 4 выходов)
    label_ds = tf.data.Dataset.from_tensor_slices((
        labels_count, labels_color, labels_fill, labels_shape
    ))

    # Объединяем картинки и метки
    ds = tf.data.Dataset.zip((img_ds, label_ds))
    
    # Оптимизация под железо
    ds = ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

import random

# def create_datasets(root_dir, batch_size=32, split_ratio=0.8):
#     all_paths = []
#     # Сначала собираем ВСЕ пути (как мы делали раньше)
#     # ... (код обхода папок) ...

#     # Перемешиваем пути один раз
#     random.seed(42) # Фиксируем сид для воспроизводимости
#     random.shuffle(all_paths)

#     # Точка разделения
#     split_idx = int(len(all_paths) * split_ratio)
#     train_paths = all_paths[:split_idx]
#     val_paths = all_paths[split_idx:]

#     # Создаем два отдельных объекта tf.data.Dataset
#     train_ds = build_pipeline(train_paths, batch_size)
#     val_ds = build_pipeline(val_paths, batch_size)

#     return train_ds, val_ds