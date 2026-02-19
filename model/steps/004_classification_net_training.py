import os
import tensorflow as tf
from tensorflow.keras import layers, applications, optimizers, models

import config as cfg
import dataset as dtst

# Идея - сделать сеть
# которая будет иметь свой инпут, передавать его на MobileNetV3
# у которой будет 3 головы - кол-во, форма и тип закраски
# возможно она должна быть чб чтобы поэткономить место
# и отдельно брать свой инпут и оценивать цвет (hinge loss)???
# 

class MultiHeadMobileNet(tf.keras.Model):
    def __init__(self, checkpoint_path=cfg.CKPT_PATH):
        super(MultiHeadMobileNet, self).__init__()
        self.ckpt_path = checkpoint_path
        
        self.input_spec_layer = layers.InputLayer(shape=(cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3))

        # Загрузка базы
        self.base_model = applications.MobileNetV3Large(
            input_shape=(cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3),
            include_top=False,
            weights='imagenet'
        )
        self.base_model.trainable = False  # По умолчанию заморожена
        
        self.pooling = layers.GlobalAveragePooling2D()
        
        # self.custom_branch = models.Sequential([
        #     layers.Conv2D(32, (3, 3), strides=2, activation='relu'),
        #      layers.GlobalAveragePooling2D(name='color_avg_pool'),
        #      layers.Dense(16, activation='relu'),
        #      layers.Dense(1, activation='linear', name='hinge_output')
        # ], name="color")

        # Создаем отдельные головы (Dense слои)
        # num_classes_list — список кол-ва классов для каждой головы
        self.heads = [layers.Dense(n, activation='softmax', name=f'head_{i}') 
                      for i, n in enumerate([3, 3, 3, 3])]

    def call(self, inputs, training=False):
        # Пропускаем вход через общий слой (опционально, для типизации)
        x_input = inputs 
        
        # Поток А: MobileNet
        x_mobile = self.base_model(x_input, training=training)
        x_mobile = self.pooling(x_mobile)

        # Поток Б: Ваша ветка
        # x_custom = self.custom_branch(x_input)

        # Возвращаем список выходов от каждой головы
        outputs = [head(x_mobile) for head in self.heads]
        # outputs.append(x_custom)
        return outputs

    def load_or_init(self):
        """Логика загрузки весов без лишних запросов к сети"""
        self.build((None, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3))

        if os.path.exists(self.ckpt_path):
            print(f"--- Найдено! Загружаю локальные веса: {self.ckpt_path} ---")
            # Для Subclassing модели сначала нужно сделать фиктивный проход, 
            # чтобы построить переменные, либо использовать build()
            self.build((None, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3))
            self.load_weights(self.ckpt_path)
        else:
            print("--- Чекпоинтов нет. Используем ImageNet ---")

# --- Настройка обучения ---

# Список классов для  подзадач
model = MultiHeadMobileNet()
model.load_or_init()

# Оптимизатор и функции потерь (по одной на каждую голову)
optimizer = optimizers.Adam(learning_rate=1e-3)
loss_fns = [tf.keras.losses.SparseCategoricalCrossentropy()]
#loss_fns.append(tf.keras.losses.Hinge())
loss_fns.append(tf.keras.losses.SparseCategoricalCrossentropy())
loss_fns.append(tf.keras.losses.SparseCategoricalCrossentropy())
loss_fns.append(tf.keras.losses.SparseCategoricalCrossentropy())

# --- Императивный цикл обучения (Custom Training Loop) ---

@tf.function # Ускорение графа (оптимизация под GPU/NPU)
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        # Суммируем потери от всех голов
        total_loss = sum([loss_fns[i](labels[i], predictions[i]) for i in range(4)])
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss

# --- Фаза Fine-tuning ---
def enable_fine_tuning(lr=1e-5):
    print("--- Включаем Fine-tuning всего графа ---")
    model.base_model.trainable = True
    optimizer.learning_rate = lr

# Пример сохранения
# model.save_weights(cfg.CKPT_PATH)


# Использование
dataset = dtst.create('model/steps/wrk/mixed/')

EPOCHS = 100
# for images, labels in dataset:
#     loss = train_step(images, labels) # labels уже содержит 4 списка меток
for epoch in range(EPOCHS):
    print(f"\n--- Эпоха {epoch + 1}/{EPOCHS} ---")
    
    epoch_loss = 0
    num_batches = 0
    
    for images, labels in dataset:
        loss = train_step(images, labels)
        epoch_loss += loss
        num_batches += 1
        
        if num_batches % 10 == 0:
            print(f"Батч {num_batches}, Текущий Loss: {loss:.4f}")

    # Средний лосс за эпоху
    print(f"Средний Loss за эпоху: {epoch_loss / num_batches:.4f}")
    
    # # Периодическое сохранение чекпоинта (ваше условие)
    # model.save_weights(model.ckpt_path)
    # print(f"Чекпоинт сохранен: {model.ckpt_path}")