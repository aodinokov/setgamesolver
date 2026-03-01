import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import datasets as dtst
import models as mdl
import config as cfg

# this file contains all utilities commands to train/evaluate and etc models

def classification_train(
        dataset_path="wrk/mixed/",      # use dataset from this path
        from_checkpoint_path = None,    # load checkpoint before training (if set)
        checkpoint_path = "checkpoints/default.classification.setgamemodel.weights.h5",         # store checkpoints periodically here (if set)
        # checkpoint_path can be "ex2/cp-{epoch:04d}.weights.h5"
        finetune = False,               # enable fine_tune if enabled (retrain base model)
        finetune_lr = 1e-5,             # adjust learning rate for finetune (useful for finetuning, since it should be smaller than usual LR)
        epoch_number = 1000             # default max
        ):
    model = mdl.SetgameClassificationModel()
    # if "from" checkpoint path is set - use it
    if not from_checkpoint_path is None:
        model.load(from_checkpoint_path)

    callbacks = []
    learning_rate = 1e-3
    if finetune:
        learning_rate = finetune_lr
        model.set_fine_tuning(True)
        # added potentially this will add value for finetuning
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=20,
            min_lr=1e-7
        ))


    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                loss={'count_output': 'sparse_categorical_crossentropy', 
                    'color_output': 'sparse_categorical_crossentropy',
                    'fill_output': 'sparse_categorical_crossentropy', 
                    'shape_output': 'sparse_categorical_crossentropy'},
                loss_weights={'count_output': 1.0, 'color_output': 1.0,'fill_output': 1.0, 'shape_output': 1.0},
                metrics=['accuracy', 'accuracy', 'accuracy', 'accuracy'])   # maybe need 'sparse_categorical_accuracy',...

    # Display the model summary
    model.summary()

    # some callback
    # let's do this by default
    callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        )
    )
    if not checkpoint_path is None:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath = checkpoint_path,
            verbose = 1,
            save_weights_only = True,
            save_freq = 'epoch')
        )

    train_ds, val_ds = dtst.ClassificationDataset.create(dataset_path)
    history = model.fit(train_ds, 
                        epochs=epoch_number, 
                        batch_size=32, 
                        validation_data=val_ds,
                        verbose=1,
                        callbacks=callbacks)

    # Final evaluattion of the model
    results = model.evaluate(val_ds, verbose=0)
    total_loss = results[0]
    print(f"Test total loss: {total_loss:.4f}")
    test_losses = results[1:5]
    for name, lss in zip(['Count', 'Color', 'Fill', 'Shape'], test_losses):
        print(f"{name} Loss: {lss:.4f}")
    test_accuracies = results[5:9]
    for name, acc in zip(['Count', 'Color', 'Fill', 'Shape'], test_accuracies):
        print(f"{name} Accuracy: {acc:.4f}")

    # Make predictions by original model
    for images, labels in val_ds.take(1):
        predictions = model.predict(images)

        for name, preds in predictions.items():
            p = np.argmax(preds, axis=1)
            actuals = labels[name].numpy()
            print(f"{name} predictions: {p[:5]}")
            print(f"{name} actuals:     {actuals[:5]}")
        break

def classification_export(
        from_checkpoint_path = "checkpoints/default.classification.setgamemodel.weights.h5",
        export_path = "export/classification-setgamemodel",
        tflite = False, # also convert to tflite
        tflite_path = "tflite/classification-setgamemodel.tflite"
        ):
    model = mdl.SetgameClassificationModel()
    # if "from" checkpoint path is set - use it
    if not from_checkpoint_path is None:
        model.load(from_checkpoint_path)
    
    model.export(export_path)

    if tflite:
        # in order to convert to tflite we need the exporeted model path
        converter = tf.lite.TFLiteConverter.from_saved_model(export_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        # add metadata:
        # see https://ai.google.dev/edge/litert/conversion/tensorflow/metadata on how to update tflite metadata
        # those steps required some fixes to the code in tensorflow_lite_support.
        # see local tensorflow_lite_support/Readme.md
        import flatbuffers
        from tensorflow_lite_support.metadata import metadata as _metadata
        from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
        
        labels_file_paths = [
            "classification-setgamemodel-count.labels",
            "classification-setgamemodel-color.labels",
            "classification-setgamemodel-fill.labels",
            "classification-setgamemodel-shape.labels"
        ]


        # Creates model info.
        model_meta = _metadata_fb.ModelMetadataT()
        model_meta.name = "MobileNetV3 setgame card classifier"
        model_meta.description = ("Classify the cards 4 features, each can have 3 values: count, color, fill, shape")
        model_meta.version = "v1"
        model_meta.author = "TensorFlow"
        model_meta.license = ("Apache License. Version 2.0 "
                            "http://www.apache.org/licenses/LICENSE-2.0.")
        
        # Creates input info.
        input_meta = _metadata_fb.TensorMetadataT()
        input_meta.name = "image"
        input_meta.description = (
            "Input image to be classified. The expected image is {0} x {1}, with "
            "three channels (red, blue, and green) per pixel. Each value in the "
            "tensor is a single byte between 0 and 255.".format(cfg.IMG_SIZE[0], cfg.IMG_SIZE[0]))
        input_meta.content = _metadata_fb.ContentT()
        input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = (
            _metadata_fb.ColorSpaceType.RGB)
        input_meta.content.contentPropertiesType = (
            _metadata_fb.ContentProperties.ImageProperties)
        input_normalization = _metadata_fb.ProcessUnitT()
        input_normalization.optionsType = (
            _metadata_fb.ProcessUnitOptions.NormalizationOptions)
        input_normalization.options = _metadata_fb.NormalizationOptionsT()
        input_normalization.options.mean = [127.5]
        input_normalization.options.std = [127.5]
        input_meta.processUnits = [input_normalization]
        input_stats = _metadata_fb.StatsT()
        input_stats.max = [255]
        input_stats.min = [0]
        input_meta.stats = input_stats

        # Creates output info.
        output_meta_arr = []
        for i in range(4):
            output_meta_arr.append(_metadata_fb.TensorMetadataT())
            output_meta_arr[i].name = f"probability{i}" # TODO: change to the proper name
            output_meta_arr[i].description = "Probabilities of the 1 of the 3 values of the feature."
            output_meta_arr[i].content = _metadata_fb.ContentT()
            output_meta_arr[i].content.content_properties = _metadata_fb.FeaturePropertiesT()
            output_meta_arr[i].content.contentPropertiesType = (
                _metadata_fb.ContentProperties.FeatureProperties)
            output_stats = _metadata_fb.StatsT()
            output_stats.max = [1.0]
            output_stats.min = [0.0]
            output_meta_arr[i].stats = output_stats
            # not sure:
            label_file = _metadata_fb.AssociatedFileT()
            label_file.name = os.path.basename(labels_file_paths[i])
            label_file.description = "Labels for objects that the model can recognize."
            label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
            output_meta_arr[i].associatedFiles = [label_file]

        # Creates subgraph info.
        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = output_meta_arr
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(
            model_meta.Pack(b),
            _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        # Pack metadata and associated files into the modely
        populator = _metadata.MetadataPopulator.with_model_file(tflite_path)
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files(labels_file_paths)
        populator.populate()

        # import zipfile
        # import io

        # def pack_tflite_with_metadata(model_path, output_path, label_files, metadata_json_path):
        #     # 1. Читаем исходную модель
        #     with open(model_path, 'rb') as f:
        #         model_content = f.read()

        #     # 2. Создаем Zip-архив в памяти
        #     zip_buffer = io.BytesIO()
        #     with zipfile.ZipFile(zip_buffer, 'w') as zf:
        #         # Добавляем JSON описания
        #         zf.write(metadata_json_path, "metadata.json")
        #         # Добавляем все твои файлы меток
        #         for file in label_files:
        #             zf.write(file, file)

        #     # 3. Склеиваем модель и Zip-архив
        #     # В спецификации TFLite Zip-архив просто дописывается в конец файла
        #     with open(output_path, 'wb') as f:
        #         f.write(model_content)
        #         f.write(zip_buffer.getvalue())

        # # Использование для твоего случая:
        # labels = [
        #     "classification-setgamemodel-color.labels",
        #     "classification-setgamemodel-shape.labels",
        #     "classification-setgamemodel-fill.labels",
        #     "classification-setgamemodel-count.labels"
        # ]
        # pack_tflite_with_metadata(
        #     tflite_path, 
        #     f"{tflite_path}.meta", 
        #     labels, 
        #     "classification-setgamemodel-metadata.json"
        # )

        # import flatbuffers
        # import os
        # import sys

        # # Твои сгенерированные классы
        # from tflite.Model import Model
        # from tflite.ModelMetadata import ModelMetadata
        # from tflite.Metadata import Metadata
        # from tflite.Buffer import Buffer

        # def patch_model(input_path, output_path):
        #     with open(input_path, 'rb') as f:
        #         buf = bytearray(f.read())

        #     # 1. Распаковываем текущую модель
        #     orig_model = Model.GetRootAs(buf, 0)
        #     builder = flatbuffers.Builder(len(buf) + 1024)

        #     # 2. Создаем Payload (само тело метаданных)
        #     # Здесь мы создаем ModelMetadata, который Android Studio ждет внутри буфера
        #     desc = builder.CreateString("SetGameSolver")
        #     ModelMetadataStart(builder)
        #     ModelMetadataAddDescription(builder, desc)
        #     # В реальном сценарии здесь добавляются оффсеты для под-таблиц (Input/Output)
        #     metadata_payload_offset = ModelMetadataEnd(builder)
        #     builder.Finish(metadata_payload_offset)
        #     payload_bytes = builder.Output()

        #     # 3. Пересобираем буферы
        #     # Нам нужно скопировать все старые буферы и добавить наш новый
        #     new_buffers = []
        #     for i in range(orig_model.BuffersLength()):
        #         old_data = orig_model.Buffers(i).DataAsNumpy()
        #         data_off = builder.CreateByteVector(old_data)
        #         BufferStart(builder)
        #         BufferAddData(builder, data_off)
        #         new_buffers.append(BufferEnd(builder))

        #     # Добавляем наш новый буфер с метаданными
        #     payload_off = builder.CreateByteVector(payload_bytes)
        #     BufferStart(builder)
        #     BufferAddData(builder, payload_off)
        #     new_metadata_buffer_idx = len(new_buffers)
        #     new_buffers.append(BufferEnd(builder))

        #     ModelStartBuffersVector(builder, len(new_buffers))
        #     for b in reversed(new_buffers):
        #         builder.PrependUOffsetTRelative(b)
        #     final_buffers_v = builder.EndVector()

        #     # 4. Пересобираем таблицу Metadata
        #     new_metadata_entries = []
        #     # Копируем старые (min_runtime_version, CONVERSION_METADATA)
        #     for i in range(orig_model.MetadataLength()):
        #         old_entry = orig_model.Metadata(i)
        #         name_off = builder.CreateString(old_entry.Name().decode('utf-8'))
        #         MetadataStart(builder)
        #         MetadataAddName(builder, name_off)
        #         MetadataAddBuffer(builder, old_entry.Buffer())
        #         new_metadata_entries.append(MetadataEnd(builder))

        #     # Добавляем запись TFLITE_METADATA, указывающую на наш новый буфер
        #     name_off = builder.CreateString("TFLITE_METADATA")
        #     MetadataStart(builder)
        #     MetadataAddName(builder, name_off)
        #     MetadataAddBuffer(builder, new_metadata_buffer_idx)
        #     new_metadata_entries.append(MetadataEnd(builder))

        #     ModelStartMetadataVector(builder, len(new_metadata_entries))
        #     for m in reversed(new_metadata_entries):
        #         builder.PrependUOffsetTRelative(m)
        #     final_metadata_v = builder.EndVector()

        #     # 5. Финализируем модель (копируем остальные поля)
        #     # ВНИМАНИЕ: Для полной работы нужно скопировать Subgraphs, Tensors и т.д.
        #     # Но так как мы не хотим хачить всю структуру, используй этот принцип:
        #     ModelStart(builder)
        #     ModelAddBuffers(builder, final_buffers_v)
        #     ModelAddMetadata(builder, final_metadata_v)
        #     # ... здесь должны быть вызовы для остальных полей из orig_model ...
        #     model_off = ModelEnd(builder)
        #     builder.Finish(model_off, b'TFL3') # Обязательный file_identifier

        #     with open(output_path, 'wb') as f:
        #         f.write(builder.Output())

        # # ВНИМАНИЕ: Полная пересборка через Builder в Python требует копирования 
        # # КАЖДОГО поля (Subgraphs, OperatorCodes и т.д.).

# main utility code
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="System entry point for multiple ML commands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- classification_train ---
    ctrain_parser = subparsers.add_parser(
        "classification_train", 
        help="Start the model training process"
    )    
    ctrain_parser.add_argument("--dataset_path", type=str, default="wrk/mixed/", 
                              help="Path to dataset directory")
    ctrain_parser.add_argument("--from_checkpoint_path", type=str, default=None, 
                              help="Initial weights to load")
    ctrain_parser.add_argument("--checkpoint_path", type=str, default="checkpoints/default.classification.setgamemodel.weights.h5", 
                              help="Directory to save periodic checkpoints")    
    ctrain_parser.add_argument("--finetune", action="store_true", 
                              help="Enable fine-tuning of the base model")
    ctrain_parser.add_argument("--finetune_lr", type=float, default=1e-5, 
                              help="Learning rate for fine-tuning")
    ctrain_parser.add_argument("--epoch_number", type=int, default=1000, 
                              help="Total number of epochs")

    # --- classification_export ---
    cexport_parser = subparsers.add_parser(
        "classification_export", 
        help="Load checkpoint generated by classification_train command and export (includes also conversion to tflite)"
    )    

    cexport_parser.add_argument("--from_checkpoint_path", type=str, default="checkpoints/default.classification.setgamemodel.weights.h5", 
                              help="Initial weights to load")
    cexport_parser.add_argument("--export_path", type=str, default="export/classification-setgamemodel", 
                              help="Path to the directory where exported model will be put")
    cexport_parser.add_argument("--tflite", action="store_true", 
                              help="Enable also conversion to tflite")
    cexport_parser.add_argument("--tflite_path", type=str, default="tflite/classification-setgamemodel.tflite", 
                              help="Path to the tflite file where exported tflite version of the model will be put")


    #split and print debug
    params = vars(parser.parse_args())
    command = params.pop("command")
    if not command is None:
        print(f"Got command '{command}' with params {params}")

    # try:
    if command == "classification_train":
            classification_train(**params)
    elif command == "classification_export":
            classification_export(**params)
    elif command is None:
        parser.print_help()            
    # except KeyboardInterrupt:
    #     print("\n[System] Process interrupted.")
    #     sys.exit(0)
    # except Exception as e:
    #     print(f"\n[System Error] {e}")
    #     sys.exit(1)

if __name__ == "__main__":
    main()