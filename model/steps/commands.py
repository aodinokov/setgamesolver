import numpy as np
import tensorflow as tf
from tensorflow import keras

import datasets as dtst
import models as mdl

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
            patience=5,
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
    train_parser = subparsers.add_parser(
        "classification_train", 
        help="Start the model training process"
    )    
    train_parser.add_argument("--dataset_path", type=str, default="wrk/mixed/", 
                              help="Path to dataset directory")
    train_parser.add_argument("--from_checkpoint_path", type=str, default=None, 
                              help="Initial weights to load")
    train_parser.add_argument("--checkpoint_path", type=str, default="checkpoints/default.classification.setgamemodel.weights.h5", 
                              help="Directory to save periodic checkpoints")    
    train_parser.add_argument("--finetune", action="store_true", 
                              help="Enable fine-tuning of the base model")
    train_parser.add_argument("--finetune_lr", type=float, default=1e-5, 
                              help="Learning rate for fine-tuning")
    train_parser.add_argument("--epoch_number", type=int, default=1000, 
                              help="Total number of epochs")


    #split and print debug
    params = vars(parser.parse_args())
    command = params.pop("command")
    if not command is None:
        print(f"Got command '{command}' with params {params}")

    # try:
    if command == "classification_train":
            classification_train(**params)
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