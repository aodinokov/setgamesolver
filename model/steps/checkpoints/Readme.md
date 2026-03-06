This folder contains some good checkpoints which may be used in future

[2026-02-23.tuned.classification.setgamemodel](2026-02-23.tuned.classification.setgamemodel.weights.h5)
was tuned using [this script](../010_classification_model.sh) based on the checkpoint [2026-02-22.trained.classification.setgamemodel](2026-02-22.trained.classification.setgamemodel.weights.h5)

The final results were:

```
Epoch 1000: saving model to checkpoints/default.classification.setgamemodel.weights.h5

Epoch 1000: finished saving model to checkpoints/default.classification.setgamemodel.weights.h5
404/404 ━━━━━━━━━━━━━━━━━━━━ 28s 55ms/step - color_output_accuracy: 1.0000 - color_output_loss: 8.6927e-06 - count_output_accuracy: 1.0000 - count_output_loss: 9.6317e-06 - fill_output_accuracy: 1.0000 - fill_output_loss: 9.5510e-06 - loss: 2.4279e-04 - shape_output_accuracy: 1.0000 - shape_output_loss: 8.4563e-06 - val_color_output_accuracy: 1.0000 - val_color_output_loss: 6.1507e-07 - val_count_output_accuracy: 1.0000 - val_count_output_loss: 5.2426e-07 - val_fill_output_accuracy: 1.0000 - val_fill_output_loss: 5.6690e-07 - val_loss: 2.2729e-04 - val_shape_output_accuracy: 1.0000 - val_shape_output_loss: 1.9127e-05 - learning_rate: 1.0000e-07
Restoring model weights from the end of the best epoch: 995.
Test total loss: 0.0002
Count Loss: 0.0000
Color Loss: 0.0000
Fill Loss: 0.0000
Shape Loss: 0.0000
Count Accuracy: 1.0000
Color Accuracy: 1.0000
Fill Accuracy: 1.0000
Shape Accuracy: 1.0000
1/1 ━━━━━━━━━━━━━━━━━━━━ 8s 8s/step
count_output predictions: [2 1 0 0 0]
count_output actuals:     [2 1 0 0 0]
color_output predictions: [0 1 1 1 2]
color_output actuals:     [0 1 1 1 2]
fill_output predictions: [1 2 2 1 1]
fill_output actuals:     [1 2 2 1 1]
shape_output predictions: [2 2 1 2 1]
shape_output actuals:     [2 2 1 2 1]
```