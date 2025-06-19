import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import datetime # For TensorBoard logs
import os



# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Custom GCN Layer (as provided previously)
class CustomGCNConv(layers.Layer):
    def __init__(self, channels, activation=None, kernel_regularizer=None, **kwargs):
        super(CustomGCNConv, self).__init__(**kwargs)
        self.channels = channels
        self.activation = keras.activations.get(activation)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.dense = layers.Dense(channels, kernel_regularizer=self.kernel_regularizer, use_bias=False) # Bias handled by BatchNormalization or added later

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="bias",
            shape=(self.channels,),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        X, A = inputs

        # Compute D_hat_inv_sqrt for symmetric normalization
        D_hat = tf.reduce_sum(A, axis=-1, keepdims=True)
        # Add a small epsilon to avoid division by zero or inf
        D_hat_inv_sqrt = tf.pow(D_hat + 1e-10, -0.5)
        D_hat_inv_sqrt = tf.where(tf.math.is_inf(D_hat_inv_sqrt), tf.zeros_like(D_hat_inv_sqrt), D_hat_inv_sqrt)

        # Symmetrically normalized adjacency matrix: D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
        # For batch processing, use element-wise multiplication with broadcasting
        A_norm = D_hat_inv_sqrt * A * tf.transpose(D_hat_inv_sqrt, perm=[0, 2, 1])

        # Graph convolution: A_norm @ X_transformed
        # First, apply the dense layer (W) to X
        X_transformed = self.dense(X)
        output = tf.matmul(A_norm, X_transformed)

        # Add bias
        output = output + self.bias

        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super(CustomGCNConv, self).get_config()
        config.update({
            "channels": self.channels,
            "activation": keras.activations.serialize(self.activation),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


# Custom Global Sum Pooling Layer
class GlobalSumPool(layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalSumPool, self).__init__(**kwargs)

    def call(self, inputs):
        # Sums features of all nodes in a graph
        return tf.reduce_sum(inputs, axis=1) # Sum over the node dimension

    def get_config(self):
        return super(GlobalSumPool, self).get_config()


def build_gnn_model(input_shape_X, input_shape_A, num_classes):
    X_input = keras.Input(shape=input_shape_X, name='node_features')
    A_input = keras.Input(shape=input_shape_A, name='adjacency_matrix', dtype=tf.float32)

    # GCN Layers with Batch Normalization and Dropout
    # Increased channels and added a deeper layer for more capacity
    x = CustomGCNConv(256, activation=None, kernel_regularizer=keras.regularizers.l2(1e-4))([X_input, A_input])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = CustomGCNConv(256, activation=None, kernel_regularizer=keras.regularizers.l2(1e-4))([x, A_input])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = CustomGCNConv(128, activation=None, kernel_regularizer=keras.regularizers.l2(1e-4))([x, A_input])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Global Pooling Layer
    x = GlobalSumPool()(x)

    # Fully Connected Layers for Classification Head
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    output = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=[X_input, A_input], outputs=output)
    return model

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix on Test Set', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix using actual class names.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(len(classes) + 2, len(classes) + 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=classes, yticklabels=classes,
                cbar=True, linewidths=.5, linecolor='black')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_accuracy_loss(history):
    """
    Plots the training and validation accuracy and loss curves.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('accuracy_loss_curves.png')
    plt.show()

def train_model_main(dataset_path):
    print(f"\n🚀 Starting training process for ISL GNN model using data from: {dataset_path}")

    # Load and preprocess data using the ISLDataset class
    # This will also save classes.txt
    dataset = ISLDataset(dataset_path)
    (x_train, a_train, y_train_true), \
    (x_val, a_val, y_val_true), \
    (x_test, a_test, y_test_true) = dataset.load_and_preprocess_data()

    num_classes = len(dataset.classes)
    if num_classes < 2:
        raise ValueError(f"Found only {num_classes} class. At least 2 classes are required for classification.")

    # Convert integer labels to one-hot encoding for Keras
    y_train_one_hot = keras.utils.to_categorical(y_train_true, num_classes=num_classes)
    y_val_one_hot = keras.utils.to_categorical(y_val_true, num_classes=num_classes)
    y_test_one_hot = keras.utils.to_categorical(y_test_true, num_classes=num_classes)

    # Build the model
    input_shape_X = x_train.shape[1:]  # (num_landmarks, num_features_per_landmark) e.g., (21, 3)
    input_shape_A = a_train.shape[1:]  # (num_landmarks, num_landmarks) e.g., (21, 21)
    model = build_gnn_model(input_shape_X, input_shape_A, num_classes)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), # Slightly reduced LR
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Callbacks for better training control
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=70, restore_best_weights=True, verbose=1), # Increased patience
        keras.callbacks.ModelCheckpoint('best_isl_gnn_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=1e-7, verbose=1), # Adjusted parameters
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    print("\nTraining the GNN model...")
    history = model.fit(
        [x_train, a_train], y_train_one_hot,
        epochs=800, # Max epochs, EarlyStopping will stop it sooner if performance plateaus
        batch_size=32,
        validation_data=([x_val, a_val], y_val_one_hot),
        callbacks=callbacks,
        verbose=1
    )

    # Load the best model saved by ModelCheckpoint for final evaluation
    try:
        best_model = tf.keras.models.load_model('best_isl_gnn_model.h5',
                                                custom_objects={'CustomGCNConv': CustomGCNConv,
                                                                'GlobalSumPool': GlobalSumPool})
        print("\n✅ Loaded best model for final evaluation.")
    except Exception as e:
        print(f"\n🚫 Could not load 'best_isl_gnn_model.h5'. Using the last trained model. Error: {e}")
        best_model = model # Fallback to the last trained model if loading fails

    # Evaluate the final (best) model on the test set
    print("\n📊 Evaluating final (best) model on the test set...")
    test_loss, test_acc = best_model.evaluate([x_test, a_test], y_test_one_hot, verbose=1)
    print(f"\n🎉 Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")

    # Generate predictions for the confusion matrix
    y_pred_probs = best_model.predict([x_test, a_test])
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Plot Confusion Matrix using actual class names
    print("\nGenerating Confusion Matrix...")
    plot_confusion_matrix(y_test_true, y_pred_classes, dataset.classes, title='Confusion Matrix on Test Set')

    # Plot Accuracy and Loss graphs
    print("\nGenerating Accuracy and Loss graphs...")
    plot_accuracy_loss(history)

    # Save the final model for prediction script
    final_model_path = "final_isl_gnn_model_full.h5"
    best_model.save(final_model_path)
    print(f"\n✅ Final (best) model saved to {final_model_path}")

    print("\nTo analyze training, run: tensorboard --logdir logs/fit")
    print("\nTo run real-time prediction, execute 'predict_realtime.py'.")

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"🔥 TensorFlow is using GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️ No GPU found, TensorFlow will use CPU. Training might be slow.")

    dataset_base_path = '/content/sign_image/imagedata-main'

    if not os.path.exists(dataset_base_path):
        print(f"Error: Dataset path '{dataset_base_path}' not found.")
        print("Please create the 'ISL_Dataset' directory and populate it with class subfolders and images.")
        print("Example structure: ISL_Dataset/A/img_01.jpg, ISL_Dataset/B/img_01.jpg, etc.")
    else:
        try:
            train_model_main(dataset_base_path)
            print(f"\n✨ Training process successfully completed! Model and class names saved.")
        except ValueError as ve:
            print(f"\n❌ Training failed: {ve}")
        except Exception as e:
            print(f"\nAn unexpected error occurred during training: {e}")