import tensorflow as tf
from model_utils import CustomGCNConv, GlobalSumPool  # Use your TF definitions


def debug_tf_model(h5_path):
    print("--- TensorFlow Model Diagnostic ---")
    try:
        model = tf.keras.models.load_model(h5_path,
                                           custom_objects={'CustomGCNConv': CustomGCNConv,
                                                           'GlobalSumPool': GlobalSumPool},
                                           compile=False)
        model.summary()

        print("\n--- Detailed Weight Map ---")
        for i, layer in enumerate(model.layers):
            weights = layer.get_weights()
            if weights:
                print(f"Layer {i}: {layer.name} ({type(layer).__name__})")
                for j, w in enumerate(weights):
                    print(f"  -> Weight {j} Shape: {w.shape}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    debug_tf_model('final_isl_gnn_model_full.h5')