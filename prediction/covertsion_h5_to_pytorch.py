import tensorflow as tf
import torch
import numpy as np
from gpu_utils import GNNModel


def convert():
    # Load TF Model
    from model_utils import CustomGCNConv as TFGCN, GlobalSumPool as TFPool
    tf_model = tf.keras.models.load_model('final_isl_gnn_model_full.h5',
                                          custom_objects={'CustomGCNConv': TFGCN, 'GlobalSumPool': TFPool},
                                          compile=False)

    pt_model = GNNModel(num_classes=12)

    # Layer Mapping Table (TF Layer Index -> PT Layer Object)
    mapping = {
        2: pt_model.gcn1, 3: pt_model.bn1,
        6: pt_model.gcn2, 7: pt_model.bn2,
        10: pt_model.gcn3, 11: pt_model.bn3,
        15: pt_model.fc1, 16: pt_model.bn4,
        18: pt_model.fc2, 19: pt_model.bn5,
        21: pt_model.fc3
    }

    with torch.no_grad():
        for tf_idx, pt_layer in mapping.items():
            weights = tf_model.layers[tf_idx].get_weights()

            if isinstance(pt_layer, torch.nn.BatchNorm1d):
                # TF BN: [gamma, beta, mean, var] -> PT BN: [weight, bias, running_mean, running_var]
                pt_layer.weight.copy_(torch.from_numpy(weights[0]))
                pt_layer.bias.copy_(torch.from_numpy(weights[1]))
                pt_layer.running_mean.copy_(torch.from_numpy(weights[2]))
                pt_layer.running_var.copy_(torch.from_numpy(weights[3]))

            elif "custom_gcn" in tf_model.layers[tf_idx].name:
                # GCN: Weight 0 = Bias, Weight 1 = Kernel
                pt_layer.bias.copy_(torch.from_numpy(weights[0]))
                pt_layer.linear.weight.copy_(torch.from_numpy(weights[1]).t())

            else:  # Standard Dense
                pt_layer.weight.copy_(torch.from_numpy(weights[0]).t())
                pt_layer.bias.copy_(torch.from_numpy(weights[1]))

    torch.save(pt_model.state_dict(), 'final_isl_gnn_model_full.pth')
    print("✅ Conversion Complete for 12 classes.")


if __name__ == "__main__":
    convert()