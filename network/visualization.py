# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:47:47 2020.

@author: pielsticker
"""
import numpy as np
import matplotlib as plt
from vis.visualization import visualize_saliency, visualize_cam
from vis.utils import utils
from keras import activations

#%%
class AttentiveResonseMap:
    def __init__(self, classifier):
        model = classifier.model
        X = classifier.X
        X_train = classifier.X_train
        X_val = classifier.X_val
        X_test = classifier.X_test
        y = classifier.y
        y_train = classifier.y_train
        y_val = classifier.y_val
        y_test = classifier.y_test

        class_idx = 0
        indices = np.where(y_test[:, class_idx] == 1.0)[0]

        # pick some random input from here.
        idx = indices[0]
        # =============================================================================
        #         # Lets sanity check the picked image.
        #         plt.rcParams['figure.figsize'] = (18, 6)
        #         plt.imshow(x_test[idx][..., 0])
        # =============================================================================

        # Utility to search for layer index by name.
        # Alternatively we can specify this as -1 since it corresponds to the last layer.
        layer_idx = utils.find_layer_idx(model, "preds")

        # Swap softmax with linear
        model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(model)

        for modifier in ["guided"]:
            grads = visualize_saliency(
                model,
                layer_idx,
                filter_indices=class_idx,
                seed_input=X_test[idx],
                backprop_modifier=modifier,
            )
            plt.figure()
            plt.title(modifier)
            plt.imshow(grads, cmap="jet")

        for class_idx in np.arange(10):
            indices = np.where(y_test[:, class_idx] == 1.0)[0]
            idx = indices[0]

            f, ax = plt.subplots(1, 2)
            ax[0].imshow(X_test[idx][..., 0])

        for i, modifier in enumerate(["guided"]):
            grads = visualize_cam(
                model,
                layer_idx,
                filter_indices=class_idx,
                seed_input=X_test[idx],
                backprop_modifier=modifier,
            )
            ax[i + 1].set_title(modifier)
            ax[i + 1].imshow(grads, cmap="jet")
