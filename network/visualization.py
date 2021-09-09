# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:47:47 2020.

@author: pielsticker
"""
import numpy as np
import matplotlib as plt
import vis

from keras import activations

#%%
class AttentiveResponseMap:
    def __init__(self, model, X, y, energies, labels):
        self.model = model
        self.X = X
        self.y = y
        self.energies = energies
        self.labels = labels

        # Swap softmax with linear in the last layer.
        self.model.layers[-1].activation = activations.linear
        self.model = vis.utils.utils.apply_modifications(model)

    def plot_saliency_maps(self, no_of_spectra=10):
        fig, ax = plt.subplots(self.X.shape[0], self.y.shape[1] + 1)

        for i in range(no_of_spectra):
            intensity = self.X[i]
            ax[i, 0] = self._plot_spectrum(ax[i, 0], intensity)

            for j in range(self.labels):
                grads = vis.visualization.visualize_saliency(
                    self.model,
                    layer_idx=-1,
                    filter_indices=j,
                    seed_input=intensity,
                    backprop_modifier="guided",
                )
            ax[i, j + 1].set_title(f"guided, label {self.labels[j]}")
            ax[i, j + 1] = ax._plot_spectrum(ax[i, j + 1], grads)

        fig.tight_layout()

    def cam_map(self, no_of_spectra=10):
        fig, ax = plt.subplots(self.shape[0], self.y.shape[1] + 1)

        for i in range(no_of_spectra):
            intensity = self.X[i]
            ax[i, 0] = self._plot_spectrum(ax[i, 0], intensity)

            for j in range(self.labels):
                grads = vis.visualization.visualize_cam(
                    self.model,
                    layer_idx=-1,
                    filter_indices=j,
                    seed_input=intensity,
                    backprop_modifier="guided",
                )
            ax[i, j + 1].set_title(f"guided, label {self.labels[j]}")
            ax[i, j + 1] = ax._plot_spectrum(ax[i, j + 1], grads)

        fig.tight_layout()

    def _plot_spectrum(self, ax, X, y, cmap="jet"):
        if not isinstance(y, list):
            y = [y]

        for yi in y:
            ax.plot(self.energies, yi)
            ax.set_xlim(
                left=np.max(self.energies), right=np.min(self.energies)
            )
            ax.set_xlabel("Binding energy (eV)")
            ax.set_ylabel("Intensity (arb. units)")

        return ax


# =============================================================================
# class LayerActivationPlot:
#     def __init__(self):
#         pass
#
#     def plot_activations_at_layers(self, model, layer_output, layer_num, num_filters, X_train):
#         plt.figure(figsize=(4, 12))
#         label = ['Mn2+', 'Mn3+', 'Mn4+']
#         valence = label[np.argmax(layer_output[-1])]
#         name = str(model.layers[layer_num]).split()[0].split('.')[3] + ' Activations- Label: '+str(valence)
#         save_name = str(name)+'_'+str(valence)+'.png'
#         if layer_num != 16:
#             label = ['Filter 1', 'Filter 2', 'Filter 3']
#             valence = label[np.argmax(layer_output[-1])]
#             name = f"Convolution Activations (Layer {str(layer_num)}"
#             save_name = name + '.png'
#         num_filters = layer_output[layer_num].shape[1]
#         offset = []
#         for i in range(num_filters):
#             offset.append(max(layer_output[layer_num][:,i]))
#             shift = 2*(sum(offset) - min(layer_output[layer_num][:,i]))
#
#             plt.plot(np.linspace(620, 660, len(layer_output[layer_num][:,i])), layer_output[layer_num][:,i]+shift, 'black', label=str(i), marker='.')
#             if layer_num != 16:
#                 plt.axhline(shift, c = 'black')
#         #plt.legend()
#         plt.xticks([])
#         plt.yticks([])
#         plt.title(name)
#         #plt.ylabel('Intensity')
#         #plt.xlabel('Energy (eV)')
#         plt.tight_layout()
#         plt.savefig(os.path.join(path_to_output, save_name))
#
# =============================================================================
