#
# Copyright the xpsdeeplearning authors.
#
# This file is part of xpsdeeplearning.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Plot and save all plots for publication.
"""

import os

import class_distribution_plot
import multiple_elements_loss_examples
import multiple_elements_peak_fitting_histograms
import multiple_elements_references
import sample_size_effect
import sim_values_effect
import sim_visualization
import simulate_dataset_for_fit_comparison
import single_elements_cnn_histograms
import single_elements_fit_comparison_histograms
import single_elements_loss
import single_elements_peak_fitting
import single_elements_references
import single_examples
import single_spectrum
import uncertainty
import window


def run_all():
    """Run main plot function of all pubplots scripts."""

    modules = [
        class_distribution_plot,
        multiple_elements_loss_examples,
        multiple_elements_peak_fitting_histograms,
        multiple_elements_references,
        sample_size_effect,
        sim_values_effect,
        sim_visualization,
        # simulate_dataset_for_fit_comparison,
        single_elements_cnn_histograms,
        single_elements_fit_comparison_histograms,
        single_elements_loss,
        single_elements_peak_fitting,
        single_elements_references,
        single_examples,
        # single_spectrum,
        uncertainty,
        window,
    ]

    for module in modules:
        module.main()


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    run_all()
