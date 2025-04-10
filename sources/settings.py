# Copyright (c) 2022-2024, RTE (http://www.rte-france.com)
# See AUTHORS.md
# All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the oasis project.

"""
Class that contain the parameters for the analysis of the signals.
More information can be found in resources/settings.yaml or in the reference paper.

the main families of parameters are:
- filter parameters: determine which channels can be dismissed
- iteration parameters: concern the loop with the transition band
- statistical parameters: to assess whether or not there are oscillations
"""

import os
import pandas as pd
import yaml


DEFAULT_SETTINGS = os.path.join(os.path.dirname(__file__), "..", "resources", "settings.yaml")


class Settings:
    def __init__(self, settings_file=None):
        if settings_file is None:
            settings_file = DEFAULT_SETTINGS
        with open(settings_file, 'r') as file:
            self._config = yaml.safe_load(file)
        for key, value in self._config.items():
            setattr(self, key, value['value'])

    def __repr__(self):
        params = [f"{key}={getattr(self, key)}" for key in self._config]
        return f"Settings({', '.join(params)})"

    def get_description(self, key):
        if key in self._config and 'description' in self._config[key]:
            return self._config[key]['description']
        else:
            raise KeyError(f"No description available for parameter '{key}'")

    # Data filtering and detrending parameters
    def get_NA_threshold(self):
        return self.NA_threshold

    def get_min_nb_samples_amb(self):
        return self.min_nb_samples_amb

    def get_min_nb_samples_osc(self):
        return self.min_nb_samples_osc

    def get_max_consecutive_NA(self):
        return self.max_consecutive_NA

    def get_min_output_threshold(self):
        return self.min_output_threshold

    def get_min_diff_threshold(self):
        return self.min_diff_threshold

    def get_detrending_method(self):
        return self.detrending_method

    def get_median_filter_order(self):
        return self.median_filter_order

    # Iteration parameters
    def get_transition_band_starting_amplitude(self):
        return self.transition_band_starting_amplitude

    def get_transition_band_maximal_amplitude(self):
        return self.transition_band_maximal_amplitude

    def get_transition_band_amplitude_increment(self):
        return self.transition_band_amplitude_increment

    def is_debug(self):
        return self.debug

    # Statistical parameters
    def get_pmin_osc(self):
        return self.pmin_osc

    def get_confidence_osc(self):
        return self.confidence_osc

    def get_pmax_amb(self):
        return self.pmax_amb

    def get_confidence_amb(self):
        return self.confidence_amb

    def filter_params(self):
        data = {"parameter": ["NA_threshold", "min_nb_samples_amb", "get_min_nb_samples_osc", "max_consecutive_NA",
                              "min_output_threshold", "min_diff_threshold",
                              "detrending_method", "median_filter_order"],
                "value": [self.get_NA_threshold(),
                          self.get_min_nb_samples_amb(),
                          self.get_min_nb_samples_osc(),
                          self.get_max_consecutive_NA(),
                          self.get_min_output_threshold(),
                          self.get_min_diff_threshold(),
                          self.get_detrending_method(),
                          self.get_median_filter_order()],
                "description": [self.get_description("NA_threshold"),
                                self.get_description("min_nb_samples_amb"),
                                self.get_description("min_nb_samples_osc"),
                                self.get_description("max_consecutive_NA"),
                                self.get_description("min_output_threshold"),
                                self.get_description("min_diff_threshold"),
                                self.get_description("detrending_method"),
                                self.get_description("median_filter_order")]}
        filter_parameters = pd.DataFrame(data)
        return filter_parameters

    def iteration_params(self):
        data = {"parameter": ["transition_band_starting_amplitude", "transition_band_maximal_amplitude",
                              "transition_band_amplitude_increment", "debug"],
                "value": [self.get_transition_band_starting_amplitude(),
                          self.get_transition_band_maximal_amplitude(),
                          self.get_transition_band_amplitude_increment(),
                          self.is_debug()],
                "description": [self.get_description("transition_band_starting_amplitude"),
                                self.get_description("transition_band_maximal_amplitude"),
                                self.get_description("transition_band_amplitude_increment"),
                                self.get_description("debug")]}
        iteration_parameters = pd.DataFrame(data)
        iteration_parameters["value"] = iteration_parameters["value"].astype(str)
        return iteration_parameters


    def statistical_params(self):
        data = {"parameter": ["pmin_osc", "confidence_osc", "pmax_amb", "confidence_amb"],
                "value": [self.get_pmin_osc(),
                          self.get_confidence_osc(),
                          self.get_pmax_amb(),
                          self.get_confidence_amb()],
                "description": [self.get_description("pmin_osc"),
                                self.get_description("confidence_osc"),
                                self.get_description("pmax_amb"),
                                self.get_description("confidence_amb")]}
        statistical_parameters = pd.DataFrame(data)
        return statistical_parameters


