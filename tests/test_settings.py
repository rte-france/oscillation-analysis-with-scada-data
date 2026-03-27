# Copyright (c) 2022-2026, RTE (http://www.rte-france.com)
# See AUTHORS.md
# All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the oasis project.
import unittest
import os
import tempfile
import yaml
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sources")))
from settings import Settings


class TestSettings(unittest.TestCase):
    def setUp(self):
        self.config = {
            "na_threshold": {
                "value": 0.2,
                "description": "Maximum 20% NA values present in the channel data, otherwise the channel is ignored"
            },
            "min_nb_samples_amb": {
                "value": 30,
                "description": "Minimum number of non-NA values during the ambient window for a channel to be considered"
            },
            "min_nb_samples_osc": {
                "value": 30,
                "description": "Minimum number of non-NA values during the oscillation window for a channel to be considered"
            },
            "max_consecutive_na": {
                "value": 1,
                "description": "Maximum number of consecutive NA values tolerated, otherwise the channel is ignored"
            },
            "min_output_threshold": {
                "value": 10,
                "description": "If the max value of the channel is below this threshold, the channel is ignored"
            },
            "min_diff_threshold": {
                "value": 5,
                "description": "Minimum difference between min and max of the channel values for the channel to be considered"
            },
            "min_number_different_values": {
                "value": 10,
                "description": "The minimum number of different values a channel must contain. This parameter was introduced to discard channels with quantification problems"
            },
            "detrending_method": {
                "value": 0,
                "description": "0 for difference method, 1 for median filtering method"
            },
            "median_filter_order": {
                "value": 15,
                "description": "Order of the median filter applied for detrending the data. Useful only if the selected detrending method is \"median filter\""
            },
            "transition_band_starting_amplitude": {
                "value": 10,
                "description": "Amplitude of the transition band that is used at the first iteration"
            },
            "transition_band_maximal_amplitude": {
                "value": 100,
                "description": "Maximal value for the transition band amplitude for the iterations"
            },
            "transition_band_amplitude_increment": {
                "value": 5,
                "description": "Increment of transition band for iteration."
            },
            "p_value": {
                "value": 0.2,
                "description": "P value for the statistical test"
            },
            "confidence": {
                "value": 0.95,
                "description": "Confidence level for statistical test"
            },
            "lambda_transition_band_osc_window": {
                "value": 2,
                "description": "Factor applied to determine the width of the transition band for the oscillation window compared to the ambient window"
            },
            "debug": {
                "value": True,
                "description": "If \"True\", additional logs will be available in the log file"
            }
        }

        self.temp_settings = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".yaml")
        yaml.dump(self.config, self.temp_settings)
        self.temp_settings.close()

    def tearDown(self):
        os.remove(self.temp_settings.name)

    def test_initialisation_and_getters(self):
        s = Settings(self.temp_settings.name)
        self.assertEqual(s.na_threshold, 0.2)
        self.assertEqual(s.get_na_threshold(), 0.2)
        self.assertEqual(s.get_min_nb_samples_amb(), 30)
        self.assertEqual(s.get_min_nb_samples_osc(), 30)
        self.assertEqual(s.get_max_consecutive_na(), 1)
        self.assertEqual(s.get_min_output_threshold(), 10)
        self.assertEqual(s.get_min_diff_threshold(), 5)
        self.assertEqual(s.get_min_number_different_values(), 10)
        self.assertEqual(s.get_detrending_method(), 0)
        self.assertEqual(s.get_median_filter_order(), 15)
        self.assertEqual(s.get_transition_band_starting_amplitude(), 10)
        self.assertEqual(s.get_transition_band_maximal_amplitude(), 100)
        self.assertEqual(s.get_transition_band_amplitude_increment(), 5)
        self.assertTrue(s.is_debug())
        self.assertEqual(s.get_p_value(), 0.2)
        self.assertEqual(s.get_confidence(), 0.95)
        self.assertEqual(s.get_lambda_transition_band_osc_window(), 2)

    def test_repr(self):
        s = Settings(self.temp_settings.name)
        r = repr(s)
        self.assertTrue(r.startswith("Settings("))
        self.assertIn("na_threshold=0.2", r)
        self.assertIn("debug=True", r)

    def test_get_description(self):
        s = Settings(self.temp_settings.name)
        self.assertEqual(
            s.get_description("na_threshold"),
            "Maximum 20% NA values present in the channel data, otherwise the channel is ignored"
        )
        with self.assertRaises(KeyError):
            s.get_description("unknown_key")

    def test_filter_params(self):
        s = Settings(self.temp_settings.name)
        df = s.filter_params()
        self.assertEqual(df.shape, (9, 3))

        expected_params = [
            "na_threshold",
            "min_nb_samples_amb",
            "get_min_nb_samples_osc",
            "max_consecutive_na",
            "min_output_threshold",
            "min_diff_threshold",
            "min_number_different_values",
            "detrending_method",
            "median_filter_order",
        ]
        self.assertCountEqual(df["parameter"].tolist(), expected_params)

    def test_iteration_params(self):
        s = Settings(self.temp_settings.name)
        df = s.iteration_params()
        self.assertEqual(df.shape, (4, 3))

        expected_params = [
            "transition_band_starting_amplitude",
            "transition_band_maximal_amplitude",
            "transition_band_amplitude_increment",
            "debug",
        ]
        self.assertCountEqual(df["parameter"].tolist(), expected_params)

    def test_statistical_params(self):
        s = Settings(self.temp_settings.name)
        df = s.statistical_params()
        self.assertEqual(df.shape, (3, 3))

        expected_params = [
            "p_value",
            "confidence",
            "lambda_transition_band_osc_window"
        ]
        self.assertCountEqual(df["parameter"].tolist(), expected_params)


if __name__ == "__main__":
    unittest.main()