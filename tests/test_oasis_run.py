# Copyright (c) 2022-2026, RTE (http://www.rte-france.com)
# See AUTHORS.md
# All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the oasis project.
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sources")))
from oasis_run import (
    binomial,
    zero_crossings,
    count_transitions,
    determine_min_number_successes_to_reach_confidence,
    identify_suspicious_channels,
)


class DummySettings:
    def get_pmin_osc(self): return 0.2
    def get_pmax_amb(self): return 0.1
    def get_transition_band_starting_amplitude(self): return 2
    def get_transition_band_maximal_amplitude(self): return 6
    def get_transition_band_amplitude_increment(self): return 2
    @property
    def confidence_osc(self): return 0.9
    @property
    def confidence_amb(self): return 0.9


class DummyLogger:
    def warning(self, *args, **kwargs): pass  # dummy, for testing purposes
    def info(self, *args, **kwargs): pass  # dummy, for testing purposes
    def debug(self, *args, **kwargs): pass  # dummy, for testing purposes


class TestOasisRun(unittest.TestCase):

    def test_binomial(self):
        self.assertAlmostEqual(binomial(3, 6, 0.5), 0.3125)

    def test_zero_crossings(self):
        arr = np.array([1, 0, -2, 3, -1])
        crossings = zero_crossings(arr)
        self.assertTrue(isinstance(crossings, np.ndarray))

    def test_count_transitions(self):
        arr = np.array([5, 0, -5, 5, -5])
        res = count_transitions(arr, 4)
        self.assertTrue(isinstance(res, int))
        with self.assertRaises(ValueError):
            count_transitions(arr, -1)

    def test_determine_min_number_successes_to_reach_confidence(self):
        k = determine_min_number_successes_to_reach_confidence(5, 0.4, 0.9)
        self.assertTrue(isinstance(k, int))
        self.assertTrue(k >= 0)

    def test_identify_suspicious_channels(self):
        idx = pd.date_range("2024-08-27", periods=8, freq="min")
        data_osc = pd.DataFrame({
            "chanA": np.linspace(0, 9, 8),
            "chanB": np.ones(8),
        }, index=idx)
        data_amb = pd.DataFrame({
            "chanA": np.linspace(0, 9, 8),
            "chanB": np.zeros(8)
        }, index=idx)
        settings = DummySettings()
        logger = DummyLogger()
        suspicious_channels, final_amp = identify_suspicious_channels(data_amb, data_osc, settings, logger)
        self.assertTrue(isinstance(suspicious_channels, dict))
        self.assertTrue(final_amp in [settings.get_transition_band_starting_amplitude(),
                                      settings.get_transition_band_maximal_amplitude()])


if __name__ == "__main__":
    unittest.main()
