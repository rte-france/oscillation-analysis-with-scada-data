# Copyright (c) 2022-2026, RTE (http://www.rte-france.com)
# See AUTHORS.md
# All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the oasis project.
import unittest
import sys
import os
import yaml
import tempfile
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sources")))
from oasis_run import main
from settings import Settings
from data_management import read_and_format_from_json


class TestTemplateTestCase(unittest.TestCase):

    def test_main(self):
        settings_file = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "resources",
            "settings.yaml")
        )
        scada_data_json = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "nrt", "template_test_case",
            "template_test_case.json")
        )

        settings = Settings(settings_file)
        scada_data, osc_start, osc_end = read_and_format_from_json(scada_data_json)

        with tempfile.TemporaryDirectory() as output_folder:
            main(scada_data, osc_start, osc_end, settings, output_folder)
            output_files = os.listdir(output_folder)
            expected_output_files = [
                "oasis_output.json",
                "plots.html",
                "oasis_run.log"
            ]
            self.assertCountEqual(output_files, expected_output_files)
            output_json = os.path.join(output_folder, "oasis_output.json")
            with open(output_json, 'r') as f:
                output_results = yaml.safe_load(f)
                suspicious_channels = output_results["suspicious_channels"]
                expected_suspicious_channels = {'P_23': 0.36}
                self.assertEqual(suspicious_channels, expected_suspicious_channels)


if __name__ == "__main__":
    unittest.main()

