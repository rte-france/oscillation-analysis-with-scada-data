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
import logging
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sources")))
from sources.logger_management import (
    create_logger, clear_log_panel, clear_streamlit_logger,
    StreamlitLoggerHandler, add_log_msg_info, add_log_msg_debug,
    add_log_msg_warning, add_log_msg_error
)


class DummyLogArea:
    def __init__(self):
        self.last_code = None
        self.last_text = None
    def code(self, text):
        self.last_code = text
    def text(self, text):
        self.last_text = text


class TestLoggerManagement(unittest.TestCase):

    def test_create_logger_and_log(self):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            path = tf.name
        logger = create_logger(path, debug=True)

        # StreamHandler is removed to not pollute the console output
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

        logger.debug("msg1")
        logger.info("msg2")
        logger.warning("msg3")
        logger.error("msg4")
        logger.handlers[0].flush()
        with open(path) as f:
            logs = f.read()
        self.assertIn("msg1", logs)
        self.assertIn("msg2", logs)
        self.assertIn("msg3", logs)
        self.assertIn("msg4", logs)
        os.remove(path)

    def test_streamlit_logger_handler_emit_and_clean(self):
        log_area = DummyLogArea()
        handler = StreamlitLoggerHandler(log_area)
        record = logging.LogRecord("t", logging.INFO, None, None, "hello", None, None)
        handler.emit(record)
        self.assertIn("hello", log_area.last_code)
        handler.clean()
        self.assertEqual(log_area.last_text, "")

    def test_clear_log_panel_and_streamlit_logger(self):
        log_area = DummyLogArea()
        handler = StreamlitLoggerHandler(log_area)
        logger = logging.getLogger("test_logger__" + str(id(self)))
        logger.addHandler(handler)
        clear_log_panel(logger)
        self.assertEqual(log_area.last_text, "")

        logger2 = logging.getLogger("test_logger2__" + str(id(self)))
        handler2 = StreamlitLoggerHandler(log_area)
        logger2.addHandler(handler2)
        clear_streamlit_logger(logger2)
        self.assertEqual(log_area.last_text, "")

    def test_add_log_msg_functions(self):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            path = tf.name
        logger = create_logger(path, debug=True)

        # StreamHandler is removed to not pollute the console output
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

        add_log_msg_info("INFO_MSG", logger)
        add_log_msg_debug("DEBUG_MSG", logger)
        add_log_msg_warning("WARN_MSG", logger)
        add_log_msg_error("ERROR_MSG", logger)
        logger.handlers[0].flush()
        with open(path) as f:
            logs = f.read()
        for msg in ["INFO_MSG", "DEBUG_MSG", "WARN_MSG", "ERROR_MSG"]:
            self.assertIn(msg, logs)
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
