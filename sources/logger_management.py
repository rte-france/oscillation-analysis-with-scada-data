# Copyright (c) 2022-2024, RTE (http://www.rte-france.com)
# See AUTHORS.md
# All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the oasis project.

"""
Utility function for the loggers.
"""

import logging
import logging.handlers


def create_logger(log_file, debug=False):
    """ This logger should contain all the information related the computation """
    logger = logging.getLogger("oasis_computation")
    log_level = logging.DEBUG if debug else logging.INFO

    if logger.hasHandlers():
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

    logger.setLevel(log_level)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def create_streamlit_logger(st_log_area, debug=False):
    """ The content of this logger is displayed in the web browser.
    This content is expected to be short and its purpose is to keep track
     of the computation progress """
    streamlit_logger = logging.getLogger("oasis_streamlit")
    log_level = logging.DEBUG if debug else logging.INFO
    if not streamlit_logger.hasHandlers():
        streamlit_logger.setLevel(log_level)
        streamlit_handler = StreamlitLoggerHandler(st_log_area)
        streamlit_logger.addHandler(streamlit_handler)
    return streamlit_logger


class StreamlitLoggerHandler(logging.Handler):
    """ Custom logging handler to send logs to a Streamlit text area """
    def __init__(self, log_area):
        super().__init__()
        self.log_area = log_area
        self.logs = ""

    def emit(self, record):
        log_entry = self.format(record)
        self.logs += log_entry + "\n"
        self.log_area.code(self.logs)

    def clean(self):
        self.logs = ""
        self.log_area.text("")


def clear_log_panel(logger):
    for handler in logger.handlers:
        if isinstance(handler, StreamlitLoggerHandler):
            handler.clean()


def clear_streamlit_logger(streamlit_logger):
    if streamlit_logger is not None:
        for handler in streamlit_logger.handlers:
            if isinstance(handler, StreamlitLoggerHandler):
                handler.clean()


def add_log_msg_debug(msg, logger, streamlit_logger=None):
    if streamlit_logger is not None:
        streamlit_logger.debug(msg)
    logger.debug(msg)


def add_log_msg_info(msg, logger, streamlit_logger=None):
    if streamlit_logger is not None:
        streamlit_logger.info(msg)
    logger.info(msg)


def add_log_msg_warning(msg, logger, streamlit_logger=None):
    if streamlit_logger is not None:
        streamlit_logger.warning(msg)
    logger.warning(msg)


def add_log_msg_error(msg, logger, streamlit_logger=None):
    if streamlit_logger is not None:
        streamlit_logger.error(msg)
    logger.error(msg)
