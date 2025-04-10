# Copyright (c) 2022-2024, RTE (http://www.rte-france.com)
# See AUTHORS.md
# All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the oasis project.

"""
Function to launch OASIS from the graphical interface.
"""

import os
import pandas as pd
from datetime import datetime
import streamlit as st
from PIL import Image
import oasis_run
from settings import Settings
from data_management import read_and_format_uploaded_file, create_output_folder
from logger_management import create_streamlit_logger


app_title = 'Oscillation Analysis with SCADA using Inferential Statistics'
st.set_page_config(page_title=app_title, layout="wide")


def create_webpage_header():
    col_1, col_2, col_3, col_4 = st.columns([14, 4, 4, 4])
    script_directory = os.path.dirname(os.path.abspath(__file__))
    with col_1:
        st.title("Anomaly Detection using SCADA data")
    with col_2:
        st.write("")  # for spacing
    with col_3:
        esic_logo_path = os.path.join(script_directory, "..", "resources", "ESIC_logo.jpeg")
        esic_logo = Image.open(esic_logo_path)
        st.image(esic_logo, use_container_width=False, width=115)
    with col_4:
        rte_logo_path = os.path.join(script_directory, "..", "resources", "rte_logo.png")
        rte_logo = Image.open(rte_logo_path)
        st.image(rte_logo, use_container_width=False, width=100)


def display_record_information(scada_data, osc_start, osc_end ):
    first_datetime = scada_data.index[0]
    last_datetime = scada_data.index[-1]
    st.info(f"First timestamp of the SCADA record: {first_datetime}  \n"
            f"Last timestamp of the SCADA record: {last_datetime}  \n"
            f"\n"
            f"Beginning of oscillations: {osc_start}  \n"
            f"End of oscillations: {osc_end}")


def create_parameter_panel(settings):
    if "display_parameter" not in st.session_state:
        st.session_state["display_parameter"] = None

    col_1, col_2, col_3, col_4 = st.columns([1, 1, 1, 3])
    with col_1:
        create_parameter_button("Filter Parameters", key="fp")
    with col_2:
        create_parameter_button("Iteration Parameters", key="ip")
    with col_3:
        create_parameter_button("Statistical Parameters", key="sp")

    print_parameters_table(settings)


def print_parameters_table(settings):
    if st.session_state["display_parameter"] == "fp":
        st.dataframe(settings.filter_params(), hide_index=True, use_container_width=True)
    elif st.session_state["display_parameter"] == "ip":
        st.dataframe(settings.iteration_params(), hide_index=True, use_container_width=True)
    elif st.session_state["display_parameter"] == "sp":
        st.dataframe(settings.statistical_params(), hide_index=True, use_container_width=True)


def create_parameter_button(name, key):
    if key not in st.session_state:
        st.session_state[key] = False

    if st.button(name):
        st.session_state[key] = not st.session_state[key]
        if st.session_state["display_parameter"] == key:
            st.session_state["display_parameter"] = None
        else:
            st.session_state["display_parameter"] = key


def create_run_algorithm_button(
    scada_data: pd.DataFrame,
    osc_start: datetime,
    osc_end: datetime,
    settings: Settings
):
    output_folder = create_output_folder()
    if st.button("Run Algorithm", type="primary"):
        st.subheader("Log")
        log_area = st.empty()
        streamlit_logger = create_streamlit_logger(log_area, debug=False)
        fig_list = oasis_run.main(scada_data, osc_start, osc_end, settings, output_folder,
                       streamlit_logger)
        if fig_list is not None:
            st.session_state["fig_list"] = fig_list

    if "fig_list" in st.session_state:
        fig_list = st.session_state["fig_list"]
        for fig in fig_list:
            st.plotly_chart(fig, use_container_width=True)


def main():
    # Initialize the settings with default values
    settings = Settings()

    # Creating the webapp
    create_webpage_header()

    uploaded_file = st.file_uploader("Upload a json file", type="json")
    if uploaded_file is not None:
        try:
            scada_data, osc_start, osc_end = read_and_format_uploaded_file(uploaded_file)
            if "scada_data" not in st.session_state:
                st.session_state["scada_data"] = scada_data
            elif not scada_data.equals(st.session_state["scada_data"]):
                if "fig_list" in st.session_state:
                    del st.session_state["fig_list"]
                st.session_state["scada_data"] = scada_data
        except FileNotFoundError as e:
            st.error("File not found error. Maybe a path problem with the csv of the SCADA data."
                     " Consider using an absolute path.")
            st.exception(e)
        except KeyError as e:
            st.error("Missing field in the jason file.")
            st.exception(e)
        display_record_information(scada_data, osc_start, osc_end)
        create_parameter_panel(settings)
        create_run_algorithm_button(scada_data, osc_start, osc_end, settings)
    else:
        if "fig_list" in st.session_state:
            del st.session_state["fig_list"]


if __name__ == "__main__":
    main()
