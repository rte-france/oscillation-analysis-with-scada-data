# Copyright (c) 2022-2024, RTE (http://www.rte-france.com)
# See AUTHORS.md
# All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the oasis project.

"""
Main function for the detection of oscillations.

It can be run directly from the command line,
or can be called by the the graphical interface.
"""

import os
import sys
import json
from scipy.stats import binom
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from settings import Settings
from logger_management import create_logger, clear_streamlit_logger, \
    add_log_msg_debug, add_log_msg_info, add_log_msg_warning, add_log_msg_error
from data_management import channel_filtering, detrend, split_windows, is_scada_data_acceptable


def determine_min_number_successes_to_reach_confidence(n, p, confidence):
    """
    - n : Total number of trials for the window
    - p : Probability of success for each trial
    - confidence : confidence level, for instance 0.95

    Returns:
    - k : minimum number of successes that should be obtained to claim
          that there are oscillations in the window
    """
    cum = 0
    k = 0
    while cum <= confidence:
        cum += binomial(k, n, p)
        k += 1
    return k-1


def identify_suspicious_channels(
        detrended_scada_data_amb,
        detrended_scada_data_osc,
        settings,
        logger
):
    # Calculate Oscillation Window Threshold
    number_samples_oscillation_window = detrended_scada_data_osc.shape[0]
    osc_win_min = determine_min_number_successes_to_reach_confidence(
        number_samples_oscillation_window, settings.get_pmin_osc(), settings.confidence_osc)

    # Calculate Ambient Window Thresholds
    number_samples_ambient_window = detrended_scada_data_amb.shape[0]
    amb_win_max = determine_min_number_successes_to_reach_confidence(
        number_samples_ambient_window, settings.get_pmax_amb(), settings.confidence_amb)

    # Transition band loop
    # Important: depending on the dataset and the settings, it is possible that no channel
    # is highlighted for the first iterations, and that afterwards some begins to be ranked.
    # Be careful if you want to refactor the code in order to break the loop
    loop_count = 0  # just for logs, to follow what happens at each iteration
    suspicious_channels = dict()
    final_transition_band_amplitude = settings.get_transition_band_starting_amplitude()
    transition_band_current_amplitude = settings.get_transition_band_starting_amplitude()
    while (transition_band_current_amplitude <= settings.get_transition_band_maximal_amplitude()):
        ranking_factors = dict()
        for channel in detrended_scada_data_osc.columns:
            # Calculate number of transitions in the SCADA data
            number_transitions_osc = count_transitions(detrended_scada_data_osc[channel].values,
                                                       transition_band_current_amplitude)
            number_transitions_amb = count_transitions(detrended_scada_data_amb[channel].values,
                                                       transition_band_current_amplitude)

            if number_transitions_osc <= osc_win_min:
                # Not enough transitions during the oscillation window => channel discarded
                continue
            if number_transitions_amb >= amb_win_max:
                # Too many transitions during the ambient window => discarded
                continue

            # if we are still here, then the channel is suspicious as a source of forced oscillations
            ranking_factors[channel] = round(number_transitions_osc / number_samples_oscillation_window, 2)

        ranking_factors_sorted = sorted(ranking_factors.items(), key=lambda x: x[1], reverse=True)
        msg = "iteration {} - transition_band_current_amplitude = {}"\
            .format(loop_count, transition_band_current_amplitude)
        add_log_msg_debug(msg, logger)
        msg = "Ranked channels and ranking_factor: {}".format(ranking_factors_sorted)
        add_log_msg_debug(msg, logger)

        if len(ranking_factors_sorted) > 0:
            suspicious_channels = ranking_factors
            final_transition_band_amplitude = transition_band_current_amplitude

        transition_band_current_amplitude = transition_band_current_amplitude \
                                            + settings.get_transition_band_amplitude_increment()
        loop_count = loop_count + 1

    return suspicious_channels, final_transition_band_amplitude


def zero_crossings(data: np.ndarray):
    signs = np.sign(data)
    # Handle zeros explicitly: if zero, take the sign of the next non-zero element
    # Illustration of the problem: [1, 0, -1] => should be seen as 1 crossing
    signs[signs == 0] = np.sign(data[np.nonzero(data)][0]) if np.any(data != 0) else 0
    # Find where the sign changes
    crossings = np.where(np.diff(signs) != 0)[0]
    return crossings


def count_transitions(data: np.ndarray, tested_amplitude: float):
    """
    :param data: a numpy Series
    :param tested_amplitude: float (should be positive even though it would work with negative value)
    :return: number of transitions between regions

    Let's consider 3 regions:
    - R1: a value is in R1 if it is above tested_amplitude
    - R2: a value is in R2 if it is between tested_amplitude and -tested_amplitude
    - R3: a value is in R3 if it is below -tested_amplitude

    This function will count the number of transitions between regions
    """
    if tested_amplitude < 0:
        raise ValueError("Tested_amplitude = {}: should be positive.".format(tested_amplitude))

    # count transitions between R1 and another region
    data_upper = data - tested_amplitude
    loc_transition_with_R1 = zero_crossings(data_upper)

    # count transitions between R3 and another region
    data_lower = data + tested_amplitude
    loc_transition_with_R3 = zero_crossings(data_lower)

    # merging indices
    loc_transitions = np.union1d(loc_transition_with_R1, loc_transition_with_R3)
    number_of_transitions = len(loc_transitions)
    return number_of_transitions


def binomial(k, n, p):
    """
    - k : Number of successes
    - n : Total number of trials
    - p : Probability of success for each trial

    Returns:
    - Probability associated with getting exactly k successes in n trials.
    """
    return binom.pmf(k, n, p)


def plot_suspicious_channels(scada_data, detrended_scada_data,
                             osc_start, osc_end,
                             suspicious_channels, final_transition_band_amplitude,
                             output_folder):
    html_plots_file = os.path.join(output_folder, 'plots.html')
    fig_list = []
    html_images = []

    # Plot the graphs
    sorted_suspicious_channels = dict(sorted(suspicious_channels.items(), key=lambda item: item[1], reverse=True))
    for channel_to_plot in sorted_suspicious_channels:
        # Get data for actual graph
        channel_data = scada_data[channel_to_plot].values
        detrended_channel_data = detrended_scada_data[channel_to_plot].values

        min_y1, max_y1 = np.floor(np.min(channel_data)), np.ceil(np.max(channel_data))
        min_y2, max_y2 = np.floor(np.min(detrended_channel_data)), np.ceil(np.max(detrended_channel_data))

        # Create Figure with two subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            channel_to_plot + " raw", channel_to_plot + " detrended"
        ))

        # Raw Plot
        osc_start_plot = max(scada_data.index.min(), osc_start)
        osc_end_plot = min(scada_data.index.max(), osc_end)

        fig.add_trace(
            go.Scatter(x=scada_data.index, y=channel_data,
                       mode='lines', name='Raw', line=dict(color='red', width=1)),
            row=1, col=1
        )
        fig.add_shape(
            type="rect",
            x0=osc_start_plot, x1=osc_end_plot,
            y0=min_y1 - 100, y1=max_y1 - min_y1 + 200,
            fillcolor="rgba(179, 229, 229, 0.2)",
            line_width=0,
            row=1, col=1
        )
        fig.update_yaxes(title_text="MW", range=[min_y1 - 10, max_y1 + 10], row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=1)

        # Difference Plot
        fig.add_trace(
            go.Scatter(x=detrended_scada_data.index, y=detrended_channel_data,
                       mode='lines', name='Detrended', line=dict(color='blue', width=1)),
            row=1, col=2
        )
        fig.add_shape(
            type="rect",
            x0=osc_start_plot, x1=osc_end_plot,
            y0=min_y2 - 100, y1=max_y2 - min_y2 + 200,
            fillcolor="rgba(179, 229, 229, 0.2)",
            line_width=0,
            row=1, col=2
        )
        fig.add_shape(
            type="line",
            x0=scada_data.index.min(), x1=scada_data.index.max(),
            y0=final_transition_band_amplitude, y1=final_transition_band_amplitude,
            line=dict(color="Green", width=2, dash="dash"),
            row=1, col=2
        )

        fig.add_shape(
            type="line",
            x0=scada_data.index.min(), x1=scada_data.index.max(),
            y0=-final_transition_band_amplitude, y1=-final_transition_band_amplitude,
            line=dict(color="Green", width=2, dash="dash"),
            row=1, col=2
        )

        fig.update_yaxes(title_text="MW", range=[min_y2 - 10, max_y2 + 10], row=1, col=2)
        fig.update_xaxes(title_text="Time", row=1, col=2)

        # Save figure as html
        fig_list.append(fig)
        html_images.append(pio.to_html(fig, full_html=False))

    # Create html
    with open(html_plots_file, "w", encoding="utf-8") as html_file:
        html_file.write("<!DOCTYPE html>\n<html>\n<body>\n")
        html_file.write("<h1 style='text-align: center;'>Graphs</h1>\n")
        for html_fig in html_images:
            html_file.write(html_fig)
            html_file.write("<hr>\n")
        html_file.write("</body>\n</html>\n")

    return fig_list


def write_beginning_of_computation_logs(scada_data, osc_start, osc_end, logger, streamlit_logger):
    first_datetime = scada_data.index[0]
    last_datetime = scada_data.index[-1]
    add_log_msg_info("============ RUNNING ALGORITHM ============", logger, streamlit_logger)
    add_log_msg_info("", logger)
    add_log_msg_info("First datetime of the SCADA record: " + first_datetime.strftime("%Y%m%d_%Hh%Mm%S"), logger)
    add_log_msg_info("Last datetime of the SCADA record: " + last_datetime.strftime("%Y%m%d_%Hh%Mm%S"), logger)
    add_log_msg_info("Beginning of oscillations: " + osc_start.strftime("%Y%m%d_%Hh%Mm%S"), logger)
    add_log_msg_info("End of oscillations: " + osc_end.strftime("%Y%m%d_%Hh%Mm%S"), logger)
    add_log_msg_info("Number of channels to analyze: " + str(scada_data.shape[1]), logger)
    add_log_msg_info("", logger)


def write_no_channel_error_logs(logger, streamlit_logger):
    msg = "The SCADA data do not contain any channel to analyze: stopping the algorithm"
    add_log_msg_error(msg, logger, streamlit_logger)
    add_log_msg_info("", logger)
    msg = "============ END OF COMPUTATION ==========="
    add_log_msg_info(msg, logger, streamlit_logger)


def write_processing_outputs_logs(suspicious_channels, logger, streamlit_logger):
    add_log_msg_info("", logger)
    add_log_msg_info("PROCESSING OUTPUTS", logger, streamlit_logger)
    if len(suspicious_channels) == 0:
        msg = "No channel flagged during the analysis."
        add_log_msg_warning(msg, logger, streamlit_logger)
    else:
        if len(suspicious_channels) == 1:
            suspicious_channel_id = next(iter(suspicious_channels))
            msg = "=> The suspicious channel is {}".format(suspicious_channel_id)
            add_log_msg_info(msg, logger, streamlit_logger)
        else:
            suspicious_channels_sorted = sorted(suspicious_channels.items(), key=lambda x: x[1], reverse=True)
            msg = "=> The suspicious channels are"
            for channel, ranking_factor in suspicious_channels_sorted:
                msg += " {} (ranking factor of {}) ".format(channel, ranking_factor)
            add_log_msg_info(msg, logger, streamlit_logger)


def write_end_of_computation_logs(output_folder, logger, streamlit_logger):
    msg = "Output files can be found in " + str(output_folder)
    add_log_msg_info(msg, logger, streamlit_logger)
    add_log_msg_info("", logger)
    msg = "============ END OF COMPUTATION ==========="
    add_log_msg_info(msg, logger, streamlit_logger)


def end_computation_unacceptable_scada_data(output_json, output_summary_dict, logger, streamlit_logger):
    output_summary_dict["computation_status"] = "NOK"
    save_output_summary(output_json, output_summary_dict)
    write_no_channel_error_logs(logger, streamlit_logger)


def save_output_summary(output_json, output_summary_dict):
    json.dump(output_summary_dict, open(output_json, 'w'), indent=2)


def main(
        scada_data: pd.DataFrame,
        osc_start: datetime, osc_end: datetime,
        settings: Settings,
        output_folder: str,
        streamlit_logger=None
):
    # Init some variables
    output_json = os.path.join(output_folder, "oasis_output.json")
    output_summary_dict = dict()
    log_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
    logger = create_logger(log_file, debug=settings.is_debug())
    clear_streamlit_logger(streamlit_logger)

    # some logs
    write_beginning_of_computation_logs(scada_data, osc_start, osc_end, logger, streamlit_logger)

    # prepare the data
    add_log_msg_info("PREPROCESSING THE SCADA DATA", logger, streamlit_logger)
    scada_data = channel_filtering(scada_data, osc_start, osc_end, settings, logger)
    if not is_scada_data_acceptable(scada_data):
        end_computation_unacceptable_scada_data(
            output_json, output_summary_dict, logger, streamlit_logger)
        return
    detrended_scada_data = detrend(scada_data, settings.get_detrending_method(),
                                   settings.get_median_filter_order())
    detrended_scada_data_amb, detrended_scada_data_osc = split_windows(detrended_scada_data,
                                                                       osc_start, osc_end)

    # Identifying the likely sources of forced oscillations
    add_log_msg_info("", logger)
    msg = "IDENTIFYING THE MOST LIKELY SOURCES OF FORCED OSCILLATIONS"
    add_log_msg_info(msg, logger, streamlit_logger)
    suspicious_channels, final_transition_band_amplitude = identify_suspicious_channels(
        detrended_scada_data_amb, detrended_scada_data_osc, settings, logger)
    output_summary_dict["suspicious_channels"] = suspicious_channels

    # Processing outputs
    write_processing_outputs_logs(suspicious_channels, logger, streamlit_logger)
    fig_list = plot_suspicious_channels(scada_data, detrended_scada_data, osc_start, osc_end,
                             suspicious_channels, final_transition_band_amplitude, output_folder)
    save_output_summary(output_json, output_summary_dict)
    write_end_of_computation_logs(output_folder, logger, streamlit_logger)

    return fig_list


if __name__ == "__main__":
    import argparse
    from data_management import read_and_format_from_json

    parser = argparse.ArgumentParser(
        description="Oscillation Analysis with SCADA using Inferential Statistics"
    )
    parser.add_argument(
        '--input-file',
        dest='input_file',
        required=True,
        help='json file that contains a link to the SCADA data as well as the start time and end time of oscillations'
    )
    parser.add_argument(
        '--output-folder',
        dest=None,
        help="optional - folder where the outputs are dropped"
    )
    parser.add_argument(
        '--settings-file',
        dest='settings_file',
        default=None,
        help="optional - file that contains the settings for the algorithm"
    )
    args = parser.parse_args()
    input_file = args.input_file
    settings_file = args.settings_file
    output_folder = args.output_folder

    # Testing input data
    if not os.path.isfile(input_file):
        print(f"Error: The input file '{input_file}' does not exist.")
        exit(1)
    if settings_file and not os.path.isfile(settings_file):
        print(f"Error: The settings file '{settings_file}' does not exist.")
        exit(1)

    try:
        scada_data, osc_start, osc_end = read_and_format_from_json(input_file)
    except FileNotFoundError as e:
        print("File not found error. Maybe a path problem with the csv of the SCADA data."
                 " Consider using an absolute path.")
        print(e)
        sys.exit(1)
    except KeyError as e:
        print("Missing field in the json file:", e)
        sys.exit(1)

    # output_folder
    script_directory = os.path.dirname(os.path.abspath(__file__))
    if output_folder is None:
        output_folder = os.path.join(script_directory, "..", "default_output_folder")

    # load Settings
    if settings_file is None:
        settings = Settings()
    else:
        settings = Settings(settings_file)

    # run
    main(scada_data, osc_start, osc_end, settings, output_folder)
