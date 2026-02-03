# Copyright (c) 2022-2024, RTE (http://www.rte-france.com)
# See AUTHORS.md
# All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the oasis project.

"""
Utility functions to manipulate the SCADA data.
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy.ndimage import median_filter


def create_output_folder(output_folder=None):
    if output_folder is None:
        script_directory = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(script_directory, "..", "default_output_folder")
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    return output_folder


def distribute_seconds_within_minute(series):
    """
    Adjust a pandas Series of datetime objects to distribute missing seconds evenly within each minute.

    Parameters:
    series (pd.Series): Series of datetime objects

    Returns:
    pd.Series: Updated Series with distributed seconds
    """
    # Initialize an empty list for the adjusted times
    adjusted_times = []

    # Group timestamps by unique minute
    grouped = series.groupby(series.dt.floor('min'))

    for minute, group in grouped:
        n = len(group)  # Number of entries for the current minute
        if n == 1 or group.iloc[0].second != 0:  # If seconds are defined, leave as is
            adjusted_times.extend(group)
        else:
            # Distribute seconds evenly within the minute
            seconds = np.linspace(0, 60, num=n, endpoint=False, dtype=int)
            adjusted_times.extend([minute + pd.Timedelta(seconds=s) for s in seconds])

    return pd.Series(adjusted_times)


def is_channel_empty(channel):
    return channel == ''


def has_too_low_max(col_values, settings):
    value_max = np.max(np.abs(col_values))
    return value_max <= settings.get_min_output_threshold(), value_max


def has_too_low_diff(col_values, settings):
    max_diff = np.max(col_values) - np.min(col_values)
    return max_diff <= settings.get_min_diff_threshold(), max_diff


def has_quantif_problem(col_values, settings):
    nb_different = len(set(col_values))
    return nb_different < settings.get_min_number_different_values(), nb_different


def too_many_na(col_values, settings):
    na_prop = np.isnan(col_values).sum() / len(col_values)
    return na_prop >= settings.get_na_threshold(), na_prop


def not_enough_samples(col_series, threshold):
    return col_series.count() < threshold, col_series.count()


def too_many_consecutive_na(col_values, settings):
    na_counts = np.isnan(col_values).astype(int)
    consecutive_counts = na_counts.cumsum()
    max_consecutive = consecutive_counts[~np.isnan(col_values)].max()
    return max_consecutive > settings.get_max_consecutive_na(), max_consecutive


def is_channel_to_remove(channel, col_values, col_values_amb, col_values_osc,
                         settings, logger):
    if is_channel_empty(channel):
        logger.warning("A column with no id has been found: the column is ignored")
        return True

    failed, val = has_too_low_max(col_values, settings)
    if failed:
        logger.warning(f"(Max abs) channel {channel} too small ({val} < {settings.get_min_output_threshold()})")
        return True

    failed, val = has_too_low_diff(col_values, settings)
    if failed:
        logger.warning(f"(Diff min/max) channel {channel} too low ({val} < {settings.get_min_diff_threshold()})")
        return True

    failed, val = has_quantif_problem(col_values, settings)
    if failed:
        logger.warning(
            f"(Quantification) channel {channel} has too few different values ({val} < {settings.get_min_number_different_values()})")
        return True

    failed, val = too_many_na(col_values, settings)
    if failed:
        logger.warning(f"(NA proportion) channel {channel} has too many NA ({val:.1%})")
        return True

    failed, val = not_enough_samples(col_values_osc, settings.get_min_nb_samples_osc())
    if failed:
        logger.warning(f"channel {channel}: not enough osc samples ({val} < {settings.get_min_nb_samples_osc()})")
        return True

    failed, val = not_enough_samples(col_values_amb, settings.get_min_nb_samples_amb())
    if failed:
        logger.warning(f"channel {channel}: not enough amb samples ({val} < {settings.get_min_nb_samples_amb()})")
        return True

    failed, val = too_many_consecutive_na(col_values, settings)
    if failed:
        logger.warning(f"channel {channel}: too many consecutive NA ({val} > {settings.get_max_consecutive_na()})")
        return True


def channel_filtering(scada_data, osc_start, osc_end, settings, logger):
    """
    The scada_data are processed in order to remove some channels which
    are irrelevant for analysis
    """
    scada_data_amb, scada_data_osc = split_windows(scada_data, osc_start, osc_end)

    channels_to_remove = []
    for channel in scada_data.columns:
        col_values = scada_data[channel]
        col_values_amb = scada_data_amb[channel]
        col_values_osc = scada_data_osc[channel]

        if is_channel_to_remove(channel, col_values, col_values_amb, col_values_osc,
                                settings, logger):
            channels_to_remove.append(channel)
            continue

        # TODO: this part with interpolation is questionnable
        na_counts = np.isnan(col_values).astype(int)
        consecutive_counts = na_counts.cumsum()
        max_consecutive = consecutive_counts[~np.isnan(col_values)].max()
        if max_consecutive > settings.get_max_consecutive_na():
            # If false, drop the entire column
            logger.warning("Too many consecutive NA values for channel {}: "
                           "this channel is ignored".format(channel))
            channels_to_remove.append(channel)
            continue
        if (max_consecutive > 0) and (max_consecutive <= settings.get_max_consecutive_na()):
            # If true, perform linear interpolation for the column
            indices = np.arange(len(col_values))
            nan_indices = indices[np.isnan(col_values)]
            col_values = col_values.copy()
            col_values[nan_indices] = np.interp(
                nan_indices,
                indices[~np.isnan(col_values)],
                col_values[~np.isnan(col_values)]
            )
            scada_data.loc[:, channel] = col_values

    scada_data = scada_data.drop(columns=channels_to_remove)
    logger.info("Number of remaining channels to analyze: " + str(scada_data.shape[1]))

    return scada_data


def is_scada_data_acceptable(scada_data):
    nb_channels = scada_data.shape[1]
    if nb_channels < 1:
        return False
    return True


def detrend(scada_data, detrending_method=0, median_filter_order=15):
    if detrending_method == 0:
        detrended_scada_data = differencing_method(scada_data)
    else:
        detrended_scada_data = median_filter_method(scada_data, median_filter_order)
    return detrended_scada_data


def differencing_method(scada_data):
    channel_data = scada_data.values
    diffs = np.diff(channel_data, axis=0)
    first_row = channel_data[0:1, :] - channel_data[1:2, :]
    detrended_scada_data_values = np.vstack([first_row, diffs])
    detrended_scada_data = pd.DataFrame(detrended_scada_data_values,
                                        index=scada_data.index, columns=scada_data.columns)
    return detrended_scada_data


def median_filter_method(scada_data, median_filter_order=15):
    if median_filter_order < 1:
        raise ValueError("Median_filter_order must be at least 1.")
    if median_filter_order > len(scada_data):
        raise ValueError("Median_filter_order cannot be larger than the length of the data.")

    channel_data = scada_data.values
    trends = np.apply_along_axis(median_filter, axis=0, arr=channel_data, size=median_filter_order)
    detrended_scada_data_values = channel_data - trends
    detrended_scada_data = pd.DataFrame(detrended_scada_data_values,
                                        index=scada_data.index, columns=scada_data.columns)
    return detrended_scada_data


def subtract_mean(detrended_scada_data):
    return detrended_scada_data - detrended_scada_data.mean()


def split_windows(detrended_scada_data, osc_start, osc_end):
    mask_osc = (detrended_scada_data.index >= osc_start) & (detrended_scada_data.index <= osc_end)
    detrended_scada_data_osc = detrended_scada_data[mask_osc]
    mask_amb = (detrended_scada_data.index < osc_start) | (detrended_scada_data.index > osc_end)
    detrended_scada_data_amb = detrended_scada_data[mask_amb]
    detrended_scada_data_osc = subtract_mean(detrended_scada_data_osc)
    detrended_scada_data_amb = subtract_mean(detrended_scada_data_amb)
    return detrended_scada_data_amb, detrended_scada_data_osc


def read_and_format_from_json(test_case_json):
    with open(test_case_json) as file:
        test_case_data = json.load(file)
    return read_and_format(test_case_data)


def read_and_format_uploaded_file(uploaded_file):
    test_case_data = json.load(uploaded_file)
    return read_and_format(test_case_data)


def read_and_format(test_case_data):
    try:
        osc_start = test_case_data["osc_start"]
        if len(osc_start.split(":")) == 2:
            osc_start = osc_start + ":00"
        osc_start = datetime.strptime(osc_start, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        raise ValueError(f"Invalid format for osc_start: {osc_start}. Expected format: '%Y-%m-%d %H:%M:%S'.") from e
    try:
        osc_end = test_case_data["osc_end"]
        if len(osc_end.split(":")) == 2:
            osc_end = osc_end + ":00"
        osc_end = datetime.strptime(osc_end, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        raise ValueError(f"Invalid format for osc_end: {osc_end}. Expected format: '%Y-%m-%d %H:%M:%S'.") from e
    if osc_end <= osc_start:
        raise ValueError(f"osc_end {osc_end} must be greater than osc_start {osc_start}.")

    scada_measurements_file = test_case_data["scada_file"]
    try:
        scada_data = pd.read_csv(scada_measurements_file, na_values=["INVA"])
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {scada_measurements_file}.")
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"File is empty: {scada_measurements_file}.") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing file: {scada_measurements_file}.") from e

    scada_data["time"] = pd.to_datetime(scada_data["time"])
    if scada_data["time"].duplicated().any():
        # likely the SCADA timestamps contain only hours and minutes
        scada_data["time"] = distribute_seconds_within_minute(scada_data["time"])
    scada_data.set_index('time', inplace=True)
    sorted_idx = scada_data.index.sort_values()
    scada_data = scada_data.loc[sorted_idx]

    return scada_data, osc_start, osc_end
