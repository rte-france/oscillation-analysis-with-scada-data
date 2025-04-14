<!-- 
    # Copyright (c) 2022-2024, RTE (http://www.rte-france.com)
    # See AUTHORS.md
    # All rights reserved.
    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, you can obtain one at http://mozilla.org/MPL/2.0/.
    # SPDX-License-Identifier: MPL-2.0
    # This file is part of the oasis project.
-->

# OASIS - Oscillation Analysis with SCADA data using Inferential Statistics

Python implementation of a method to locate the source of forced oscillations in a power system.

## Table of Contents

- [About](#about)
- [Contributors](#contributors)
- [License](#license)
- [Reference paper](#reference-paper)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  * [Inputs format](#inputs-format)
  * [Settings](#settings)
  * [Using the command line](#using-the-command-line)
  * [Using the web interface](#using-the-web-interface)  
  * [Outputs](#outputs) 

## About

Interarea oscillations are emerging as serious operational 
concerns in modern power systems because of changing 
intermittent generation patterns and unusual transmission power 
flows. Moreover, forced oscillations can interact with natural 
electromechanical modes and the resonant oscillations can be 
observed in wide regions of the interconnections 
(e.g. January 11, 2019 eastern system event). 
Source location of such resonant forced oscillation events 
is especially challenging. Oscillations may be observed using synchrophasors, 
but their installation costs and communication requirements are such that
the power grid observability using synchrophasors remains limited.

On the other hand, SCADA measurements have been in implementation
since the 1970s and they outnumber the synchrophasors. 
Specifically, SCADA measurements are available at almost every 
synchronous generator as well as at the bulk interconnection 
interface for most of the renewable generators in the power grid. 

Since SCADA data is sampled asynchronously and in a somewhat random fashion,
they may be sufficient in order to locate the source of forced
ocillation.

The code in this github project is an implementation of a statistical
method for localizing the source of forced oscillations in a power system
using SCADA data.

## Contributors

This code has been developped by [Washington State University](https://school.eecs.wsu.edu/)
and [RTE](https://www.rte-france.com/) (French Transmission System Operator).
 
The main contributors are listed in *AUTHORS.md*.

## License

This project is licensed under the terms of the 
[Mozilla Public License V2.0](http://mozilla.org/MPL/2.0). 
See [LICENSE](LICENSE.txt) for more information.

## Reference paper

If you use this implementation in your work or research, 
it would be appreciated if you could quote the following document 
in your publications or presentations:

[Oscillation Analysis with SCADA using Inferential Statistics (OASIS)](https://documents.pserc.wisc.edu/documents/publications/reports/2024_reports/S_103G_Final_Report.pdf)

## Requirements

* Python >= 3.8.10

## Installation

It is recommended to install the package and its dependencies 
within a python virtual environment: 

```bash
python3 -m venv venv
source venv/bin/activate
```

Make sure that `pip` is up-to-date:
```bash
pip install --upgrade pip
```

Install the necessary libraries:
```bash
pip install -r requirements.txt
```

## Usage

### Input format

The test case data are expected to be provided as a json file:

```json
{
    "osc_start": "20210430_04h53m00",
    "osc_end": "20210430_05h07m00",
    "scada_file": "path/to/scada_data.csv"
}
```

The scada_file should be either the absolute path, or the
relative path from the working directory.

Its content is expected to be formatted like below, using the comma
as separator. 

|time|P1|P2| 
| :--- |:---: | :---: | 
20210411 1h45|0|-336| 
20210411 1h45|0|-338| 

### Settings

If using the web interface, the settings will be read from 
the default settings file : `misc/settings.yaml`.

If using the command line interface, it is possible to specify
a different custom settings file.

However, these values must be modified with caution.

### Using the command line

The command line has the following form:
```bash
python3 sources/oasis_run.py --input-file test_cases/template_test_case/template_test_case.json
```

In addition to **--input-file** , it is possible to specify two other parameters:
* **--output-folder** (optional): the folder where the outputs will be written.
By default they will be written in `default_output_folder/` within the OASIS project folder.
* **--settings-file** (optional): a yaml file describing the settings. 
The default file is `misc/settings.yaml`.

### Using the web interface

* Call _streamlit_ from the command line in order to open a web browser:
```bash
streamlit run sources/oasis_gui.py
```
* Load the json file that contains the SCADA data using the **Browse files** button.
* The various parameters that are being used in the algorithm can be checked (optional)
* Click the **Run algorithm** button.

<p align="center">
<img src="resources/gui_screenshot.png">
</p>

### Outputs

The following files will be found in the output folder:
* A log file (.log)
* The plots of the SCADA values for the channels that have been evaluated
 as suspicious (.html)
* A json file (.json) that provides useful information when using OASIS in an automated
environment
