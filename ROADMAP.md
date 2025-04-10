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

# Roadmap

This file lists the main features that are envisioned for future releases.

## Improving data preparation: NaN management

If a channel has not too many NaN, they will be replaced using 
linear interpolation. As the principle of the algorithm is to
count transitions between regions, this NaN replacement strategy
is questionnable.

## Enable to change the settings from the web browser

For difficult cases, it would be convenient to be able to change the
settings from the web browser, without editing the default settings
file.