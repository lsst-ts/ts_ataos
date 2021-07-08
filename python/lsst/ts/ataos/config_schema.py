# This file is part of ts_scheduler.
#
# Developed for Vera C. Rubin Observatory Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

__all__ = ["CONFIG_SCHEMA"]

import yaml

CONFIG_SCHEMA = yaml.safe_load(
    """$schema: http://json-schema.org/draft-07/schema#
$id: https://github.com/lsst-ts/ts_ATAOS/blob/master/schema/ATAOS.yaml
# title must end with one or more spaces followed by the schema version, which must begin with "v"
title: ATAOS v3
description: Schema for ATAOS configuration files
type: object
properties:
  correction_frequency:
    description: The frequency which corrections should be applied (Hz).
    type: number
    default: 1
  m1:
    description: List of polynomial coefficients for M1 correction equation.
    type: array
    items:
      type: number
    default: [0.]
  m2:
    description: List of polynomial coefficients for M2 correction equation.
    type: array
    items:
      type: number
    default: [0.]
  hexapod_x:
    description: List of polynomial coefficients for hexapod x-correction equation.
    type: array
    items:
      type: number
    default: [0.]
  hexapod_y:
    description: List of polynomial coefficients for hexapod y-correction equation.
    type: array
    items:
      type: number
    default: [0.]
  hexapod_z:
    description: List of polynomial coefficients for hexapod z-correction equation (focus).
    type: array
    items:
      type: number
    default: [0.]
  hexapod_u:
    description: List of polynomial coefficients for hexapod u-correction equation.
    type: array
    items:
      type: number
    default: [0.]
  hexapod_v:
    description: List of polynomial coefficients for hexapod v-correction equation.
    type: array
    items:
      type: number
    default: [0.]
  hexapod_sensitivity_matrix:
    description: >-
        A matrix to map the cross terms dependencies between hexapod axis.
        For instance, if you want a correction in x-axis to result in a
        correction in u-axis to compensate for the image motion, it is possible
        to add the factor to the sensitivity matrix. By default, no cross-terms
        are added. Also, note that these affects the LUTs above so you must
        make sure to remove cross terms from the data before fitting it.
    type: array
    minItems: 6
    maxItems: 6
    items:
        type: array
        minItems: 6
        maxItems: 6
        items:
            type: number
    default:
        -
            - 1.0
            - 0.0
            - 0.0
            - 0.0
            - 0.0
            - 0.0
        -
            - 0.0
            - 1.0
            - 0.0
            - 0.0
            - 0.0
            - 0.0
        -
            - 0.0
            - 0.0
            - 1.0
            - 0.0
            - 0.0
            - 0.0
        -
            - 0.0
            - 0.0
            - 0.0
            - 1.0
            - 0.0
            - 0.0
        -
            - 0.0
            - 0.0
            - 0.0
            - 0.0
            - 1.0
            - 0.0
        -
            - 0.0
            - 0.0
            - 0.0
            - 0.0
            - 0.0
            - 1.0
  chromatic_dependence:
    description: >-
        List of polynomial coefficients for the relationship between focus
        and wavelength. Slope is in um of focus per nm.
    type: array
    items:
      type: number
    default: [0.]
  correction_tolerance:
    description: >-
        Tolerance on the correction. If the difference between the last
        applied value and the current value is inside the tolerance, no
        correction is applied.
    type: object
    properties:
      m1:
        type: number
        default: 0.
        minimum: 0.
        description: Tolerance in the M1 pressure.
      m2:
        type: number
        default: 0.
        minimum: 0.
        description: Tolerance in the M2 pressure.
      x:
        type: number
        default: 0.
        minimum: 0.
        description: Tolerance in hexapod x position.
      y:
        type: number
        default: 0.
        minimum: 0.
        description: Tolerance in hexapod y position.
      z:
        type: number
        default: 0.
        minimum: 0.
        description: Tolerance in hexapod z position.
      u:
        type: number
        default: 0.
        minimum: 0.
        description: Tolerance in hexapod u position.
      v:
        type: number
        default: 0.
        minimum: 0.
        description: Tolerance in hexapod v position.
"""
)
