#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2013-2014 ankostis@gmail.com
#
# This file is part of wltc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
'''wltc.classes.class2 module: WLTC-class data for gear-shift calculator.

Data below extracted from the specs and prepared with the followinf python scripts
found inside the source-distribution:

* ``./util/printwltcclass.py``

* ``./util/csvcolumns8to2.py``
'''
def class_data():
    data = {
        'parts': [[0, 589], [590, 1022], [1023, 1477], [1478, 1800]],
        'downscale': {
            'phases':       [1520, 1725, 1742],     ## Note: Start end end +1 from specs.
            'p_max_values': [1574, 109.9, 0.36],    ## t, V(Km/h), Accel(m/s2)
            'factor_coeffs': [None,                 ## r0, a1, b1
                              [1, .6, -.6]],
            'v_max_split': 105,                     ## V (Km/h), >
        },
        'cycle': [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2, 2.6, 4.9,
            7.3, 9.4, 11.4, 12.7, 13.3, 13.4, 13.3, 13.1, 12.5, 11.1, 8.9, 6.2, 3.8, 1.8, 0.0, 0.0,
            0.0, 0.0, 1.5, 2.8, 3.6, 4.5, 5.3, 6.0, 6.6, 7.3, 7.9, 8.6, 9.3, 10.0, 10.8, 11.6,
            12.4, 13.2, 14.2, 14.8, 14.7, 14.4, 14.1, 13.6, 13.0, 12.4, 11.8, 11.2, 10.6, 9.9, 9.0, 8.2,
            7.0, 4.8, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 1.4, 2.3, 3.5, 4.7,
            5.9, 7.4, 9.2, 11.7, 13.5, 15.0, 16.2, 16.8, 17.5, 18.8, 20.3, 22.0, 23.6, 24.8, 25.6, 26.3,
            27.2, 28.3, 29.6, 30.9, 32.2, 33.4, 35.1, 37.2, 38.7, 39.0, 40.1, 40.4, 39.7, 36.8, 35.1, 32.2,
            31.1, 30.8, 29.7, 29.4, 29.0, 28.5, 26.0, 23.4, 20.7, 17.4, 15.2, 13.5, 13.0, 12.4, 12.3, 12.2,
            12.3, 12.4, 12.5, 12.7, 12.8, 13.2, 14.3, 16.5, 19.4, 21.7, 23.1, 23.5, 24.2, 24.8, 25.4, 25.8,
            26.5, 27.2, 28.3, 29.9, 32.4, 35.1, 37.5, 39.2, 40.5, 41.4, 42.0, 42.5, 43.2, 44.4, 45.9, 47.6,
            49.0, 50.0, 50.2, 50.1, 49.8, 49.4, 48.9, 48.5, 48.3, 48.2, 47.9, 47.1, 45.5, 43.2, 40.6, 38.5,
            36.9, 35.9, 35.3, 34.8, 34.5, 34.2, 34.0, 33.8, 33.6, 33.5, 33.5, 33.4, 33.3, 33.3, 33.2, 33.1,
            33.0, 32.9, 32.8, 32.7, 32.5, 32.3, 31.8, 31.4, 30.9, 30.6, 30.6, 30.7, 32.0, 33.5, 35.8, 37.6,
            38.8, 39.6, 40.1, 40.9, 41.8, 43.3, 44.7, 46.4, 47.9, 49.6, 49.6, 48.8, 48.0, 47.5, 47.1, 46.9,
            45.8, 45.8, 45.8, 45.9, 46.2, 46.4, 46.6, 46.8, 47.0, 47.3, 47.5, 47.9, 48.3, 48.3, 48.2, 48.0,
            47.7, 47.2, 46.5, 45.2, 43.7, 42.0, 40.4, 39.0, 37.7, 36.4, 35.2, 34.3, 33.8, 33.3, 32.5, 30.9,
            28.6, 25.9, 23.1, 20.1, 17.3, 15.1, 13.7, 13.4, 13.9, 15.0, 16.3, 17.4, 18.2, 18.6, 19.0, 19.4,
            19.8, 20.1, 20.5, 20.2, 18.6, 16.5, 14.4, 13.4, 12.9, 12.7, 12.4, 12.4, 12.8, 14.1, 16.2, 18.8,
            21.9, 25.0, 28.4, 31.3, 34.0, 34.6, 33.9, 31.9, 30.0, 29.0, 27.9, 27.1, 26.4, 25.9, 25.5, 25.0,
            24.6, 23.9, 23.0, 21.8, 20.7, 19.6, 18.7, 18.1, 17.5, 16.7, 15.4, 13.6, 11.2, 8.6, 6.0, 3.1,
            1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4, 3.2, 5.6, 8.1, 10.3, 12.1, 12.6, 13.6,
            14.5, 15.6, 16.8, 18.2, 19.6, 20.9, 22.3, 23.8, 25.4, 27.0, 28.6, 30.2, 31.2, 31.2, 30.7, 29.5,
            28.6, 27.7, 26.9, 26.1, 25.4, 24.6, 23.6, 22.6, 21.7, 20.7, 19.8, 18.8, 17.7, 16.6, 15.6, 14.8,
            14.3, 13.8, 13.4, 13.1, 12.8, 12.3, 11.6, 10.5, 9.0, 7.2, 5.2, 2.9, 1.2, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.4, 2.5, 5.2, 7.9, 10.3, 12.7, 15.0, 17.4, 19.7, 21.9, 24.1, 26.2, 28.1, 29.7, 31.3,
            33.0, 34.7, 36.3, 38.1, 39.4, 40.4, 41.2, 42.1, 43.2, 44.3, 45.7, 45.4, 44.5, 42.5, 39.5, 36.5,
            33.5, 30.4, 27.0, 23.6, 21.0, 19.5, 17.6, 16.1, 14.5, 13.5, 13.7, 16.0, 18.1, 20.8, 21.5, 22.5,
            23.4, 24.5, 25.6, 26.0, 26.5, 26.9, 27.3, 27.9, 30.3, 33.2, 35.4, 38.0, 40.1, 42.7, 44.5, 46.3,
            47.6, 48.8, 49.7, 50.6, 51.4, 51.4, 50.2, 47.1, 44.5, 41.5, 38.5, 35.5, 32.5, 29.5, 26.5, 23.5,
            20.4, 17.5, 14.5, 11.5, 8.5, 5.6, 2.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6, 3.6, 6.3, 9.0, 11.8, 14.2, 16.6,
            18.5, 20.8, 23.4, 26.9, 30.3, 32.8, 34.1, 34.2, 33.6, 32.1, 30.0, 27.5, 25.1, 22.8, 20.5, 17.9,
            15.1, 13.4, 12.8, 13.7, 16.0, 18.1, 20.8, 23.7, 26.5, 29.3, 32.0, 34.5, 36.8, 38.6, 39.8, 40.6,
            41.1, 41.9, 42.8, 44.3, 45.7, 47.4, 48.9, 50.6, 52.0, 53.7, 55.0, 56.8, 58.0, 59.8, 61.1, 62.4,
            63.0, 63.5, 63.0, 62.0, 60.4, 58.6, 56.7, 55.0, 53.7, 52.7, 51.9, 51.4, 51.0, 50.7, 50.6, 50.8,
            51.2, 51.7, 52.3, 53.1, 53.8, 54.5, 55.1, 55.9, 56.5, 57.1, 57.8, 58.5, 59.3, 60.2, 61.3, 62.4,
            63.4, 64.4, 65.4, 66.3, 67.2, 68.0, 68.8, 69.5, 70.1, 70.6, 71.0, 71.6, 72.2, 72.8, 73.5, 74.1,
            74.3, 74.3, 73.7, 71.9, 70.5, 68.9, 67.4, 66.0, 64.7, 63.7, 62.9, 62.2, 61.7, 61.2, 60.7, 60.3,
            59.9, 59.6, 59.3, 59.0, 58.6, 58.0, 57.5, 56.9, 56.3, 55.9, 55.6, 55.3, 55.1, 54.8, 54.6, 54.5,
            54.3, 53.9, 53.4, 52.6, 51.5, 50.2, 48.7, 47.0, 45.1, 43.0, 40.6, 38.1, 35.4, 32.7, 30.0, 27.5,
            25.3, 23.4, 22.0, 20.8, 19.8, 18.9, 18.0, 17.0, 16.1, 15.5, 14.4, 14.9, 15.9, 17.1, 18.3, 19.4,
            20.4, 21.2, 21.9, 22.7, 23.4, 24.2, 24.3, 24.2, 24.1, 23.8, 23.0, 22.6, 21.7, 21.3, 20.3, 19.1,
            18.1, 16.9, 16.0, 14.8, 14.5, 13.7, 13.5, 12.9, 12.7, 12.5, 12.5, 12.6, 13.0, 13.6, 14.6, 15.7,
            17.1, 18.7, 20.2, 21.9, 23.6, 25.4, 27.1, 28.9, 30.4, 32.0, 33.4, 35.0, 36.4, 38.1, 39.7, 41.6,
            43.3, 45.1, 46.9, 48.7, 50.5, 52.4, 54.1, 55.7, 56.8, 57.9, 59.0, 59.9, 60.7, 61.4, 62.0, 62.5,
            62.9, 63.2, 63.4, 63.7, 64.0, 64.4, 64.9, 65.5, 66.2, 67.0, 67.8, 68.6, 69.4, 70.1, 70.9, 71.7,
            72.5, 73.2, 73.8, 74.4, 74.7, 74.7, 74.6, 74.2, 73.5, 72.6, 71.8, 71.0, 70.1, 69.4, 68.9, 68.4,
            67.9, 67.1, 65.8, 63.9, 61.4, 58.4, 55.4, 52.4, 50.0, 48.3, 47.3, 46.8, 46.9, 47.1, 47.5, 47.8,
            48.3, 48.8, 49.5, 50.2, 50.8, 51.4, 51.8, 51.9, 51.7, 51.2, 50.4, 49.2, 47.7, 46.3, 45.1, 44.2,
            43.7, 43.4, 43.1, 42.5, 41.8, 41.1, 40.3, 39.7, 39.3, 39.2, 39.3, 39.6, 40.0, 40.7, 41.4, 42.2,
            43.1, 44.1, 44.9, 45.6, 46.4, 47.0, 47.8, 48.3, 48.9, 49.4, 49.8, 49.6, 49.3, 49.0, 48.5, 48.0,
            47.5, 47.0, 46.9, 46.8, 46.8, 46.8, 46.9, 46.9, 46.9, 46.9, 46.9, 46.8, 46.6, 46.4, 46.0, 45.5,
            45.0, 44.5, 44.2, 43.9, 43.7, 43.6, 43.6, 43.5, 43.5, 43.4, 43.3, 43.1, 42.9, 42.7, 42.5, 42.4,
            42.2, 42.1, 42.0, 41.8, 41.7, 41.5, 41.3, 41.1, 40.8, 40.3, 39.6, 38.5, 37.0, 35.1, 33.0, 30.6,
            27.9, 25.1, 22.0, 18.8, 15.5, 12.3, 8.8, 6.0, 3.6, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.1, 3.0, 5.7, 8.4, 11.1, 14.0, 17.0, 20.1, 22.7, 23.6, 24.5, 24.8, 25.1,
            25.3, 25.5, 25.7, 25.8, 25.9, 26.0, 26.1, 26.3, 26.5, 26.8, 27.1, 27.5, 28.0, 28.6, 29.3, 30.4,
            31.8, 33.7, 35.8, 37.8, 39.5, 40.8, 41.8, 42.4, 43.0, 43.4, 44.0, 44.4, 45.0, 45.4, 46.0, 46.4,
            47.0, 47.4, 48.0, 48.4, 49.0, 49.4, 50.0, 50.4, 50.8, 51.1, 51.3, 51.3, 51.3, 51.3, 51.3, 51.3,
            51.3, 51.4, 51.6, 51.8, 52.1, 52.3, 52.6, 52.8, 52.9, 53.0, 53.0, 53.0, 53.1, 53.2, 53.3, 53.4,
            53.5, 53.7, 55.0, 56.8, 58.8, 60.9, 63.0, 65.0, 66.9, 68.6, 70.1, 71.5, 72.8, 73.9, 74.9, 75.7,
            76.4, 77.1, 77.6, 78.0, 78.2, 78.4, 78.5, 78.5, 78.6, 78.7, 78.9, 79.1, 79.4, 79.8, 80.1, 80.5,
            80.8, 81.0, 81.2, 81.3, 81.2, 81.0, 80.6, 80.0, 79.1, 78.0, 76.8, 75.5, 74.1, 72.9, 71.9, 71.2,
            70.9, 71.0, 71.5, 72.3, 73.2, 74.1, 74.9, 75.4, 75.5, 75.2, 74.5, 73.3, 71.7, 69.9, 67.9, 65.7,
            63.5, 61.2, 59.0, 56.8, 54.7, 52.7, 50.9, 49.4, 48.1, 47.1, 46.5, 46.3, 46.5, 47.2, 48.3, 49.7,
            51.3, 53.0, 54.9, 56.7, 58.6, 60.2, 61.6, 62.2, 62.5, 62.8, 62.9, 63.0, 63.0, 63.1, 63.2, 63.3,
            63.5, 63.7, 63.9, 64.1, 64.3, 66.1, 67.9, 69.7, 71.4, 73.1, 74.7, 76.2, 77.5, 78.6, 79.7, 80.6,
            81.5, 82.2, 83.0, 83.7, 84.4, 84.9, 85.1, 85.2, 84.9, 84.4, 83.6, 82.7, 81.5, 80.1, 78.7, 77.4,
            76.2, 75.4, 74.8, 74.3, 73.8, 73.2, 72.4, 71.6, 70.8, 69.9, 67.9, 65.7, 63.5, 61.2, 59.0, 56.8,
            54.7, 52.7, 50.9, 49.4, 48.1, 47.1, 46.5, 46.3, 45.1, 43.0, 40.6, 38.1, 35.4, 32.7, 30.0, 29.9,
            30.0, 30.2, 30.4, 30.6, 31.6, 33.0, 33.9, 34.8, 35.7, 36.6, 37.5, 38.4, 39.3, 40.2, 40.8, 41.7,
            42.4, 43.1, 43.6, 44.2, 44.8, 45.5, 46.3, 47.2, 48.1, 49.1, 50.0, 51.0, 51.9, 52.7, 53.7, 55.0,
            56.8, 58.8, 60.9, 63.0, 65.0, 66.9, 68.6, 70.1, 71.0, 71.8, 72.8, 72.9, 73.0, 72.3, 71.9, 71.3,
            70.9, 70.5, 70.0, 69.6, 69.2, 68.8, 68.4, 67.9, 67.5, 67.2, 66.8, 65.6, 63.3, 60.2, 56.2, 52.2,
            48.4, 45.0, 41.6, 38.6, 36.4, 34.8, 34.2, 34.7, 36.3, 38.5, 41.0, 43.7, 46.5, 49.1, 51.6, 53.9,
            56.0, 57.9, 59.7, 61.2, 62.5, 63.5, 64.3, 65.3, 66.3, 67.3, 68.3, 69.3, 70.3, 70.8, 70.8, 70.8,
            70.9, 70.9, 70.9, 70.9, 71.0, 71.0, 71.1, 71.2, 71.3, 71.4, 71.5, 71.7, 71.8, 71.9, 71.9, 71.9,
            71.9, 71.9, 71.9, 71.9, 72.0, 72.1, 72.4, 72.7, 73.1, 73.4, 73.8, 74.0, 74.1, 74.0, 73.0, 72.0,
            71.0, 70.0, 69.0, 68.0, 67.7, 66.7, 66.6, 66.7, 66.8, 66.9, 66.9, 66.9, 66.9, 66.9, 66.9, 66.9,
            67.0, 67.1, 67.3, 67.5, 67.8, 68.2, 68.6, 69.0, 69.3, 69.3, 69.2, 68.8, 68.2, 67.6, 67.4, 67.2,
            66.9, 66.3, 65.4, 64.0, 62.4, 60.6, 58.6, 56.7, 54.8, 53.0, 51.3, 49.6, 47.8, 45.5, 42.8, 39.8,
            36.5, 33.0, 29.5, 25.8, 22.1, 18.6, 15.3, 12.4, 9.6, 6.6, 3.8, 1.6, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 2.3, 4.6, 6.5, 8.9, 10.9, 13.5, 15.2, 17.6,
            19.3, 21.4, 23.0, 25.0, 26.5, 28.4, 29.8, 31.7, 33.7, 35.8, 38.1, 40.5, 42.2, 43.5, 44.5, 45.2,
            45.8, 46.6, 47.4, 48.5, 49.7, 51.3, 52.9, 54.3, 55.6, 56.8, 57.9, 58.9, 59.7, 60.3, 60.7, 60.9,
            61.0, 61.1, 61.4, 61.8, 62.5, 63.4, 64.5, 65.7, 66.9, 68.1, 69.1, 70.0, 70.9, 71.8, 72.6, 73.4,
            74.0, 74.7, 75.2, 75.7, 76.4, 77.2, 78.2, 78.9, 79.9, 81.1, 82.4, 83.7, 85.4, 87.0, 88.3, 89.5,
            90.5, 91.3, 92.2, 93.0, 93.8, 94.6, 95.3, 95.9, 96.6, 97.4, 98.1, 98.7, 99.5, 100.3, 101.1, 101.9,
            102.8, 103.8, 105.0, 106.1, 107.4, 108.7, 109.9, 111.2, 112.3, 113.4, 114.4, 115.3, 116.1, 116.8, 117.4, 117.7,
            118.2, 118.1, 117.7, 117.0, 116.1, 115.2, 114.4, 113.6, 113.0, 112.6, 112.2, 111.9, 111.6, 111.2, 110.7, 110.1,
            109.3, 108.4, 107.4, 106.7, 106.3, 106.2, 106.4, 107.0, 107.5, 107.9, 108.4, 108.9, 109.5, 110.2, 110.9, 111.6,
            112.2, 112.8, 113.3, 113.7, 114.1, 114.4, 114.6, 114.7, 114.7, 114.7, 114.6, 114.5, 114.5, 114.5, 114.7, 115.0,
            115.6, 116.4, 117.3, 118.2, 118.8, 119.3, 119.6, 119.7, 119.5, 119.3, 119.2, 119.0, 118.8, 118.8, 118.8, 118.8,
            118.8, 118.9, 119.0, 119.0, 119.1, 119.2, 119.4, 119.6, 119.9, 120.1, 120.3, 120.4, 120.5, 120.5, 120.5, 120.5,
            120.4, 120.3, 120.1, 119.9, 119.6, 119.5, 119.4, 119.3, 119.3, 119.4, 119.5, 119.5, 119.6, 119.6, 119.6, 119.4,
            119.3, 119.0, 118.8, 118.7, 118.8, 119.0, 119.2, 119.6, 120.0, 120.3, 120.5, 120.7, 120.9, 121.0, 121.1, 121.2,
            121.3, 121.4, 121.5, 121.5, 121.5, 121.4, 121.3, 121.1, 120.9, 120.6, 120.4, 120.2, 120.1, 119.9, 119.8, 119.8,
            119.9, 120.0, 120.2, 120.4, 120.8, 121.1, 121.6, 121.8, 122.1, 122.4, 122.7, 122.8, 123.1, 123.1, 122.8, 122.3,
            121.3, 119.9, 118.1, 115.9, 113.5, 111.1, 108.6, 106.2, 104.0, 101.1, 98.3, 95.7, 93.5, 91.5, 90.7, 90.4,
            90.2, 90.2, 90.1, 90.0, 89.8, 89.6, 89.4, 89.2, 88.9, 88.5, 88.1, 87.6, 87.1, 86.6, 86.1, 85.5,
            85.0, 84.4, 83.8, 83.2, 82.6, 81.9, 81.1, 80.0, 78.7, 76.9, 74.6, 72.0, 69.0, 65.6, 62.1, 58.5,
            54.7, 50.9, 47.3, 43.8, 40.4, 37.4, 34.3, 31.3, 28.3, 25.2, 22.0, 18.9, 16.1, 13.4, 11.1, 8.9,
            6.9, 4.9, 2.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    }

    return data
