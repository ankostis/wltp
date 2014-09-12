#!/usr/bin/env python
#
# Copyright 2013-2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
'''WLTC class data for the highest-power class of vehicles.

Data below extracted from the GTR specs and prepared with the following python scripts
found inside the source-distribution:

* :file:`devtools/printwltcclass.py`
* :file:`devtools/csvcolumns8to2.py`
'''

def class_data_a():
    """
    Cycles for vehicles with :abbr:`PMR` > 34 W/kg and max-velocity < 120 km/h.
    """

    data = {
        'pmr_limits': [34, float('inf')],           ## PMR (low, high]
        'velocity_limits': [0, 120],                ## Km/h [low, high)
        'parts': [[0, 589], [590, 1022], [1023, 1477], [1478, 1800]],
        'downscale': {
            'phases': [1533, 1724, 1762],           ## Note: Start end end +1 from specs.
            'p_max_values': [1566, 111.9, 0.50],    ## t, V(Km/h), Accel(m/s2)
            'factor_coeffs': [[1.3, .65, -.65],     ## r0, a1, b1 x 2
                              [1.0, .65, -.65]],
            'v_max_split': 112,                     ## V (Km/h), >
        },
        'cycle': [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 1.7, 5.4, 9.9,
            13.1, 16.9, 21.7, 26.0, 27.5, 28.1, 28.3, 28.8, 29.1, 30.8, 31.9, 34.1, 36.6, 39.1, 41.3, 42.5,
            43.3, 43.9, 44.4, 44.5, 44.2, 42.7, 39.9, 37.0, 34.6, 32.3, 29.0, 25.1, 22.2, 20.9, 20.4, 19.5,
            18.4, 17.8, 17.8, 17.4, 15.7, 13.1, 12.1, 12.0, 12.0, 12.0, 12.3, 12.6, 14.7, 15.3, 15.9, 16.2,
            17.1, 17.8, 18.1, 18.4, 20.3, 23.2, 26.5, 29.8, 32.6, 34.4, 35.5, 36.4, 37.4, 38.5, 39.3, 39.5,
            39.0, 38.5, 37.3, 37.0, 36.7, 35.9, 35.3, 34.6, 34.2, 31.9, 27.3, 22.0, 17.0, 14.2, 12.0, 9.1,
            5.8, 3.6, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 1.9, 6.1, 11.7, 16.4, 18.9,
            19.9, 20.8, 22.8, 25.4, 27.7, 29.2, 29.8, 29.4, 27.2, 22.6, 17.3, 13.3, 12.0, 12.6, 14.1, 17.2,
            20.1, 23.4, 25.5, 27.6, 29.5, 31.1, 32.1, 33.2, 35.2, 37.2, 38.0, 37.4, 35.1, 31.0, 27.1, 25.3,
            25.1, 25.9, 27.8, 29.2, 29.6, 29.5, 29.2, 28.3, 26.1, 23.6, 21.0, 18.9, 17.1, 15.7, 14.5, 13.7,
            12.9, 12.5, 12.2, 12.0, 12.0, 12.0, 12.0, 12.5, 13.0, 14.0, 15.0, 16.5, 19.0, 21.2, 23.8, 26.9,
            29.6, 32.0, 35.2, 37.5, 39.2, 40.5, 41.6, 43.1, 45.0, 47.1, 49.0, 50.6, 51.8, 52.7, 53.1, 53.5,
            53.8, 54.2, 54.8, 55.3, 55.8, 56.2, 56.5, 56.5, 56.2, 54.9, 52.9, 51.0, 49.8, 49.2, 48.4, 46.9,
            44.3, 41.5, 39.5, 37.0, 34.6, 32.3, 29.0, 25.1, 22.2, 20.9, 20.4, 19.5, 18.4, 17.8, 17.8, 17.4,
            15.7, 14.5, 15.4, 17.9, 20.6, 23.2, 25.7, 28.7, 32.5, 36.1, 39.0, 40.8, 42.9, 44.4, 45.9, 46.0,
            45.6, 45.3, 43.7, 40.8, 38.0, 34.4, 30.9, 25.5, 21.4, 20.2, 22.9, 26.6, 30.2, 34.1, 37.4, 40.7,
            44.0, 47.3, 49.2, 49.8, 49.2, 48.1, 47.3, 46.8, 46.7, 46.8, 47.1, 47.3, 47.3, 47.1, 46.6, 45.8,
            44.8, 43.3, 41.8, 40.8, 40.3, 40.1, 39.7, 39.2, 38.5, 37.4, 36.0, 34.4, 33.0, 31.7, 30.0, 28.0,
            26.1, 25.6, 24.9, 24.9, 24.3, 23.9, 23.9, 23.6, 23.3, 20.5, 17.5, 16.9, 16.7, 15.9, 15.6, 15.0,
            14.5, 14.3, 14.5, 15.4, 17.8, 21.1, 24.1, 25.0, 25.3, 25.5, 26.4, 26.6, 27.1, 27.7, 28.1, 28.2,
            28.1, 28.0, 27.9, 27.9, 28.1, 28.2, 28.0, 26.9, 25.0, 23.2, 21.9, 21.1, 20.7, 20.7, 20.8, 21.2,
            22.1, 23.5, 24.3, 24.5, 23.8, 21.3, 17.7, 14.4, 11.9, 10.2, 8.9, 8.0, 7.2, 6.1, 4.9, 3.7,
            2.3, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 2.1, 4.8, 8.3, 12.3, 16.6, 20.9, 24.2,
            25.6, 25.6, 24.9, 23.3, 21.6, 20.2, 18.7, 17.0, 15.3, 14.2, 13.9, 14.0, 14.2, 14.5, 14.9, 15.9,
            17.4, 18.7, 19.1, 18.8, 17.6, 16.6, 16.2, 16.4, 17.2, 19.1, 22.6, 27.4, 31.6, 33.4, 33.5, 32.8,
            31.9, 31.3, 31.1, 30.6, 29.2, 26.7, 23.0, 18.2, 12.9, 7.7, 3.8, 1.3, 0.2, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.5, 2.5, 6.6, 11.8, 16.8, 20.5, 21.9, 21.9, 21.3, 20.3, 19.2, 17.8, 15.5, 11.9, 7.6, 4.0,
            2.0, 1.0, 0.0, 0.0, 0.0, 0.2, 1.2, 3.2, 5.2, 8.2, 13.0, 18.8, 23.1, 24.5, 24.5, 24.3,
            23.6, 22.3, 20.1, 18.5, 17.2, 16.3, 15.4, 14.7, 14.3, 13.7, 13.3, 13.1, 13.1, 13.3, 13.8, 14.5,
            16.5, 17.0, 17.0, 17.0, 15.4, 10.1, 4.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.1, 5.2, 9.2, 13.5, 18.1, 22.3,
            26.0, 29.3, 32.8, 36.0, 39.2, 42.5, 45.7, 48.2, 48.4, 48.2, 47.8, 47.0, 45.9, 44.9, 44.4, 44.3,
            44.5, 45.1, 45.7, 46.0, 46.0, 46.0, 46.1, 46.7, 47.7, 48.9, 50.3, 51.6, 52.6, 53.0, 53.0, 52.9,
            52.7, 52.6, 53.1, 54.3, 55.2, 55.5, 55.9, 56.3, 56.7, 56.9, 56.8, 56.0, 54.2, 52.1, 50.1, 47.2,
            43.2, 39.2, 36.5, 34.3, 31.0, 26.0, 20.7, 15.4, 13.1, 12.0, 12.5, 14.0, 19.0, 23.2, 28.0, 32.0,
            34.0, 36.0, 38.0, 40.0, 40.3, 40.5, 39.0, 35.7, 31.8, 27.1, 22.8, 21.1, 18.9, 18.9, 21.3, 23.9,
            25.9, 28.4, 30.3, 30.9, 31.1, 31.8, 32.7, 33.2, 32.4, 28.3, 25.8, 23.1, 21.8, 21.2, 21.0, 21.0,
            20.9, 19.9, 17.9, 15.1, 12.8, 12.0, 13.2, 17.1, 21.1, 21.8, 21.2, 18.5, 13.9, 12.0, 12.0, 13.0,
            16.3, 20.5, 23.9, 26.0, 28.0, 31.5, 33.4, 36.0, 37.8, 40.2, 41.6, 41.9, 42.0, 42.2, 42.4, 42.7,
            43.1, 43.7, 44.0, 44.1, 45.3, 46.4, 47.2, 47.3, 47.4, 47.4, 47.5, 47.9, 48.6, 49.4, 49.8, 49.8,
            49.7, 49.3, 48.5, 47.6, 46.3, 43.7, 39.3, 34.1, 29.0, 23.7, 18.4, 14.3, 12.0, 12.8, 16.0, 20.4,
            24.0, 29.0, 32.2, 36.8, 39.4, 43.2, 45.8, 49.2, 51.4, 54.2, 56.0, 58.3, 59.8, 61.7, 62.7, 63.3,
            63.6, 64.0, 64.7, 65.2, 65.3, 65.3, 65.4, 65.7, 66.0, 65.6, 63.5, 59.7, 54.6, 49.3, 44.9, 42.3,
            41.4, 41.3, 43.0, 45.0, 46.5, 48.3, 49.5, 51.2, 52.2, 51.6, 49.7, 47.4, 43.7, 39.7, 35.5, 31.1,
            26.3, 21.9, 18.0, 17.0, 18.0, 21.4, 24.8, 27.9, 30.8, 33.0, 35.1, 37.1, 38.9, 41.4, 44.0, 46.3,
            47.7, 48.2, 48.7, 49.3, 49.8, 50.2, 50.9, 51.8, 52.5, 53.3, 54.5, 55.7, 56.5, 56.8, 57.0, 57.2,
            57.7, 58.7, 60.1, 61.1, 61.7, 62.3, 62.9, 63.3, 63.4, 63.5, 63.9, 64.4, 65.0, 65.6, 66.6, 67.4,
            68.2, 69.1, 70.0, 70.8, 71.5, 72.4, 73.0, 73.7, 74.4, 74.9, 75.3, 75.6, 75.8, 76.6, 76.5, 76.2,
            75.8, 75.4, 74.8, 73.9, 72.7, 71.3, 70.4, 70.0, 70.0, 69.0, 68.0, 67.3, 66.2, 64.8, 63.6, 62.6,
            62.1, 61.9, 61.9, 61.8, 61.5, 60.9, 59.7, 54.6, 49.3, 44.9, 42.3, 41.4, 41.3, 42.1, 44.7, 46.0,
            48.8, 50.1, 51.3, 54.1, 55.2, 56.2, 56.1, 56.1, 56.5, 57.5, 59.2, 60.7, 61.8, 62.3, 62.7, 62.0,
            61.3, 60.9, 60.5, 60.2, 59.8, 59.4, 58.6, 57.5, 56.6, 56.0, 55.5, 55.0, 54.4, 54.1, 54.0, 53.9,
            53.9, 54.0, 54.2, 55.0, 55.8, 56.2, 56.1, 55.1, 52.7, 48.4, 43.1, 37.8, 32.5, 27.2, 25.1, 27.0,
            29.8, 33.8, 37.0, 40.7, 43.0, 45.6, 46.9, 47.0, 46.9, 46.5, 45.8, 44.3, 41.3, 36.5, 31.7, 27.0,
            24.7, 19.3, 16.0, 13.2, 10.7, 8.8, 7.2, 5.5, 3.2, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.8, 3.6, 8.6, 14.6, 20.0, 24.4, 28.2, 31.7, 35.0, 37.6, 39.7, 41.5, 43.6,
            46.0, 48.4, 50.5, 51.9, 52.6, 52.8, 52.9, 53.1, 53.3, 53.1, 52.3, 50.7, 48.8, 46.5, 43.8, 40.3,
            36.0, 30.7, 25.4, 21.0, 16.7, 13.4, 12.0, 12.1, 12.8, 15.6, 19.9, 23.4, 24.6, 27.0, 29.0, 32.0,
            34.8, 37.7, 40.8, 43.2, 46.0, 48.0, 50.7, 52.0, 54.5, 55.9, 57.4, 58.1, 58.4, 58.8, 58.8, 58.6,
            58.7, 58.8, 58.8, 58.8, 59.1, 60.1, 61.7, 63.0, 63.7, 63.9, 63.5, 62.3, 60.3, 58.9, 58.4, 58.8,
            60.2, 62.3, 63.9, 64.5, 64.4, 63.5, 62.0, 61.2, 61.3, 61.7, 62.0, 64.6, 66.0, 66.2, 65.8, 64.7,
            63.6, 62.9, 62.4, 61.7, 60.1, 57.3, 55.8, 50.5, 45.2, 40.1, 36.2, 32.9, 29.8, 26.6, 23.0, 19.4,
            16.3, 14.6, 14.2, 14.3, 14.6, 15.1, 16.4, 19.1, 22.5, 24.4, 24.8, 22.7, 17.4, 13.8, 12.0, 12.0,
            12.0, 13.9, 17.7, 22.8, 27.3, 31.2, 35.2, 39.4, 42.5, 45.4, 48.2, 50.3, 52.6, 54.5, 56.6, 58.3,
            60.0, 61.5, 63.1, 64.3, 65.7, 67.1, 68.3, 69.7, 70.6, 71.6, 72.6, 73.5, 74.2, 74.9, 75.6, 76.3,
            77.1, 77.9, 78.5, 79.0, 79.7, 80.3, 81.0, 81.6, 82.4, 82.9, 83.4, 83.8, 84.2, 84.7, 85.2, 85.6,
            86.3, 86.8, 87.4, 88.0, 88.3, 88.7, 89.0, 89.3, 89.8, 90.2, 90.6, 91.0, 91.3, 91.6, 91.9, 92.2,
            92.8, 93.1, 93.3, 93.5, 93.7, 93.9, 94.0, 94.1, 94.3, 94.4, 94.6, 94.7, 94.8, 95.0, 95.1, 95.3,
            95.4, 95.6, 95.7, 95.8, 96.0, 96.1, 96.3, 96.4, 96.6, 96.8, 97.0, 97.2, 97.3, 97.4, 97.4, 97.4,
            97.4, 97.3, 97.3, 97.3, 97.3, 97.2, 97.1, 97.0, 96.9, 96.7, 96.4, 96.1, 95.7, 95.5, 95.3, 95.2,
            95.0, 94.9, 94.7, 94.5, 94.4, 94.4, 94.3, 94.3, 94.1, 93.9, 93.4, 92.8, 92.0, 91.3, 90.6, 90.0,
            89.3, 88.7, 88.1, 87.4, 86.7, 86.0, 85.3, 84.7, 84.1, 83.5, 82.9, 82.3, 81.7, 81.1, 80.5, 79.9,
            79.4, 79.1, 78.8, 78.5, 78.2, 77.9, 77.6, 77.3, 77.0, 76.7, 76.0, 76.0, 76.0, 75.9, 76.0, 76.0,
            76.1, 76.3, 76.5, 76.6, 76.8, 77.1, 77.1, 77.2, 77.2, 77.6, 78.0, 78.4, 78.8, 79.2, 80.3, 80.8,
            81.0, 81.0, 81.0, 81.0, 81.0, 80.9, 80.6, 80.3, 80.0, 79.9, 79.8, 79.8, 79.8, 79.9, 80.0, 80.4,
            80.8, 81.2, 81.5, 81.6, 81.6, 81.4, 80.7, 79.6, 78.2, 76.8, 75.3, 73.8, 72.1, 70.2, 68.2, 66.1,
            63.8, 61.6, 60.2, 59.8, 60.4, 61.8, 62.6, 62.7, 61.9, 60.0, 58.4, 57.8, 57.8, 57.8, 57.3, 56.2,
            54.3, 50.8, 45.5, 40.2, 34.9, 29.6, 28.7, 29.3, 30.5, 31.7, 32.9, 35.0, 38.0, 40.5, 42.7, 45.8,
            47.5, 48.9, 49.4, 49.4, 49.2, 48.7, 47.9, 46.9, 45.6, 44.2, 42.7, 40.7, 37.1, 33.9, 30.6, 28.6,
            27.3, 27.2, 27.5, 27.4, 27.1, 26.7, 26.8, 28.2, 31.1, 34.8, 38.4, 40.9, 41.7, 40.9, 38.3, 35.3,
            34.3, 34.6, 36.3, 39.5, 41.8, 42.5, 41.9, 40.1, 36.6, 31.3, 26.0, 20.6, 19.1, 19.7, 21.1, 22.0,
            22.1, 21.4, 19.6, 18.3, 18.0, 18.3, 18.5, 17.9, 15.0, 9.9, 4.6, 1.2, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2, 4.4, 6.3, 7.9, 9.2, 10.4, 11.5, 12.9, 14.7,
            17.0, 19.8, 23.1, 26.7, 30.5, 34.1, 37.5, 40.6, 43.3, 45.7, 47.7, 49.3, 50.5, 51.3, 52.1, 52.7,
            53.4, 54.0, 54.5, 55.0, 55.6, 56.3, 57.2, 58.5, 60.2, 62.3, 64.7, 67.1, 69.2, 70.7, 71.9, 72.7,
            73.4, 73.8, 74.1, 74.0, 73.6, 72.5, 70.8, 68.6, 66.2, 64.0, 62.2, 60.9, 60.2, 60.0, 60.4, 61.4,
            63.2, 65.6, 68.4, 71.6, 74.9, 78.4, 81.8, 84.9, 87.4, 89.0, 90.0, 90.6, 91.0, 91.5, 92.0, 92.7,
            93.4, 94.2, 94.9, 95.7, 96.6, 97.7, 98.9, 100.4, 102.0, 103.6, 105.2, 106.8, 108.5, 110.2, 111.9, 113.7,
            115.3, 116.8, 118.2, 119.5, 120.7, 121.8, 122.6, 123.2, 123.6, 123.7, 123.6, 123.3, 123.0, 122.5, 122.1, 121.5,
            120.8, 120.0, 119.1, 118.1, 117.1, 116.2, 115.5, 114.9, 114.5, 114.1, 113.9, 113.7, 113.3, 112.9, 112.2, 111.4,
            110.5, 109.5, 108.5, 107.7, 107.1, 106.6, 106.4, 106.2, 106.2, 106.2, 106.4, 106.5, 106.8, 107.2, 107.8, 108.5,
            109.4, 110.5, 111.7, 113.0, 114.1, 115.1, 115.9, 116.5, 116.7, 116.6, 116.2, 115.2, 113.8, 112.0, 110.1, 108.3,
            107.0, 106.1, 105.8, 105.7, 105.7, 105.6, 105.3, 104.9, 104.4, 104.0, 103.8, 103.9, 104.4, 105.1, 106.1, 107.2,
            108.5, 109.9, 111.3, 112.7, 113.9, 115.0, 116.0, 116.8, 117.6, 118.4, 119.2, 120.0, 120.8, 121.6, 122.3, 123.1,
            123.8, 124.4, 125.0, 125.4, 125.8, 126.1, 126.4, 126.6, 126.7, 126.8, 126.9, 126.9, 126.9, 126.8, 126.6, 126.3,
            126.0, 125.7, 125.6, 125.6, 125.8, 126.2, 126.6, 127.0, 127.4, 127.6, 127.8, 127.9, 128.0, 128.1, 128.2, 128.3,
            128.4, 128.5, 128.6, 128.6, 128.5, 128.3, 128.1, 127.9, 127.6, 127.4, 127.2, 127.0, 126.9, 126.8, 126.7, 126.8,
            126.9, 127.1, 127.4, 127.7, 128.1, 128.5, 129.0, 129.5, 130.1, 130.6, 131.0, 131.2, 131.3, 131.2, 130.7, 129.8,
            128.4, 126.5, 124.1, 121.6, 119.0, 116.5, 114.1, 111.8, 109.5, 107.1, 104.8, 102.5, 100.4, 98.6, 97.2, 95.9,
            94.8, 93.8, 92.8, 91.8, 91.0, 90.2, 89.6, 89.1, 88.6, 88.1, 87.6, 87.1, 86.6, 86.1, 85.5, 85.0,
            84.4, 83.8, 83.2, 82.6, 82.0, 81.3, 80.4, 79.1, 77.4, 75.1, 72.3, 69.1, 65.9, 62.7, 59.7, 57.0,
            54.6, 52.2, 49.7, 46.8, 43.5, 39.9, 36.4, 33.2, 30.5, 28.3, 26.3, 24.4, 22.5, 20.5, 18.2, 15.5,
            12.3, 8.7, 5.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    }

    return data


def class_data_b():
    """
    Cycles for vehicles with :abbr:`PMR` > 34 W/kg and max-velocity >= 120 km/h.
    """

    cycle = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 1.7, 5.4, 9.9,
        13.1, 16.9, 21.7, 26.0, 27.5, 28.1, 28.3, 28.8, 29.1, 30.8, 31.9, 34.1, 36.6, 39.1, 41.3, 42.5,
        43.3, 43.9, 44.4, 44.5, 44.2, 42.7, 39.9, 37.0, 34.6, 32.3, 29.0, 25.1, 22.2, 20.9, 20.4, 19.5,
        18.4, 17.8, 17.8, 17.4, 15.7, 13.1, 12.1, 12.0, 12.0, 12.0, 12.3, 12.6, 14.7, 15.3, 15.9, 16.2,
        17.1, 17.8, 18.1, 18.4, 20.3, 23.2, 26.5, 29.8, 32.6, 34.4, 35.5, 36.4, 37.4, 38.5, 39.3, 39.5,
        39.0, 38.5, 37.3, 37.0, 36.7, 35.9, 35.3, 34.6, 34.2, 31.9, 27.3, 22.0, 17.0, 14.2, 12.0, 9.1,
        5.8, 3.6, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 1.9, 6.1, 11.7, 16.4, 18.9,
        19.9, 20.8, 22.8, 25.4, 27.7, 29.2, 29.8, 29.4, 27.2, 22.6, 17.3, 13.3, 12.0, 12.6, 14.1, 17.2,
        20.1, 23.4, 25.5, 27.6, 29.5, 31.1, 32.1, 33.2, 35.2, 37.2, 38.0, 37.4, 35.1, 31.0, 27.1, 25.3,
        25.1, 25.9, 27.8, 29.2, 29.6, 29.5, 29.2, 28.3, 26.1, 23.6, 21.0, 18.9, 17.1, 15.7, 14.5, 13.7,
        12.9, 12.5, 12.2, 12.0, 12.0, 12.0, 12.0, 12.5, 13.0, 14.0, 15.0, 16.5, 19.0, 21.2, 23.8, 26.9,
        29.6, 32.0, 35.2, 37.5, 39.2, 40.5, 41.6, 43.1, 45.0, 47.1, 49.0, 50.6, 51.8, 52.7, 53.1, 53.5,
        53.8, 54.2, 54.8, 55.3, 55.8, 56.2, 56.5, 56.5, 56.2, 54.9, 52.9, 51.0, 49.8, 49.2, 48.4, 46.9,
        44.3, 41.5, 39.5, 37.0, 34.6, 32.3, 29.0, 25.1, 22.2, 20.9, 20.4, 19.5, 18.4, 17.8, 17.8, 17.4,
        15.7, 14.5, 15.4, 17.9, 20.6, 23.2, 25.7, 28.7, 32.5, 36.1, 39.0, 40.8, 42.9, 44.4, 45.9, 46.0,
        45.6, 45.3, 43.7, 40.8, 38.0, 34.4, 30.9, 25.5, 21.4, 20.2, 22.9, 26.6, 30.2, 34.1, 37.4, 40.7,
        44.0, 47.3, 49.2, 49.8, 49.2, 48.1, 47.3, 46.8, 46.7, 46.8, 47.1, 47.3, 47.3, 47.1, 46.6, 45.8,
        44.8, 43.3, 41.8, 40.8, 40.3, 40.1, 39.7, 39.2, 38.5, 37.4, 36.0, 34.4, 33.0, 31.7, 30.0, 28.0,
        26.1, 25.6, 24.9, 24.9, 24.3, 23.9, 23.9, 23.6, 23.3, 20.5, 17.5, 16.9, 16.7, 15.9, 15.6, 15.0,
        14.5, 14.3, 14.5, 15.4, 17.8, 21.1, 24.1, 25.0, 25.3, 25.5, 26.4, 26.6, 27.1, 27.7, 28.1, 28.2,
        28.1, 28.0, 27.9, 27.9, 28.1, 28.2, 28.0, 26.9, 25.0, 23.2, 21.9, 21.1, 20.7, 20.7, 20.8, 21.2,
        22.1, 23.5, 24.3, 24.5, 23.8, 21.3, 17.7, 14.4, 11.9, 10.2, 8.9, 8.0, 7.2, 6.1, 4.9, 3.7,
        2.3, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 2.1, 4.8, 8.3, 12.3, 16.6, 20.9, 24.2,
        25.6, 25.6, 24.9, 23.3, 21.6, 20.2, 18.7, 17.0, 15.3, 14.2, 13.9, 14.0, 14.2, 14.5, 14.9, 15.9,
        17.4, 18.7, 19.1, 18.8, 17.6, 16.6, 16.2, 16.4, 17.2, 19.1, 22.6, 27.4, 31.6, 33.4, 33.5, 32.8,
        31.9, 31.3, 31.1, 30.6, 29.2, 26.7, 23.0, 18.2, 12.9, 7.7, 3.8, 1.3, 0.2, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.5, 2.5, 6.6, 11.8, 16.8, 20.5, 21.9, 21.9, 21.3, 20.3, 19.2, 17.8, 15.5, 11.9, 7.6, 4.0,
        2.0, 1.0, 0.0, 0.0, 0.0, 0.2, 1.2, 3.2, 5.2, 8.2, 13.0, 18.8, 23.1, 24.5, 24.5, 24.3,
        23.6, 22.3, 20.1, 18.5, 17.2, 16.3, 15.4, 14.7, 14.3, 13.7, 13.3, 13.1, 13.1, 13.3, 13.8, 14.5,
        16.5, 17.0, 17.0, 17.0, 15.4, 10.1, 4.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.1, 4.8, 9.1, 14.2, 19.8, 25.5,
        30.5, 34.8, 38.8, 42.9, 46.4, 48.3, 48.7, 48.5, 48.4, 48.2, 47.8, 47.0, 45.9, 44.9, 44.4, 44.3,
        44.5, 45.1, 45.7, 46.0, 46.0, 46.0, 46.1, 46.7, 47.7, 48.9, 50.3, 51.6, 52.6, 53.0, 53.0, 52.9,
        52.7, 52.6, 53.1, 54.3, 55.2, 55.5, 55.9, 56.3, 56.7, 56.9, 56.8, 56.0, 54.2, 52.1, 50.1, 47.2,
        43.2, 39.2, 36.5, 34.3, 31.0, 26.0, 20.7, 15.4, 13.1, 12.0, 12.5, 14.0, 19.0, 23.2, 28.0, 32.0,
        34.0, 36.0, 38.0, 40.0, 40.3, 40.5, 39.0, 35.7, 31.8, 27.1, 22.8, 21.1, 18.9, 18.9, 21.3, 23.9,
        25.9, 28.4, 30.3, 30.9, 31.1, 31.8, 32.7, 33.2, 32.4, 28.3, 25.8, 23.1, 21.8, 21.2, 21.0, 21.0,
        20.9, 19.9, 17.9, 15.1, 12.8, 12.0, 13.2, 17.1, 21.1, 21.8, 21.2, 18.5, 13.9, 12.0, 12.0, 13.0,
        16.0, 18.5, 20.6, 22.5, 24.0, 26.6, 29.9, 34.8, 37.8, 40.2, 41.6, 41.9, 42.0, 42.2, 42.4, 42.7,
        43.1, 43.7, 44.0, 44.1, 45.3, 46.4, 47.2, 47.3, 47.4, 47.4, 47.5, 47.9, 48.6, 49.4, 49.8, 49.8,
        49.7, 49.3, 48.5, 47.6, 46.3, 43.7, 39.3, 34.1, 29.0, 23.7, 18.4, 14.3, 12.0, 12.8, 16.0, 19.1,
        22.4, 25.6, 30.1, 35.3, 39.9, 44.5, 47.5, 50.9, 54.1, 56.3, 58.1, 59.8, 61.1, 62.1, 62.8, 63.3,
        63.6, 64.0, 64.7, 65.2, 65.3, 65.3, 65.4, 65.7, 66.0, 65.6, 63.5, 59.7, 54.6, 49.3, 44.9, 42.3,
        41.4, 41.3, 42.1, 44.7, 48.4, 51.4, 52.7, 53.0, 52.5, 51.3, 49.7, 47.4, 43.7, 39.7, 35.5, 31.1,
        26.3, 21.9, 18.0, 17.0, 18.0, 21.4, 24.8, 27.9, 30.8, 33.0, 35.1, 37.1, 38.9, 41.4, 44.0, 46.3,
        47.7, 48.2, 48.7, 49.3, 49.8, 50.2, 50.9, 51.8, 52.5, 53.3, 54.5, 55.7, 56.5, 56.8, 57.0, 57.2,
        57.7, 58.7, 60.1, 61.1, 61.7, 62.3, 62.9, 63.3, 63.4, 63.5, 64.5, 65.8, 66.8, 67.4, 68.8, 71.1,
        72.3, 72.8, 73.4, 74.6, 76.0, 76.6, 76.5, 76.2, 75.8, 75.4, 74.8, 73.9, 72.7, 71.3, 70.4, 70.0,
        70.0, 69.0, 68.0, 68.0, 68.0, 68.1, 68.4, 68.6, 68.7, 68.5, 68.1, 67.3, 66.2, 64.8, 63.6, 62.6,
        62.1, 61.9, 61.9, 61.8, 61.5, 60.9, 59.7, 54.6, 49.3, 44.9, 42.3, 41.4, 41.3, 42.1, 44.7, 48.4,
        51.4, 52.7, 54.0, 57.0, 58.1, 59.2, 59.0, 59.1, 59.5, 60.5, 62.3, 63.9, 65.1, 64.1, 62.7, 62.0,
        61.3, 60.9, 60.5, 60.2, 59.8, 59.4, 58.6, 57.5, 56.6, 56.0, 55.5, 55.0, 54.4, 54.1, 54.0, 53.9,
        53.9, 54.0, 54.2, 55.0, 55.8, 56.2, 56.1, 55.1, 52.7, 48.4, 43.1, 37.8, 32.5, 27.2, 25.1, 26.0,
        29.3, 34.6, 40.4, 45.3, 49.0, 51.1, 52.1, 52.2, 52.1, 51.7, 50.9, 49.2, 45.9, 40.6, 35.3, 30.0,
        24.7, 19.3, 16.0, 13.2, 10.7, 8.8, 7.2, 5.5, 3.2, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.8, 3.6, 8.6, 14.6, 20.0, 24.4, 28.2, 31.7, 35.0, 37.6, 39.7, 41.5, 43.6,
        46.0, 48.4, 50.5, 51.9, 52.6, 52.8, 52.9, 53.1, 53.3, 53.1, 52.3, 50.7, 48.8, 46.5, 43.8, 40.3,
        36.0, 30.7, 25.4, 21.0, 16.7, 13.4, 12.0, 12.1, 12.8, 15.6, 19.9, 23.4, 24.6, 25.2, 26.4, 28.8,
        31.8, 35.3, 39.5, 44.5, 49.3, 53.3, 56.4, 58.9, 61.2, 62.6, 63.0, 62.5, 60.9, 59.3, 58.6, 58.6,
        58.7, 58.8, 58.8, 58.8, 59.1, 60.1, 61.7, 63.0, 63.7, 63.9, 63.5, 62.3, 60.3, 58.9, 58.4, 58.8,
        60.2, 62.3, 63.9, 64.5, 64.4, 63.5, 62.0, 61.2, 61.3, 62.6, 65.3, 68.0, 69.4, 69.7, 69.3, 68.1,
        66.9, 66.2, 65.7, 64.9, 63.2, 60.3, 55.8, 50.5, 45.2, 40.1, 36.2, 32.9, 29.8, 26.6, 23.0, 19.4,
        16.3, 14.6, 14.2, 14.3, 14.6, 15.1, 16.4, 19.1, 22.5, 24.4, 24.8, 22.7, 17.4, 13.8, 12.0, 12.0,
        12.0, 13.9, 17.7, 22.8, 27.3, 31.2, 35.2, 39.4, 42.5, 45.4, 48.2, 50.3, 52.6, 54.5, 56.6, 58.3,
        60.0, 61.5, 63.1, 64.3, 65.7, 67.1, 68.3, 69.7, 70.6, 71.6, 72.6, 73.5, 74.2, 74.9, 75.6, 76.3,
        77.1, 77.9, 78.5, 79.0, 79.7, 80.3, 81.0, 81.6, 82.4, 82.9, 83.4, 83.8, 84.2, 84.7, 85.2, 85.6,
        86.3, 86.8, 87.4, 88.0, 88.3, 88.7, 89.0, 89.3, 89.8, 90.2, 90.6, 91.0, 91.3, 91.6, 91.9, 92.2,
        92.8, 93.1, 93.3, 93.5, 93.7, 93.9, 94.0, 94.1, 94.3, 94.4, 94.6, 94.7, 94.8, 95.0, 95.1, 95.3,
        95.4, 95.6, 95.7, 95.8, 96.0, 96.1, 96.3, 96.4, 96.6, 96.8, 97.0, 97.2, 97.3, 97.4, 97.4, 97.4,
        97.4, 97.3, 97.3, 97.3, 97.3, 97.2, 97.1, 97.0, 96.9, 96.7, 96.4, 96.1, 95.7, 95.5, 95.3, 95.2,
        95.0, 94.9, 94.7, 94.5, 94.4, 94.4, 94.3, 94.3, 94.1, 93.9, 93.4, 92.8, 92.0, 91.3, 90.6, 90.0,
        89.3, 88.7, 88.1, 87.4, 86.7, 86.0, 85.3, 84.7, 84.1, 83.5, 82.9, 82.3, 81.7, 81.1, 80.5, 79.9,
        79.4, 79.1, 78.8, 78.5, 78.2, 77.9, 77.6, 77.3, 77.0, 76.7, 76.0, 76.0, 76.0, 75.9, 75.9, 75.8,
        75.7, 75.5, 75.2, 75.0, 74.7, 74.1, 73.7, 73.3, 73.5, 74.0, 74.9, 76.1, 77.7, 79.2, 80.3, 80.8,
        81.0, 81.0, 81.0, 81.0, 81.0, 80.9, 80.6, 80.3, 80.0, 79.9, 79.8, 79.8, 79.8, 79.9, 80.0, 80.4,
        80.8, 81.2, 81.5, 81.6, 81.6, 81.4, 80.7, 79.6, 78.2, 76.8, 75.3, 73.8, 72.1, 70.2, 68.2, 66.1,
        63.8, 61.6, 60.2, 59.8, 60.4, 61.8, 62.6, 62.7, 61.9, 60.0, 58.4, 57.8, 57.8, 57.8, 57.3, 56.2,
        54.3, 50.8, 45.5, 40.2, 34.9, 29.6, 27.3, 29.3, 32.9, 35.6, 36.7, 37.6, 39.4, 42.5, 46.5, 50.2,
        52.8, 54.3, 54.9, 54.9, 54.7, 54.1, 53.2, 52.1, 50.7, 49.1, 47.4, 45.2, 41.8, 36.5, 31.2, 27.6,
        26.9, 27.3, 27.5, 27.4, 27.1, 26.7, 26.8, 28.2, 31.1, 34.8, 38.4, 40.9, 41.7, 40.9, 38.3, 35.3,
        34.3, 34.6, 36.3, 39.5, 41.8, 42.5, 41.9, 40.1, 36.6, 31.3, 26.0, 20.6, 19.1, 19.7, 21.1, 22.0,
        22.1, 21.4, 19.6, 18.3, 18.0, 18.3, 18.5, 17.9, 15.0, 9.9, 4.6, 1.2, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2, 4.4, 6.3, 7.9, 9.2, 10.4, 11.5, 12.9, 14.7,
        17.0, 19.8, 23.1, 26.7, 30.5, 34.1, 37.5, 40.6, 43.3, 45.7, 47.7, 49.3, 50.5, 51.3, 52.1, 52.7,
        53.4, 54.0, 54.5, 55.0, 55.6, 56.3, 57.2, 58.5, 60.2, 62.3, 64.7, 67.1, 69.2, 70.7, 71.9, 72.7,
        73.4, 73.8, 74.1, 74.0, 73.6, 72.5, 70.8, 68.6, 66.2, 64.0, 62.2, 60.9, 60.2, 60.0, 60.4, 61.4,
        63.2, 65.6, 68.4, 71.6, 74.9, 78.4, 81.8, 84.9, 87.4, 89.0, 90.0, 90.6, 91.0, 91.5, 92.0, 92.7,
        93.4, 94.2, 94.9, 95.7, 96.6, 97.7, 98.9, 100.4, 102.0, 103.6, 105.2, 106.8, 108.5, 110.2, 111.9, 113.7,
        115.3, 116.8, 118.2, 119.5, 120.7, 121.8, 122.6, 123.2, 123.6, 123.7, 123.6, 123.3, 123.0, 122.5, 122.1, 121.5,
        120.8, 120.0, 119.1, 118.1, 117.1, 116.2, 115.5, 114.9, 114.5, 114.1, 113.9, 113.7, 113.3, 112.9, 112.2, 111.4,
        110.5, 109.5, 108.5, 107.7, 107.1, 106.6, 106.4, 106.2, 106.2, 106.2, 106.4, 106.5, 106.8, 107.2, 107.8, 108.5,
        109.4, 110.5, 111.7, 113.0, 114.1, 115.1, 115.9, 116.5, 116.7, 116.6, 116.2, 115.2, 113.8, 112.0, 110.1, 108.3,
        107.0, 106.1, 105.8, 105.7, 105.7, 105.6, 105.3, 104.9, 104.4, 104.0, 103.8, 103.9, 104.4, 105.1, 106.1, 107.2,
        108.5, 109.9, 111.3, 112.7, 113.9, 115.0, 116.0, 116.8, 117.6, 118.4, 119.2, 120.0, 120.8, 121.6, 122.3, 123.1,
        123.8, 124.4, 125.0, 125.4, 125.8, 126.1, 126.4, 126.6, 126.7, 126.8, 126.9, 126.9, 126.9, 126.8, 126.6, 126.3,
        126.0, 125.7, 125.6, 125.6, 125.8, 126.2, 126.6, 127.0, 127.4, 127.6, 127.8, 127.9, 128.0, 128.1, 128.2, 128.3,
        128.4, 128.5, 128.6, 128.6, 128.5, 128.3, 128.1, 127.9, 127.6, 127.4, 127.2, 127.0, 126.9, 126.8, 126.7, 126.8,
        126.9, 127.1, 127.4, 127.7, 128.1, 128.5, 129.0, 129.5, 130.1, 130.6, 131.0, 131.2, 131.3, 131.2, 130.7, 129.8,
        128.4, 126.5, 124.1, 121.6, 119.0, 116.5, 114.1, 111.8, 109.5, 107.1, 104.8, 102.5, 100.4, 98.6, 97.2, 95.9,
        94.8, 93.8, 92.8, 91.8, 91.0, 90.2, 89.6, 89.1, 88.6, 88.1, 87.6, 87.1, 86.6, 86.1, 85.5, 85.0,
        84.4, 83.8, 83.2, 82.6, 82.0, 81.3, 80.4, 79.1, 77.4, 75.1, 72.3, 69.1, 65.9, 62.7, 59.7, 57.0,
        54.6, 52.2, 49.7, 46.8, 43.5, 39.9, 36.4, 33.2, 30.5, 28.3, 26.3, 24.4, 22.5, 20.5, 18.2, 15.5,
        12.3, 8.7, 5.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]


    data = class_data_a()
    data['cycle'] = cycle
    data['velocity_limits'] = [120, float('inf')]   ## Km/h [low, high)

    return data

