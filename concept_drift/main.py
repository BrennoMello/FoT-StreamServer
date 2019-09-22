import os

import pandas as pd
from matplotlib import pyplot as plt
from os.path import dirname

# Baseados em estatística - média
from algorithms.cusum import CUSUM
from algorithms.page_hinkley import PH
from algorithms.ewma import EWMA

# Baseados em janela
from algorithms.adwin import ADWINChangeDetector
from algorithms.seq_drift2 import SeqDrift2ChangeDetector
from algorithms.hddm_a import HDDM_A_test


DETECTORS = {
    "CUSUM": {"class": CUSUM, "args": {"min_instance": 50, "delta": 0.001, "lambda_": 750}},
    "PH": {"class": PH, "args": {"min_instance": 50, "delta": 0.001, "lambda_": 750}},
    "EWMA": {"class": EWMA, "args": {"min_instance": 50, "lambda_": 1.5}},
    "ADWINChangeDetector": {"class": ADWINChangeDetector, "args": {"delta": 0.005}},
    "SeqDrift2ChangeDetector": {"class": SeqDrift2ChangeDetector, "args": {"block_size": 50, "delta": 0.00001}},
    "HDDM_A_test": {"class": HDDM_A_test, "args": {"drift_confidence": 0.00001}},
}

DATASET_PATH = os.path.join(dirname(dirname(__file__)), "intel-lab-dataset/dataSet_temp.txt")

raw_dataset = pd.read_csv(DATASET_PATH, delim_whitespace=True, header=0)
temperatures = raw_dataset.iloc[:, -1].dropna().head(5000).values

# No Wavelet
for detector_name in DETECTORS:
    detector_setup = DETECTORS[detector_name]
    detector_instance = detector_setup["class"](**detector_setup["args"])

    drift_indexes = []
    for index, value in enumerate(temperatures):
        _, is_drift = detector_instance.run(value)

        if is_drift:
            drift_indexes.append(index)
            detector_instance.reset()


    plt.ylim((0, 50))
    plt.suptitle('Without wavelet')
    plt.title(str(detector_name) + str(detector_setup["args"]))
    plt.plot(temperatures)

    for drift_index in drift_indexes:
        plt.axvline(drift_index, linestyle='--', linewidth=1, color='r')

    plt.show()

# With wavelets
import pywt
temperatures, _ = pywt.dwt(temperatures, 'db1')

for detector in DETECTORS:
    detector_instance = detector()

    drift_indexes = []
    for index, value in enumerate(temperatures):
        _, is_drift = detector_instance.run(value)

        if is_drift:
            drift_indexes.append(index)
            detector_instance.reset()


    plt.ylim((0, 50))
    plt.suptitle('With wavelet')
    plt.title(str(detector))
    plt.plot(temperatures)

    for drift_index in drift_indexes:
        plt.axvline(drift_index, linestyle='--', linewidth=1, color='r')

    plt.show()