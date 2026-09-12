import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq


## Extracting seismic features

## For each seismic component, a set of amplitude and frequency-domain features is calculated. These features provide a compact numerical representation of the waveform.

## The initial feature set includes RMS amplitude, standard deviation, peak amplitude, signal energy, dominant frequency, and spectral centroid.

def extract_features(window, fs=100.0):
    """
    Extract amplitude and frequency-domain features
    from a three-component seismic waveform.

    Parameters
    ----------
    window : np.ndarray
        Waveform with shape (3, n_samples).

    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Extracted features for the E, N and Z components.
    """
    features = {}

    component_names = ["E", "N", "Z"]

    for i, component in enumerate(component_names):

        signal_data = window[i]

        # Time-domain features
        rms_value = np.sqrt(np.mean(signal_data ** 2))
        peak_value = np.max(np.abs(signal_data))
        energy_value = np.sum(signal_data ** 2)

        # Frequency-domain representation
        spectrum = np.abs(rfft(signal_data))
        frequencies = rfftfreq(
            len(signal_data),
            d=1 / fs
        )

        # Ignore the zero-frequency component
        spectrum[0] = 0

        dominant_frequency = frequencies[
            np.argmax(spectrum)
        ]

        total_spectrum = np.sum(spectrum)

        if total_spectrum > 0:
            spectral_centroid = (
                np.sum(frequencies * spectrum)
                / total_spectrum
            )
        else:
            spectral_centroid = 0.0

        features[f"{component}_rms"] = rms_value
        features[f"{component}_peak"] = peak_value
        features[f"{component}_energy"] = energy_value
        features[f"{component}_dominant_frequency"] = dominant_frequency
        features[f"{component}_spectral_centroid"] = spectral_centroid

    return features


def build_feature_matrix(X, fs=100.0):
    """
    Extract features from every waveform in a dataset.

    Parameters
    ----------
    X : np.ndarray
        Waveform array with shape
        (n_samples, 3, n_timesteps).

    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per waveform.
    """
    feature_rows = []

    for window in X:
        feature_rows.append(
            extract_features(window, fs=fs)
        )

    return pd.DataFrame(feature_rows)