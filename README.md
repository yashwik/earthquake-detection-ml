# Earthquake Detection and P-Wave Picking from Seismic Waveforms Using STA/LTA and Machine Learning

## Overview

This project investigates earthquake detection and P-wave picking from seismic waveforms using a combination of traditional seismological signal-processing methods and machine learning.

The project is designed as a computational geophysics workflow, combining seismic waveform analysis with feature-based machine learning and deep learning.

The system is divided into two stages:

1. **Stage 1:** Earthquake vs. noise classification
2. **Stage 2:** P-wave detection and arrival-time picking

The INSTANCE seismic dataset is used throughout the project.

---

## Objectives

- Analyze three-component seismic waveforms from the E, N, and Z components.
- Establish a traditional seismological baseline using STA/LTA.
- Develop a feature-based machine learning detector using Random Forest.
- Develop a 1D convolutional neural network for direct waveform classification.
- Develop a P-wave detection and picking workflow.
- Compare traditional and machine learning approaches using quantitative evaluation metrics.
- Investigate the suitability of computational methods for passive seismic monitoring.

---

## Project Workflow

                    INSTANCE seismic waveforms
                              |
                              v
                         Preprocessing
                              |
                              v
                     +------------------+
                     |     Stage 1      |
                     | Earthquake       |
                     | Detection        |
                     +------------------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
           STA/LTA       Random Forest       1D CNN
              |               |               |
              +---------------+---------------+
                              |
                              v
                   Earthquake / Noise
                       Classification
                              |
                              v
                     +------------------+
                     |     Stage 2      |
                     |   P-Wave         |
                     |   Picking        |
                     +------------------+
                              |
                              v
                       P-Wave Arrival
                           Time