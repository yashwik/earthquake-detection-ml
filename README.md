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

## Project Workflow

The project is organized as a two-stage workflow:

```text
INSTANCE seismic waveforms
          |
          v
      Preprocessing
          |
          v
     Stage 1: Earthquake Detection
       /        |        \\
   STA/LTA      RF       1D CNN
       \\        |        /
        Earthquake / Noise
              |
              v
     Stage 2: P-Wave Picking
       /        |        \\
   STA/LTA    1D CNN    1D U-Net
       \\        |        /
        P arrival time
```


## Dataset

The project uses the sample subset of the **INSTANCE** seismic dataset developed by the Istituto Nazionale di Geofisica e Vulcanologia (INGV).

The sample data include:

- 10,000 earthquake recordings
- 1,000 noise recordings
- 3-component waveforms: E, N, and Z
- 100 Hz sampling rate
- 120-second waveform duration
- Annotated P-wave arrival samples

Only the sample subset is used in this project; the complete INSTANCE dataset is substantially larger.

### INSTANCE resources

- GitHub: https://github.com/INGV/instance
- INGV: https://www.pi.ingv.it/banche-dati/instance/
- Dataset paper: https://essd.copernicus.org/articles/13/5509/2021/

## Preprocessing

The seismic waveforms are processed using:

- Constant detrending
- 3–20 Hz fourth-order Butterworth band-pass filtering
- Zero-phase filtering
- Float32 conversion

The three-component waveform is retained for the machine-learning models. The Z component is used for the traditional STA/LTA baselines.

## Stage 1: Earthquake Detection

Stage 1 determines whether a waveform contains an earthquake signal or noise.

### STA/LTA

Short-Term Average / Long-Term Average is used as the traditional signal-processing baseline.

### Random Forest

The Random Forest classifier uses waveform-derived features from the E, N, and Z components:

- RMS
- Peak amplitude
- Energy
- Dominant frequency
- Spectral centroid

The final full-data model uses 300 trees, maximum depth 15, minimum samples split 5, and minimum samples leaf 2. The decision threshold is selected using the validation set.

### 1D CNN

A one-dimensional convolutional neural network operates directly on the three-component waveform and learns waveform representations without manually engineered input features.

### Stage 1 Results

Final held-out test results:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| STA/LTA | 69.77% | 98.59% | 67.98% | 80.47% |
| Random Forest | **95.99%** | **97.52%** | 98.12% | **97.82%** |
| 1D CNN | 93.04% | 93.88% | **98.85%** | 96.30% |

Random Forest achieves the strongest overall classification performance. The 1D CNN provides the highest recall, while STA/LTA is conservative and produces relatively few false positives.

## Stage 2: P-Wave Picking

Stage 2 estimates the arrival time of the P-wave on the complete 120-second earthquake waveform.

Performance is evaluated by comparing predicted arrival times with the annotated P-wave arrival samples.

### Data Split

Earthquake recordings are split by `source_id` to reduce source-level data leakage.

```text
Train:      6761
Validation: 1593
Test:       1646
```

The validation set is used for model and picking-rule selection. The held-out test set is reserved for final evaluation.

### STA/LTA P-Wave Picker

STA/LTA is applied to the preprocessed Z component.

Configuration:

- Sampling rate: 100 Hz
- STA window: 1 s
- LTA window: 10 s
- Trigger ON threshold: 3.0
- Trigger OFF threshold: 1.5

The first trigger onset is used as the predicted P-wave arrival.

### Baseline 1D CNN

The baseline CNN receives the full three-component waveform:

```text
Input:  3 × 12000
Output: 12000 sample-level scores
```

A Gaussian target centered on the annotated P-wave arrival is used for training, with sigma = 0.20 seconds. Weighted binary cross-entropy is used because the target is highly sparse.

### 1D U-Net

The U-Net is a one-dimensional convolutional architecture with encoder-decoder layers and skip connections.

Skip connections preserve higher-resolution temporal information while the deeper layers capture larger-scale waveform context.

The U-Net experiment uses a Gaussian target with sigma = 0.30 seconds and a validation-selected output threshold for P-wave candidate selection.

## Stage 2 Results

Final held-out test results:

| Model | MAE (s) | Median Error (s) | Within ±0.1 s | Within ±0.5 s | Within ±1 s | Within ±2 s |
|---|---:|---:|---:|---:|---:|---:|
| STA/LTA | **2.72** | 0.25 | 30.78% | 63.42% | 72.15% | 76.42% |
| 1D CNN | 4.95 | 0.11 | 49.15% | **82.02%** | **83.96%** | 84.99% |
| 1D U-Net | 3.85 | **0.10** | **50.91%** | 78.49% | 83.29% | **85.18%** |

### Timing Metric Interpretation

For a tolerance of ±1 second, the reported percentage is the proportion of test waveforms for which the predicted P-wave arrival is no more than 1 second earlier or later than the annotated arrival.

For example, the U-Net result of **83.29% within ±1 second** means that 83.29% of test waveforms have a P-wave prediction within 1 second of the annotated arrival.

The tolerance percentages are cumulative: a pick within ±0.5 seconds is also within ±1 second and ±2 seconds.

### Stage 2 Interpretation

STA/LTA achieves the lowest mean absolute error. The baseline CNN provides strong tolerance-based timing accuracy, including 82.02% of predictions within ±0.5 seconds. The U-Net achieves the lowest median timing error and the highest proportion of predictions within ±2 seconds, while reducing MAE relative to the baseline CNN.

The results show that no method is uniformly best across all metrics. STA/LTA remains a strong traditional baseline, while the CNN-based methods provide strong typical timing accuracy.

## Error Analysis

The baseline CNN occasionally produced very large late false peaks. In several validation examples, the network produced a local peak close to the true P arrival but selected a much later global maximum.

This behavior motivated the U-Net experiment and threshold-based candidate selection.

The U-Net reduced the large-error contribution relative to the baseline CNN, reflected by a reduction in test MAE from 4.95 seconds to 3.85 seconds.

## Repository Structure

```text
earthquake-detection-ml/
│
├── data/
│   ├── sample/
│   │   ├── data/
│   │   └── metadata/
│   └── processed/
│       └── stage2_cnn/
│           ├── train_waveforms.h5
│           ├── validation_waveforms.h5
│           └── test_waveforms.h5
│
├── notebooks/
│   ├── Waveforms.ipynb
│   ├── 01_stage1_dataset.ipynb
│   ├── 02_stage1_random_forest.ipynb
│   ├── 03_stage1_random_forest_full.ipynb
│   ├── 04_stage1_cnn.ipynb
│   ├── 05_stage2_sta_lta.ipynb
│   ├── 06_stage2_cnn.ipynb
│   ├── 07_stage2_cnn_unet.ipynb
│   └── 08_final_results.ipynb
│
├── src/
│   ├── features.py
│   └── ...
│
├── results/
│   ├── metrics/
│   └── figures/
│
├── requirements.txt
└── README.md
```

## Notebook Guide

- `Waveforms.ipynb`: official INSTANCE waveform exploration notebook.
- `01_stage1_dataset.ipynb`: Stage 1 dataset creation, preprocessing, and STA/LTA development baseline.
- `02_stage1_random_forest.ipynb`: development Random Forest experiment.
- `03_stage1_random_forest_full.ipynb`: full Stage 1 Random Forest training and evaluation.
- `04_stage1_cnn.ipynb`: Stage 1 CNN training and evaluation.
- `05_stage2_sta_lta.ipynb`: traditional STA/LTA P-wave picking.
- `06_stage2_cnn.ipynb`: baseline Stage 2 CNN P-wave localization.
- `07_stage2_cnn_unet.ipynb`: improved U-Net P-wave localization experiment.
- `08_final_results.ipynb`: final result tables and figures.

## Results Files

Final numerical results are stored under:

```text
results/metrics/
```

Important files include:

- `stage1_results.csv`
- `stage1_development_results.csv`
- `stage2_sta_lta_picking.csv`
- `stage2_cnn_picking.csv`
- `stage2_unet_picking.csv`
- `stage2_results.csv`
- `final_project_results.csv`

Final figures are stored under:

```text
results/figures/
```

## Technologies

- Python
- NumPy
- Pandas
- SciPy
- scikit-learn
- PyTorch
- ObsPy
- h5py
- Matplotlib
- Jupyter

## Geophysical Concepts

The project applies concepts from computational seismology and signal processing, including:

- Seismic waveform analysis
- P-wave arrival detection
- STA/LTA triggering
- Band-pass filtering
- Sampling and Nyquist frequency
- Signal-to-noise characteristics
- Time-domain waveform features
- Frequency-domain waveform features
- Phase picking
- Deep-learning-based temporal localization

## Limitations

- The experiments use the 10,000-event sample subset rather than the complete INSTANCE dataset.
- Stage 1 and Stage 2 are evaluated as separate tasks rather than as a fully integrated real-time system.
- Only P-wave arrival-time picking is evaluated in Stage 2.
- The U-Net uses a validation-derived output threshold before held-out test evaluation.
- Stage 2 timing-error distributions contain a small number of large errors that increase MAE.
- Performance may change for seismic networks, stations, or waveform distributions outside the sample dataset.

## Future Work

- Train on the complete INSTANCE dataset.
- Evaluate joint earthquake detection and phase picking.
- Extend picking to S-wave and additional phases.
- Compare with established phase-picking models such as PhaseNet and EQTransformer.
- Add uncertainty estimation for arrival-time predictions.
- Evaluate real-time seismic processing.
- Test cross-dataset and cross-network generalization.

## References

Menichetti, M. et al. (2021). The Italian Seismic dataset INSTANCE: A modern, FAIR seismic dataset for machine learning applications. *Earth System Science Data*.

- INSTANCE repository: https://github.com/INGV/instance
- INSTANCE dataset information: https://www.pi.ingv.it/banche-dati/instance/
- INSTANCE publication: https://essd.copernicus.org/articles/13/5509/2021/
