## MICRIGRID FAULT CLASSIFICATION USING HYBRID DEEP LEARNING ARCHITECTURE (Conv1D + Bi-GRU + Attention) 

## Overview

Microgrids (MGs) are localized energy systems that integrate renewable energy sources, storage devices, and distributed generators. They can operate autonomously or in coordination with the main grid, improving reliability, sustainability, and resilience.

However, fault detection in microgrids is far more challenging compared to conventional grids due to:

- Bidirectional power flows

- Varying fault currents

- Non-linear, dynamic operating conditions

Traditional methods (overcurrent relays, sequence component analysis, impedance-based approaches) often fail in such scenarios. To overcome these challenges, machine learning and deep learning approaches are increasingly applied for fault detection and classification.

👉 To address this, This project proposes a hybrid Conv1D + BiGRU + Attention model that captures both spatial (feature extraction via Conv1D) and temporal dependencies (via BiGRU + Attention) in current signals. This allows more robust and interpretable fault classification in AC microgrids.

## Features

- Automated preprocessing of fault current data (Ia, Ib, Ic)

- Windowing to preserve temporal sequences

- Hybrid deep learning architecture:
    - Conv1D for local feature extraction

    - BiGRU for bidirectional temporal learning

    - Custom Attention layer for interpretability (weights highlight important time steps)

- Comprehensive evaluation with confusion matrix, accuracy/loss plots, and tabular reports

- Achieves ~98% accuracy on test data, outperforming baseline methods

## Tech Stack
- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

# Dataset
Features used:
- Absolute Ia, Absolute Ib, Absolute Ic (extracted from raw currents)

Labels:
- Normal (0), Fault (1)

- Faults defined in time window 0.2s – 0.4s [ from MATLAB Simulink simulations]

# Model Architecture
Input → Conv1D(128, ReLU) → MaxPooling  
      → BiGRU(64, bidirectional)  
      → Simple Attention  
      → Dense(64, ReLU)  
      → Dense(2, Softmax)


Conv1D: learns local features from current signals

BiGRU: captures bidirectional temporal dependencies

Attention: focuses on the most informative parts of the signal

Softmax Output: provides interpretable fault probability distribution

## Results
Final Test Accuracy: ~98%