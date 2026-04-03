# 🧪 Low-Cost FF-OCT Image Processing Pipeline

This repository presents a **step-by-step image processing algorithm** for Full-Field Optical Coherence Tomography (FF-OCT) systems.  
The pipeline extracts phase information from sequential frames using signal processing techniques such as FFT filtering and the Hilbert transform.

---

## 📌 Overview

The processing workflow includes:

1. Frame acquisition and preprocessing  
2. Background removal  
3. Denoising and contrast enhancement  
4. ROI-based signal extraction  
5. Frequency filtering (FFT)  
6. Phase extraction (Hilbert transform)  
7. Phase-step detection (0, π/2, π, 3π/2)

---

## ⚙️ Requirements

```bash
pip install numpy opencv-python matplotlib scipy
```

---

## 📥 Imports

```python
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from scipy.signal import find_peaks, hilbert
```

---

## 1️⃣ Load Frames

```python
def normalize(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def load_frames(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    frames = [
        normalize(cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_GRAYSCALE)[100:858, 100:856])
        for f in files
    ]
    return np.array(frames)

path = r'YOUR_PATH_HERE'
frames = load_frames(path)
```

---

## 2️⃣ Background Removal

```python
background = np.mean(frames[0:1000], axis=0)
frames_no_bg = frames - background
```

---

## 3️⃣ Denoising & Contrast Enhancement

```python
img_clahe_np = []

for ff in frames_no_bg:
    nn = normalize(ff)

    img_denoised = cv2.fastNlMeansDenoising(
        nn, h=10, templateWindowSize=7, searchWindowSize=21
    )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_denoised)

    img_clahe_np.append(img_clahe)

img_clahe_np = np.array(img_clahe_np)
```

---

## 4️⃣ ROI Extraction

```python
image_filter = []

for ff in img_clahe_np:
    roi = ff[y1:y2, x1:x2]
    image_filter.append(np.mean(roi))

image_filter = np.array(image_filter)
np.save('roi_signal.npy', image_filter)
```

---

## 5️⃣ FFT Filtering

```python
filtered_signal = np.load('roi_signal.npy').flatten()

fft_signal = np.fft.fft(filtered_signal)
freqs = np.fft.fftfreq(len(filtered_signal), d=1)

low_cutoff = 0.01
high_cutoff = 0.15

fft_bandpass = fft_signal.copy()
fft_bandpass[np.abs(freqs) < low_cutoff] = 0
fft_bandpass[np.abs(freqs) > high_cutoff] = 0

bandpassed_signal = np.fft.ifft(fft_bandpass).real
```

---

## 6️⃣ Phase Extraction

```python
analytic_signal = hilbert(bandpassed_signal)
instantaneous_phase = np.angle(analytic_signal)
```

---

## 7️⃣ Phase Detection

```python
def extract_phase_points(phase_array, target_phase, label, atol=0.05, min_distance=10):
    indices = np.where(np.isclose(phase_array, target_phase, atol=atol))[0]

    if label in ['π', '-π']:
        indices = np.where((phase_array > 3.0) | (phase_array < -3.0))[0]

    selected = []
    prev = -min_distance

    for idx in indices:
        if idx - prev >= min_distance:
            selected.append(idx)
            prev = idx

    return np.array(selected)
```

---

## 📊 Visualization

```python
phase_targets = {'0': 0, 'π/2': np.pi/2, 'π': np.pi, '3π/2': -np.pi/2}
phase_colors = {'0': 'k', 'π/2': 'r', 'π': 'b', '3π/2': 'g'}
phase_markers = {'0': 'x', 'π/2': '^', 'π': 'o', '3π/2': 's'}

plt.figure(figsize=(15, 5))
plt.plot(bandpassed_signal, label='Bandpassed Signal')

for label, value in phase_targets.items():
    pts = extract_phase_points(instantaneous_phase, value, label)
    plt.plot(
        pts,
        bandpassed_signal[pts],
        linestyle='None',
        marker=phase_markers[label],
        color=phase_colors[label],
        label=f'Phase ≈ {label}'
    )

plt.legend()
plt.grid()
plt.xlabel("Frame Index")
plt.ylabel("Intensity")
plt.title("Phase Detection")
plt.show()
```

---

## 📈 Metrics

```python
mu_signal = np.mean(signal_region)
mu_background = np.mean(background_region)

sigma_signal = np.std(signal_region)
sigma_background = np.std(background_region)

SNR = mu_signal / sigma_background
CNR = (mu_signal - mu_background) / np.sqrt(sigma_signal**2 + sigma_background**2)
```

---
