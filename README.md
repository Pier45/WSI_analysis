# WSI Analysis — Bayesian Framework for Histopathological Image Analysis

> **Thesis**: *"Applicazione di reti bayesiane all'analisi automatica di immagini istopatologiche"*  
> **Author**: Piero Policastro — Politecnico di Torino, Ingegneria Biomedica, A.Y. 2019–2020  
> **PDF**: [webthesis.biblio.polito.it/13803](https://webthesis.biblio.polito.it/13803/)

---

## Overview

This toolkit applies **Bayesian deep learning** to the automatic analysis of colorectal histopathological whole-slide images (WSIs).

![Figure 4.3](img/Figure4.3.png)

The key contribution over classical CNNs is the ability to **quantify prediction uncertainty**, enabling the system to signal when a classification is unreliable — a critical property in clinical decision support.

Two uncertainty components are estimated via **Monte Carlo Dropout**:

| Uncertainty | Source | Meaning |
|---|---|---|
| **Aleatoric** | Noise in the data | Irreducible; related to image quality or ambiguous tissue |
| **Epistemic** | Model limitations | Reducible with more or better training data |

The framework implements two distinct Bayesian architectures, exposed through two PyQt5 graphical applications.

---

## Scientific Background

### Why Bayesian Networks?

Classical neural networks produce deterministic predictions — they assign a class label with no measure of confidence. Bayesian CNNs treat weights as **probability distributions** rather than fixed values, so running inference multiple times on the same input yields a distribution over predictions.

The posterior is approximated via two approaches:

1. **KL Divergence** (Variational Inference) — uses `Conv2DFlipout` and `DenseFlipout` layers from `tensorflow_probability`. Theoretically rigorous but computationally expensive and harder to converge.
2. **Monte Carlo Dropout** (Gal & Ghahramani, 2015) — keeps `Dropout` active at inference time. Mathematically equivalent to a Bayesian approximation with Gaussian weight priors. Faster and more stable.

> Gal & Ghahramani demonstrated that applying dropout before every weight layer is mathematically equivalent to a Bayesian system with Gaussian weight distributions.

### Dataset

The dataset consists of colorectal tissue WSIs from 27 patients (9 per class), provided by the **University of Leeds Virtual Pathology** repository.

**Three tissue classes:**

| Label | Description |
|---|---|
| `AC` | Adenocarcinoma — malignant epithelial tumour |
| `AD` | Adenoma — benign glandular lesion |
| `H` | Healthy colorectal tissue |

**Tile extraction** was performed at maximum resolution (level 0) using a 256×256 px sliding window without overlap. Images were then downscaled to 64×64 px for training, yielding equivalent accuracy at significantly lower computational cost.

![Figure 4.2](img/Figure4.2.png)

**Final dataset composition:**

| Split | Samples per class | Total |
|---|---|---|
| Train | 15 000 | 45 000 |
| Validation | 6 400 | 19 200 |
| Test | 2 700 | 8 100 |

Splits were constructed **patient-wise** — no patient appears in more than one set — to prevent data leakage.

### Model Architecture (Monte Carlo Dropout)

The backbone is a custom CNN with dropout applied at every stage (active at inference for MC sampling).

**Convolutional blocks** (×5, each = Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout):

| Block | Filters | Kernel | Pooling | Dropout |
|---|---|---|---|---|
| 1 | 16 | 6×6 | ✓ | 0.15 |
| 2 | 32 | 6×6 | ✓ | 0.25 |
| 3 | 64 | 6×6 | ✓ | 0.25 |
| 4 | 128 | 4×4 | ✓ | 0.25 |
| 5 | 256 | 4×4 | ✗ | 0.30 |
| 6 | 1024 | 3×3 | ✓ | — |

**Dense head** (after Flatten):

| Layer | Units | Dropout |
|---|---|---|
| Dense 1 | 1024 | 0.35 |
| Dense 2 | 364 | 0.25 |
| Dense 3 | 256 | — |
| Output | 3 | Softmax |

Input: 64×64×3 RGB. Activation: ReLU throughout. Loss: Categorical Cross-Entropy. Optimizer: Adadelta.

### Uncertainty Formulas

For each tile, the model is run **N times** (Monte Carlo samples). Let xᵢ be the softmax output of run *i*:

```
Epistemic = (1/N) Σ xᵢ² − [(1/N) Σ xᵢ]²

Aleatoric  = (1/N) Σ xᵢ(1 − xᵢ)

Total      = Epistemic + Aleatoric
```

### Results

**Baseline dropout model (full dataset):**

| Split | Accuracy |
|---|---|
| Train | 86.3% |
| Validation | 73.2% |
| Test | 68.1% |

**After manual data cleaning + data augmentation:**

| Split | Accuracy |
|---|---|
| Train | 79.7% |
| Validation | 77.1% |
| Test | **76.1%** |

**After automatic Bayesian data cleaning (New threshold) + data augmentation:**

| Split | Accuracy |
|---|---|
| Train | 89.3% |
| Validation | 85.7% |
| Test | **79.3%** |

> The combination of Bayesian data cleaning and data augmentation yielded an **~11 percentage point improvement** on the test set over the baseline.

### Data Cleaning via Uncertainty

The uncertainty histogram of any dataset is **bimodal** — tiles with correct predictions cluster near 0, while noisy or ambiguous tiles cluster near 0.5. Two automatic thresholding strategies are implemented:

- **Otsu threshold** (T₁): maximises inter-class variance on the uncertainty histogram. Aggressive — removes ~60% of tiles. Improves in-distribution accuracy but can hurt generalisation.
- **New threshold** (T₂): starts from T₁, finds the next peak in the histogram, then locates the point of maximum variation in [T₁, peak]. More conservative — retains ~67% of tiles with a more balanced class distribution.

Tiles with uncertainty **below** the selected threshold are kept; the rest are discarded.

---

## Repository Structure

```
WSI_analysis/
├── archive/                          # Archived experiments and old model runs
├── deepzoom/
│   └── deepzoom_server.py            # Local Flask-based DeepZoom viewer
├── icons/                            # UI icon assets
├── img/                              # Images used in README and documentation
├── styles/
│   ├── stile.txt                     # Qt stylesheet (plain text)
│   └── stileor.css                   # Qt stylesheet (CSS)
├── test/                             # Test scripts and assets
├── Classification.py                 # Model inference + uncertainty maps
├── DatasetCreation.py                # TFRecord creation utility
├── dictionary_5_js.txt               # Example uncertainty dictionary (MC=5)
├── Dockerfile                        # Container definition
├── DropOut.py                        # MC-Dropout CNN training class
├── keras_kl.py                       # Additional Bayesian architecture support
├── Kl.py                             # KL-divergence Bayesian model training
├── multi_processing_analysis.py      # Tile extraction from .svs files
├── new_train_js.txt                  # Example training set JSON
├── new_val_js.txt                    # Example validation set JSON
├── progress_bar.py                   # Qt progress dialog helper
├── requirements.txt                  # Pinned Python dependencies
├── test_widget.py                    # Confusion matrix widget
├── ui_dataclean.py                   # Data cleaning & training GUI (entry point 2)
└── ui_pyqt5.py                       # Main WSI analysis GUI (entry point 1)
```

---

## Requirements

All dependencies are fully pinned in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Version | Role |
|---|---|---|
| `tensorflow` | 2.15.0 | Model training and inference |
| `keras` | 2.15.0 | High-level neural network API |
| `numpy` | 1.26.4 | Numerical computation |
| `PyQt5` | 5.15.10 | GUI framework |
| `openslide-python` | 1.4.3 | WSI file reading |
| `openslide-bin` | 4.0.0.13 | OpenSlide native binaries |
| `scikit-learn` | 1.4.1 | Metrics and data utilities |
| `matplotlib` / `seaborn` | 3.8.4 / 0.13.0 | Plotting |
| `pandas` | 2.2.0 | Data handling |
| `scipy` | 1.12.0 | Otsu thresholding and signal processing |

Two optional dependencies are **not** included in `requirements.txt` and must be installed separately if needed:

```bash
# KL-divergence model (Kl.py) only
pip install tensorflow-probability

# DeepZoom server (deepzoom/deepzoom_server.py) only
pip install flask
```

---

## Usage

### 1. WSI Analysis GUI

```bash
python ui_pyqt5.py
```

![Figure5.1](img/Figure5.1.png)

**Workflow:**
1. **File → Select SVS** — opens a `.svs` file; a thumbnail is generated immediately and tiles are extracted in the background using all available CPU threads.
2. **Analysis → Start** (or `Ctrl+R`) — runs the Bayesian classifier. A progress bar tracks completion.
3. **View** menu — inspect results per class (`AC` / `AD` / `H`) or view uncertainty maps (total / aleatoric / epistemic).
4. **Options → Deep Zoom Viewer** (or `Ctrl+D`) — launches a local Flask server and opens the slide at full resolution in the browser via OpenSeadragon.

**Output files** (saved under `result/` in the working directory):

| File | Content |
|---|---|
| `Pred_class.png` | All-class overlay on greyscale thumbnail |
| `AC.png`, `AD.png`, `H.png` | Single-class overlays |
| `result/uncertainty/tot.png` | Total uncertainty map |
| `result/uncertainty/ale.png` | Aleatoric uncertainty map |
| `result/uncertainty/epi.png` | Epistemic uncertainty map |
| `dictionary_monte_{N}.txt` | JSON with per-tile path, class, epistemic and aleatoric values |

**JSON record format (analysis):**
```json
{
  "100": {
    "im_path": "C:/…/tile_100_3_15.png",
    "shape_x": 64,
    "shape_y": 64,
    "col": 3,
    "row": 15,
    "class": "AC",
    "epi": 0.1828,
    "ale": 0.3624
  }
}
```

![Figure5.23](img/Figure5.2-5.3.png)

### 2. Data Cleaning & Training GUI

```bash
python ui_dataclean.py
```

![Figure5.7](img/Figure5.7.png)

The interface is organised in five tabs:

| Tab | Purpose |
|---|---|
| **Get Tiles** | Select `.svs` folders (one per class: `AC`, `AD`, `H`) and extract tiles for train / val / test |
| **Training** | Choose model (Dropout or KL), set epochs, batch size, and data augmentation; live per-batch accuracy stream |
| **Uncertainty Analysis** | Classify a set with the trained model; view epistemic, aleatoric, and total uncertainty histograms |
| **Data Cleaning** | Select Otsu / New / Manual threshold; preview removed tiles per class (pie chart); export cleaned dataset |
| **Testing** | View overall and per-patient confusion matrices |

**JSON record format (data cleaning, extends analysis format):**
```json
{
  "0": {
    "name": "pz_42_AD_2",
    "true_class": "AD",
    "im_path": "C:/…/train/AD/pz_42_AD_2_tile_0_0_0.png",
    "shape_x": 64,
    "shape_y": 64,
    "col": 0,
    "row": 0,
    "pred_class": "AC",
    "epi": 0.0028,
    "ale": 0.1229
  }
}
```

### 3. DeepZoom Server (standalone)

```bash
python deepzoom/deepzoom_server.py <path/to/slide.svs>
```

Opens a browser tab at `http://127.0.0.1:5000/` with adaptive-resolution tile streaming via **OpenSeadragon**. The right panel shows all Aperio metadata; the left panel shows any associated images embedded in the `.svs` file.

> ⚠️ The file path must **not contain spaces** (Flask limitation on Windows).

---

## Docker

A `Dockerfile` is included in the repository. The container supports X11 forwarding for the PyQt5 GUI when running under WSL2.

**Build the image:**
```bash
docker build -t wsi-analysis .
```

**Run the container:**
```bash
docker run --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    wsi-analysis
```

**Run with a local data folder mounted:**
```bash
docker run --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /path/to/your/data:/data \
    wsi-analysis
```

Replace `/path/to/your/data` with the absolute path to the folder containing your `.svs` files. Inside the container the data will be accessible at `/data`.

---

## Key Design Decisions

- **Multiprocessing tile extraction**: the number of worker threads is detected automatically from the CPU thread count, so tile creation scales to available hardware without manual configuration.
- **Monte Carlo Dropout at inference**: `Dropout` layers are instantiated with `training=True`, keeping them active during prediction. Running N forward passes yields a prediction distribution from which uncertainty is derived analytically.
- **Patient-wise dataset split**: train, validation, and test sets are built from disjoint patient groups, preventing any form of patient-level data leakage.
- **Zero-padding for border tiles**: tiles at image borders are often smaller than 64×64 px. Zero-padding is applied before inference and the original dimensions are stored in the JSON to correctly reconstruct the overlay masks.

---

## Notes

- All models expect 64×64 RGB input tiles.
- The `DropOut.py` architecture is recommended over `Kl.py` for faster training, better convergence, and broader library compatibility.
- DeepZoom paths must not contain spaces (Flask limitation on Windows).
- Developed and tested on Windows (Intel i7-7700, 8 threads); also validated on Google Colab (NVIDIA Tesla K80) and HPC Polito (NVIDIA Tesla V100).

---

## References

1. Gal, Y. & Ghahramani, Z. — *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*, 2015.
2. Shridhar, K. et al. — *Bayesian Convolutional Neural Networks with Variational Inference*, 2018.
3. Ioffe, S. & Szegedy, C. — *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, 2015.
4. Blundell, C. et al. — *Weight Uncertainty in Neural Networks*, 2015.

---

## License

MIT