# WSI_analysis

WSI_analysis is a research-oriented Python toolkit for whole-slide image (WSI) analysis, classification, uncertainty estimation, and dataset cleaning. It provides both command-line modules and PyQt5 graphical interfaces for tile extraction, model inference, uncertainty visualization, and training.

## Key features

- Extract tiles from `.svs` whole-slide images using OpenSlide and multiprocessing
- Generate thumbnails and tile datasets for downstream classification
- Run classification with a trained TensorFlow/Keras model
- Estimate aleatoric and epistemic uncertainty via Monte Carlo sampling
- Create overlay masks for predicted classes and uncertainty maps
- Visualize results with a native PyQt5 GUI and a local DeepZoom web viewer
- Perform uncertainty-based dataset cleaning and threshold selection
- Train models with dropout-based and KL-based Bayesian architectures

## Repository structure

- `multi_processing_analysis.py` - tile extraction from WSI files using `openslide.deepzoom.DeepZoomGenerator`
- `Classification.py` - loads tile images, runs model prediction, computes uncertainty, and creates overlay masks
- `ui_pyqt5.py` - main GUI for whole-slide analysis, thumbnail preview, and DeepZoom visualization
- `ui_dataclean.py` - data cleaning / training GUI with uncertainty analysis and dataset management
- `uncertainty_analysis.py` - threshold estimation using Otsu-like uncertainty analysis and clean dataset generation
- `DropOut.py` - dropout-based CNN training support class
- `Kl.py` - KL-divergence / Bayesian model training support class
- `DatasetCreation.py` - TFRecord creation utility for image datasets
- `deepzoom/deepzoom_server.py` - local Flask-based DeepZoom viewer for `.svs` slides
- `progress_bar.py` - Qt progress dialog helper
- `test_widget.py` - confusion matrix visualization widget used by the GUI
- `dictionary_5_js.txt`, `new_train_js.txt`, `new_val_js.txt` - example JSON dictionaries / dataset files
- `Model_1_85aug.h5` - pre-trained model file included in repository

## Requirements

- Python 3
- PyQt5
- TensorFlow (compatible version for `tf.keras` and `tf.python_io` usage)
- OpenSlide Python bindings
- Pillow
- NumPy
- SciPy / scikit-learn
- Matplotlib
- Seaborn
- Flask (for `deepzoom/deepzoom_server.py`)
- pandas

> Note: The repository code appears to target a Windows environment and uses Windows-style paths in many examples. DeepZoom paths should not contain spaces.

## Usage

### 1. Tile extraction

Use `multi_processing_analysis.py` or the GUI in `ui_pyqt5.py` to extract tiles from `.svs` files.

- `multi_processing_analysis.py`: extracts tiles and saves them as `.png` files from a whole-slide image
- `ui_pyqt5.py`: open a `.svs`, create thumbnail, and manage analysis flows

### 2. Classification and uncertainty maps

Use `Classification.py` or the GUI in `ui_pyqt5.py` to:

- load extracted tiles
- run a TensorFlow model (e.g. `Model_1_85aug.h5`)
- compute prediction map and uncertainty overlays
- save results under `result/` and `result/uncertainty/`

### 3. DeepZoom viewing

Launch the local DeepZoom server to inspect WSI content at high resolution.

```bash
python deepzoom/deepzoom_server.py <path_to_slide.svs>
```

Then open the browser URL shown by the server.

### 4. Data cleaning and model training

Use `ui_dataclean.py` for dataset preparation, uncertainty analysis, and training workflows.

- organize image tiles into class folders (`AC`, `H`, `AD`)
- use the GUI to select train/validation/test datasets
- perform uncertainty analysis and apply thresholds
- train models using either dropout-based or KL-based architectures

## Class labels and dataset structure

The repository assumes class folders named:

- `AC`
- `H`
- `AD`

For data cleaning and training, the code expects these class folders when scanning image directories.

## Notes

- The model training code in `DropOut.py` and `Kl.py` is designed for 64x64 RGB images.
- The `Classification` module uses Monte Carlo sampling to compute uncertainty estimates.
- `uncertainty_analysis.py` includes Otsu-like thresholding and dataset reduction based on uncertainty.
- `ui_pyqt5.py` and `ui_dataclean.py` are the main application entry points for graphical workflows.

## Launch commands

```bash
python ui_pyqt5.py
python ui_dataclean.py
python deepzoom/deepzoom_server.py <slide.svs>
```

## Additional files

- `progress_bar.py`: UI progress dialog helper for long-running operations
- `test_widget.py`: confusion matrix test tab used in the GUI
- `keras_kl.py`: additional Bayesian model architecture support

## License

This repository does not include an explicit license file.
