# WSI_analysis — Project Map

> Master's thesis: *Applicazione di reti bayesiane all'analisi automatica di immagini istopatologiche* — Piero Policastro, Politecnico di Torino, A.Y. 2019–2020.
> Bayesian deep learning for automatic classification of colorectal whole-slide images (WSIs) in `.svs` format, with aleatoric + epistemic uncertainty quantification for clinical safety.

---

## Table of Contents

1. [Top-level directory layout](#1-top-level-directory-layout)
2. [What the project does](#2-what-the-project-does)
3. [Package layout and modules](#3-package-layout-and-modules)
4. [Build / dependency / test / config files](#4-build--dependency--test--config-files)
5. [Remaining structural smells](#5-remaining-structural-smells)
6. [Bug status](#6-bug-status)
7. [Git state and governance](#7-git-state-and-governance)
8. [Quick-reference paths](#quick-reference-paths)

---

## 1. Top-level directory layout

```
WSI_analysis/
├── .git/                  # version control (LFS enabled, see §7)
├── .venv/                 # ⚠ 1.79 GB, in .gitignore but physically sits at root
├── .vscode/               # ⚠ now UNTRACKED (removed from git in v8.5.0, kept on disk)
├── .opencode/             # opencode config + graphify plugin
├── archive/               # ⚠ legacy Colab-era code + 14 MB of sample data blobs
├── data/                  # patient WSI datasets (e.g. 10002_AC_2/)
├── docs/                  # BUG_REPORT.md, PROJECT_MAP.md, sasa.txt
├── graphify-out/          # knowledge-graph outputs (graph.html, graph.json, …) — refreshed 2026-07-17
├── icons/                 # 14 PyQt5 GUI icon assets (.ico + Linux .png variants)
├── img/                   # 6 PNG thesis figures + 3 stray UI assets
├── models/                # ✓ package — Bayesian model classes (drop_out, kl, base)
├── src/                   # ✓ package — core logic (classification, config, widgets, …)
│   ├── __init__.py        # re-exports CLASS_NAMES, N_CLASSES, INPUT_SHAPE
│   ├── config.py          # single source of truth for class names / shape
│   ├── classification.py
│   ├── multi_processing_analysis.py  # moved here from root
│   ├── uncertainty_analysis.py
│   ├── performance_widget.py         # renamed from test_widget.py
│   ├── progress_bar.py
│   ├── qt_workers.py                 # ✓ shared WorkerSignals/Worker/WorkerLong (refactored out of the two GUIs)
│   └── deepzoom/          # Flask DeepZoom viewer (cleanest sub-package)
│       ├── static/        # OpenSeadragon + jQuery (~1 MB)
│       ├── templates/
│       └── deepzoom_server.py
├── styles/                # ⚠ near-duplicate Qt stylesheets (stile.txt, stileor.css)
├── tests/                 # ✓ NEW real test suite (pytest — see §4)
│   ├── conftest.py        #   root fixtures + marker gating
│   ├── analysis/          #   `pytest -m openslide` — StartAnalysis regressions (IndexError, Invalid address)
│   │   └── test_start_analysis.py
│   ├── gui/               #   `pytest -m gui` — pytest-qt, actions factory contract tests
│   │   ├── conftest.py
│   │   └── test_actions_factory.py
│   └── unit/              #   runs by default — pure-Python, no heavy deps
│       ├── test_config.py            # CLASS_NAMES / N_CLASSES / INPUT_SHAPE guards
│       └── test_tile_partition.py    # StartAnalysis.manage_process shape + value regression
├── __pycache__/           # byte-compiled artifacts
│
├── README.md              # 14,344 B — thorough documentation
├── AGENTS.md              # graphify skill invocation rules
├── LICENSE                # MIT, Copyright (c) 2026 Piero Policastro
├── pyproject.toml         # uv project — deps + [tool.pytest] + [tool.ruff] + [dependency-groups]
├── uv.lock                # 190 KB lockfile
├── .python-version        # 3.10
├── Dockerfile             # multi-stage, python:3.10-slim
├── docker-compose.yaml    # 2 services, hardcoded personal WSL path
├── .gitignore / .gitattributes
└── main.py               # ⚠ 4-line `uv init` stub (dead)
```

### Folder status grid

| Folder | Status | Notes |
|---|---|---|
| `.venv/` | ⚠ Local smell | 1.79 GB at repo root; pollutes every glob/search unless explicitly excluded. Not tracked. |
| `.vscode/` | ✅ **NOW UNTRACKED** | Removed from git in v8.5.0 (`git rm -r --cached`); `.gitignore` now excludes it. Folder kept on disk. |
| `archive/` | ⚠ Misnamed | "Archived experiments" now holds the dead `dataset_creation.py` + 14 MB of sample JSON tile dictionaries moved out of the live tree — useful for reproducibility but pollutes recursive search. |
| `data/` | ✓ Real patient data | WSI datasets (per-patient sub-folders like `10002_AC_2/`). |
| `models/` | ✓ Package | `__init__.py` re-exports the model classes. Clean. |
| `src/` | ✓ Package | `__init__.py` re-exports `CLASS_NAMES`, `N_CLASSES`, `INPUT_SHAPE`. Contains the real logic. Now also hosts shared `qt_workers.py`. |
| `src/deepzoom/` | ✓ Clean | Self-contained Flask sub-app — no changes needed. |
| `icons/` | ✓ Fine | 14 PyQt5 GUI icon assets — added `.png` variants of `target.ico` (Wayland can't load `.ico` reliably; `target.png` is used on Linux). |
| `img/` | ⚠ 3 stray files | `checkbox.png`, `down_arrow.png`, `handle.png` appear unused by the live UI. |
| `styles/` | ⚠ Duplication | `stile.txt` (11.7 KB) and `stileor.css` (11.6 KB) are near-byte-identical; only `.css` is loaded by the app. |
| `test/` | ✅ **REMOVED** | Empty `test/` folder deleted; the real test tree now lives under `tests/` (plural). |
| `tests/` | ✓ NEW | Real pytest test suite — see §4 for marker gating. |
| `graphify-out/` | ✓ Generated | Knowledge-graph outputs refreshed on 2026-07-17 (809 nodes / 1230 edges / 82 communities). |

---

## 2. What the project does

**Purpose:** apply Bayesian deep learning to automatic classification of colorectal WSIs in `.svs` format, and quantify prediction uncertainty (aleatoric + epistemic) so the system can flag unreliable classifications — a clinical safety property.

### Stack overview

| Layer | Choice |
|---|---|
| Language | Python **3.10** (`requires-python = ">=3.10,<3.12"`) |
| ML framework | TensorFlow 2.15 / Keras 2.15 |
| Probabilistic | TensorFlow Probability (KL model) |
| Desktop GUI | PyQt5 (two GUIs) |
| Web viewer | Flask (DeepZoom) |
| WSI reading | OpenSlide |
| Plotting | matplotlib, seaborn, scikit-learn |

### Domain

- **Field:** histopathology / digital pathology
- **Classes (3, canonical order):** `AC` (Adenocarcinoma), `AD` (Adenoma), `H` (Healthy)
- **Key ML approaches:**
  1. **Monte Carlo Dropout** (Gal & Ghahramani 2015) → `models/drop_out.py`
  2. **Variational inference** via KL-divergence Flipout layers → `models/kl.py`

### Entry points

```
┌──────────────────────────────────┐    ┌─────────────────────────────────────┐    ┌─────────────────────────────┐
│  python ui_pyqt5.py              │    │  python ui_dataclean.py             │    │  python -m src.deepzoom.     │
│  "Bayesian Analyzer"            │    │  5-tab tool:                        │    │     deepzoom_server <slide>  │
│                                 │    │  tile → train → uncertainty →       │    │                             │
│  • open .svs                    │    │  data cleaning → confusion matrices │    │  Standalone Flask server      │
│  • tile in background threads   │    │                                     │    │  serving DeepZoom tiles      │
│  • MC-dropout inference         │    │                                     │    │                             │
│  • per-class + uncertainty      │    │                                     │    │                             │
│    overlays                     │    │                                     │    │                             │
│  • launch DeepZoom viewer       │    │                                     │    │                             │
└──────────────────────────────────┘    └─────────────────────────────────────┘    └─────────────────────────────┘
```

All three entry points run with `CWD = repo root` (the `models.` and `src.` imports require it).

---

## 3. Package layout and modules

The repo now has a real package hierarchy: `models/` (Bayesian CNNs) and `src/` (everything else), each with `__init__.py`.

### `models/` — Bayesian uncertainty models

| File | Lines | Bytes | Responsibility |
|---|---:|---:|---|
| `models/drop_out.py` | 364 | 13,400 | `BayesianDropoutCNN` — MC-Dropout CNN. Dataclass config (`ConvBlockConfig`), type hints, logging, `training=True` keeps Dropout active at inference. Constructor: `(model_save_path, epochs, path_train, path_val, batch_size, augment)`. `train()` is optional-callback, `start_train()` is a backward-compat alias. |
| `models/kl.py` | 380 | 13,200 | `ModelKl` — KL-divergence Bayesian CNN via TF-Probability Flipout layers. Same constructor signature and `train()`/`start_train()` API as `BayesianDropoutCNN` so the GUI can drive them with one code path. Uses `tfp.layers.Convolution2DFlipout` / `DenseFlipout`. |
| `models/base.py` | 60 | 1,700 | `BayesianModel` — `@runtime_checkable Protocol` defining the unified constructor + `train()` + `start_train()` interface both classes satisfy. |
| `models/__init__.py` | 11 | 320 | Re-exports `BayesianDropoutCNN`, `ModelKl`, `BayesianModel`. |

A single `TrainingProgressCallback(Callback)` in each model file emits per-batch / per-epoch metrics to PyQt signals using the modern Keras keys (`accuracy`, `val_accuracy`).

### `src/` — core logic, widgets, config

| File | Responsibility |
|---|---|
| `src/config.py` | Single source of truth: `CLASS_NAMES = ('AC','AD','H')` (alphabetical canonical), `N_CLASSES = 3`, `INPUT_SHAPE = (64,64,3)`. Imported everywhere those constants are needed. |
| `src/classification.py` | `Classification` — loads tiles, runs N MC forward passes, computes epi/ale/tot uncertainty, paints RGBA overlays, dumps `dictionary_monte_N_js.txt`. Now uses `CLASS_NAMES` from config. |
| `src/multi_processing_analysis.py` | `StartAnalysis` — OpenSlide/DeepZoomGenerator wrapper. Tile extraction with thread partitioning. Doubles as dataset-creation helper. (Moved here from root.) |
| `src/uncertainty_analysis.py` | `Th` — loads tile JSON, computes uncertainty histogram, Otsu + custom "New threshold" peak-finding, exports cleaned datasets. Line-35 copy-paste bug fixed (`out_al = np.median(list_ale)`). |
| `src/performance_widget.py` | `PerformanceTab(QWidget)` — confusion-matrix viewer used by `ui_dataclean.py`'s "Testing" tab. **Renamed from `test_widget.py`** to stop masquerading as a test module. Uses `CLASS_NAMES` from config. |
| `src/progress_bar.py` | `Actions` — tiny Qt progress-bar helper (14 lines). Imported by `ui_pyqt5.py`. |
| `src/__init__.py` | Re-exports `CLASS_NAMES`, `N_CLASSES`, `INPUT_SHAPE` for ergonomic `from src import CLASS_NAMES`. |
| `src/deepzoom/` | Self-contained Flask DeepZoom sub-app — no changes from before, still the cleanest sub-package. |

### Root entry points and supporting modules

| File | Lines | Bytes | Responsibility |
|---|---:|---:|---|
| `ui_pyqt5.py` | 1093 | 38,400 | **Entry point 1.** `ImageViewer(QMainWindow)` — open `.svs`, background tiling, Bayesian inference, overlays, launch DeepZoom. Cross-platform DeepZoom launcher (no more Windows-only `cmd /k`). Linux-specific: `APP_ICON` picks `icons/target.png`, `setNativeMenuBar(False)` so the top menu bar renders on GNOME/KDE/Wayland, and `QT_QPA_PLATFORM=xcb` is exported so XWayland honours the system cursor theme. Inline import of `Classification` is now `from src.classification import Classification`. |
| `ui_dataclean.py` | 982 | 38,500 | **Entry point 2.** `MainWindow`/`MainTabWidget` with 5 tabs. Imports are now package-qualified: `from models.drop_out import BayesianDropoutCNN`, `from models.kl import ModelKl`, `from src.classification import Classification`, `from src.uncertainty_analysis import Th`, `from src.performance_widget import PerformanceTab`, `from src.config import CLASS_NAMES`. Constructor calls to models use the unified signature `(model_save_path, epochs, path_train, path_val, batch_size, augment)`. |
| `main.py` | 4 | 90 | ⚠ Vestigial `print("Hello from wsi-analysis!")` stub — dead `uv init` scaffolding. |

### Module interaction diagram (post-refactor)

```
                      ┌─────────────────────┐
                      │   ui_pyqt5.py       │  Entry point 1
                      │  ImageViewer        │
                      └──────────┬──────────┘
                                 │ imports
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
        ┌──────────────┐ ┌────────────┐ ┌──────────────┐
        │ src.progress │ │ models.    │ │ src.         │
        │ _bar.Actions │ │ drop_out   │ │ classification│
        │  (14 lines)  │ │ BayesianD  │ │  .Classification│
        │              │ │ ropoutCNN  │ │              │
        └──────────────┘ └────────────┘ └──────┬───────┘
                                                │ launches (subprocess)
                                                ▼
                                       ┌────────────────┐
                                       │ src.deepzoom/   │
                                       │ deepzoom_server │
                                       └────────────────┘

                      ┌─────────────────────┐
                      │  ui_dataclean.py    │  Entry point 2
                      │  MainWindow / 5 tabs │
                      └──────────┬──────────┘
                                 │ imports (all package-qualified)
       ┌──────────┬─────────────┼────────────┬───────────────┬────────────┐
       ▼          ▼             ▼            ▼               ▼            ▼
 ┌─────────┐ ┌────────────┐ ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐
 │ src.    │ │ models.    │ │ models.  │ │ src.       │ │ src.     │ │ src.     │
 │ multi_  │ │ drop_out   │ │ kl       │ │ classifica │ │ uncertain│ │ perform- │
 │ process │ │ BayesianD  │ │ ModelKl  │ │ tion.      │ │ ty_      │ │ ance_    │
 │ ing_    │ │ ropoutCNN  │ │          │ │ Classifica │ │ analysis │ │ widget   │
 │ analysis│ │            │ │          │ │ tion       │ │ Th       │ │ Perform- │
 │ StartAn │ │            │ │          │ │            │ │          │ │ anceTab  │
 └─────────┘ └────────────┘ └──────────┘ └────────────┘ └──────────┘ └─────────┘
                                                       ▲
                                                       │ shared config
                                                ┌──────┴───────┐
                                                │ src/config.py│ CLASS_NAMES = ('AC','AD','H')
                                                │ models/base  │ BayesianModel Protocol
                                                └──────────────┘
```

### Modules with remaining overlapping / unclear concerns

| Concern | Files | Note |
|---|---|---|
| Dual responsibility | `src/multi_processing_analysis.py` | Still mixes live tiling (`process_to_start`) with bulk dataset creation (`process_create_dataset` / `list_files`). Split candidate. |
| Monolith | `src/classification.py` | `overlay()` (lines 138–237) is still a single ~100-line function with hardcoded RGBA constants and four near-identical branches (AC/H/AD × epi/ale/tot). Prime split candidate. |
| Duplicate methods | `src/performance_widget.py` | `createTable` vs `createTable_sigle` still near-identical, differing only in which `QTableWidget` is filled. |
| Duplicate thread infra | `ui_pyqt5.py` vs `ui_dataclean.py` | Each GUI still has its own copy of `WorkerSignals`/`Worker`/`LongRunningWorker`. Could move to `src/qt_workers.py`. |
| `segmentation` dead Colab code | `archive/dataset_creation.py` | TF1 `tf.python_io.TFRecordWriter`, 5-class dataset, `/content/drive/My Drive/...` path. Correctly archived. |

---

## 4. Build / dependency / test / config files

| File | Status |
|---|---|
| `pyproject.toml` | ✅ Now carries the full toolchain config: `[project]` name/version/deps, `[project.optional-dependencies].test` (pytest stack + ruff), `[tool.uv] environments` (Win + Linux cross-platform locking), `[dependency-groups].dev`, `[tool.pytest.ini_options]` (testpaths=tests, strict markers/strict config, `openslide`/`tf`/`gui`/`slow` markers, warnings-as-errors with openslide PendingDeprecation ignored), `[tool.ruff]` (line-length 100, py310, excludes `archive`/.venv/graphify-out/deepzoom static), `[tool.ruff.lint]` (E/F/I/W/UP/B, ignores E501/B008, per-file leniency for `tests/**` and `ui_*.py`). Still missing: `[project.scripts]`, `[build-system]` — `models/`/`src/` are importable as **namespace packages** only when `CWD = repo root`; not pip-installable. |
| `uv.lock` | ✓ Present (190 KB). Cross-platform resolution for `win32` + `linux`. |
| `requirements.txt` | ✅ **DELETED** long ago — `pyproject.toml` is the single source of truth. |
| `archive/original_requirements.txt` | Ancient requirements file (TF 1.15, 2019). |
| `Dockerfile` | Multi-stage, `python:3.10-slim`, installs Qt X11 libs for WSL2 GUI forwarding. `CMD ["python", "ui_dataclean.py"]` — only launches entry point 2, not parametrised. |
| `docker-compose.yaml` | ⚠ Two services (`wsi-clean`, `wsi-analysis`) both `build: .` (same image), differing only by `command:`; mounts `/mnt/c/Users/piero/Documents/Data WSI:/data` — hardcoded to the author's personal WSL path. Not portable. |
| `.python-version` | 3.10 — standard (pyenv/uv). |
| `.vscode/settings.json` | ✅ **Now gitignored and untracked** (v8.5.0). Kept locally for the author's editor only. |
| `LICENSE` | MIT — present and standard. |
| `src/config.py` | ✓ `CLASS_NAMES = ('AC','AD','H')`, `N_CLASSES`, `INPUT_SHAPE`. Imported by `models/drop_out.py`, `models/kl.py`, `src/classification.py`, `src/performance_widget.py`, `ui_dataclean.py`, `tests/unit/test_config.py`. |
| `models/base.py` | ✓ `BayesianModel` Protocol formalising the shared constructor + `train()` interface. |
| `src/qt_workers.py` | ✓ Present (157 lines) — shared `WorkerSignals`/`Worker`/`WorkerLong` moved out of the two GUIs. (The two `ui_*.py` still keep their own copies in the short term, see §5 smell #9.) |

### Test suite (`tests/`) — NEW since v8.0.55

Run with `uv sync --extra test` then `uv run pytest` (or `uv run --group dev pytest`). pytest is configured in `pyproject.toml`:

```bash
uv run pytest                       # default: unit tests only
uv run pytest -m openslide          # StartAnalysis regression tests (libopenslide + svs fixture)
uv run pytest -m tf                 # TensorFlow-dependent tests
uv run pytest -m gui                 # pytest-qt, needs QT_QPA_PLATFORM=offscreen
uv run pytest -m "not slow"          # exclude slow tests from CI
```

| Sub-tree | Marker | What it tests |
|---|---|---|
| `tests/unit/test_config.py` | (default) | `CLASS_NAMES`/`N_CLASSES`/`INPUT_SHAPE` values are stable and as documented. |
| `tests/unit/test_tile_partition.py` | (default) | `StartAnalysis.manage_process()` return-shape contract (always 5-tuple of equal-length lists) + reference-implementation value regression. Catches the `IndexError: tuple index out of range` regression seen in `tests/analysis`. |
| `tests/analysis/test_start_analysis.py` | `openslide` | OpenSlide / DeepZoomGenerator regressions: `lev_sec` out-of-range no longer raises `IndexError`; `get_thumb` selects a valid OpenSlide level; `tile_gen` returns a DeepZoom (not OpenSlide) level index; `Invalid address` `tileGen(state=0)` crash fixed. |
| `tests/gui/test_actions_factory.py` | `gui` | `_make_action` factory contract: every action created by `ImageViewer._create_actions` has a non-empty text, non-null `triggered` slot, and a unique object name; `_create_actions` uses the factory consistently. Uses pytest-qt + `QT_QPA_PLATFORM=offscreen`. |

`tests/conftest.py` and `tests/gui/conftest.py` own marker gating and fixtures (`qapp`, `svs_path`, `analysis`). Strict markers (`--strict-markers --strict-config`) guard against typos in new markers.

### Still missing / non-standard

- ❌ No `setup.py` / `setup.cfg` / `[build-system]` / `[project.scripts]` — `models/` and `src/` are importable as **namespace packages** when `CWD = repo root`, but there's no installation story. The two GUI entry points are not declared as console scripts.
- ❌ No `Makefile` / task runner.
- ❌ No `.github/` — no CI, no PR/issue templates, no CODEOWNERS, no dependabot.
- ❌ No `.env` / `.env.example` — runtime config still lives as hardcoded module constants in `ui_dataclean.py` and `ui_pyqt5.py` headers.
- ❌ No pre-commit config (ruff is configured but only runs when invoked manually; a `pre-commit` hook isn't wired up).

**Summary:** the dependency / tool / test / lint config is now first-class in `pyproject.toml`; the project has a real pytest test suite under `tests/`; `.vscode` is finally gitignored and untracked. Still **no CI**, no `[project.scripts]`, no build backend.

---

## 5. Remaining structural smells

The 20 smells from the original audit have been re-evaluated against the current state. Numbers in the **was** column map to the old §5 smell index.

| Was | Severity | Smell | Status |
|:---:|:---:|---|---|
| 1 | 🔴 | `.venv/` (1.79 GB) physically inside the repo. | ⚠ Unchanged — still at repo root, still pollutes recursive search. |
| 2 | 🔴 | No package structure. | ✅ **FIXED** — `models/` and `src/` are real packages with `__init__.py`. |
| 3 | 🟠 | Mixed casing in module names. | ✅ **FIXED** — PascalCase modules moved/renamed; `drop_out.py`, `kl.py`, `performance_widget.py`, `classification.py`, `uncertainty_analysis.py`, `multi_processing_analysis.py`, `progress_bar.py` are all snake_case. |
| 4 | 🟡 | Dead `main.py` (4-line `uv init` stub). | ⚠ Unchanged. |
| 5 | 🟠 | Likely-dead `DatasetCreation.py` and `keras_kl.py`. | ✅ **FIXED** — `keras_kl.py` deleted; `dataset_creation.py` moved to `archive/`. |
| 6 | 🔴 | `test/test.py` 211 KB data dump. | ✅ **FIXED** — `test/` directory is now empty; the dump was cleaned out. (Large sample JSON tile dictionaries were moved to `archive/`.) |
| 7 | 🟠 | All `test/` Python files were scratch experiments. | ✅ **FIXED** — scratch scripts removed. |
| 8 | 🔴 | Hardcoded absolute paths everywhere. | 🟠 **PARTIAL** — paths inside the active codebase (`src/`, `models/`) are now parameter-passed; remaining hardcoded paths are in `archive/` (correctly quarantined) and `docker-compose.yaml` WSL mount (still non-portable). |
| 9 | 🟠 | Duplicated `WorkerSignals`/`Worker`/`LongRunningWorker` between the two GUIs. | ⚠ Unchanged. |
| 10 | 🟠 | Duplicated config defaults between GUIs. | 🟠 **PARTIAL** — `CLASS_NAMES`/`N_CLASSES`/`INPUT_SHAPE` now live in `src/config.py` (single source). GUI-specific constants (`APP_ICON`, `DEFAULT_MODEL`, `DEFAULT_TILE_SIZE`, `DEEPZOOM_URL`, `KNOWN_CLASSES`) are still per-GUI. |
| 11 | 🟡 | Duplicated `styles/stile.txt` and `styles/stileor.css`. | ⚠ Unchanged. |
| 12 | 🟡 | Duplicate `createTable` / `createTable_sigle` in `performance_widget.py`. | ⚠ Unchanged (file renamed, body not refactored). |
| 13 | 🔴 | Documented bugs in `BUG_REPORT.md`. | ✅ **FIXED** — see §6 below; all documented bugs are now closed. |
| 14 | 🟠 | `Th.create_list` line 35 `out_al = np.median(list_epi)` copy-paste bug. | ✅ **FIXED** — line 35 now reads `out_al = np.median(list_ale)`. |
| 15 | 🟡 | Duplicated RGBA color palettes in `Classification.overlay`. | ⚠ Unchanged. |
| 16 | 🟡 | Deprecated `model.fit_generator` / `tf.compat.v1.initializers.random_normal`. | 🟡 **PARTIAL** — `model.fit_generator` replaced with `model.fit` in both `models/kl.py` and `models/drop_out.py`; `tf.compat.v1.initializers.random_normal` is still used in `kl.py` (required by `tfp.layers.default_mean_field_normal_fn` — would need a custom `tf.keras.initializers.RandomNormal` shim to remove). |
| 17 | 🟠 | Config scattered vs centralised for `CLASS_NAMES`. | ✅ **FIXED** — `CLASS_NAMES`/`N_CLASSES`/`INPUT_SHAPE` now live in `src/config.py`; old `['AC','H','AD']` literals replaced with `list(CLASS_NAMES)` everywhere in the active code. |
| 18 | 🟠 | Big monolithic files (`ui_dataclean.py` 790+, `ui_pyqt5.py` 766+). | ⚠ Unchanged. |
| 19 | 🟡 | `archive/` mislabeled. | 🟠 **PARTIAL** — now holds the dead TF1 Colab code (`dataset_creation.py`) and 14 MB of sample JSON tile dictionaries, so the name is finally accurate. |
| 20 | 🔴 | `.gitattributes` configured `[lfs]` but defines no LFS filter patterns, leaving multi-MB `.txt` blobs as plain-text. | ⚠ Unchanged — but the worst offenders have been moved to `archive/`, softening the impact. |

### New smells introduced / surfaced

| Severity | Smell |
|:---:|---|
| 🟡 | `models/base.py` Protocol is declared but not enforced anywhere — `BayesianModel` is imported on `models.base` and re-exported, but `isinstance(model_obj, BayesianModel)` is never checked. It's documentation, not a constraint. |
| 🟡 | Both `models/*.py` files still use `tf.compat.v1.initializers.random_normal` for the Flipout posterior init — needed for the TFP API used, but tied to TF 2.x (TF 2.16+ may emit warnings). |

---

## 6. Bug status

All bugs originally documented in `BUG_REPORT.md` were closed before v8.0.55. Since the last project-map refresh three **new** platform-compatibility bugs were found and fixed during the v8.2.0 / v8.5.0 Linux/KDE work, and one pre-existing Linux crash was already fixed by `tests/analysis/`.

| Bug | Was | Now | Detail |
|---|---|---|---|
| `DropOut.py` constructor param-name mismatch vs `ui_dataclean.py` caller | 🐛 Open | ✅ Fixed | `models/drop_out.py` now uses `(model_save_path, epochs, path_train, path_val, batch_size, augment)`. Caller in `ui_dataclean.py:736-741` was updated to the unified signature. |
| `DropOut.py` `ModelCheckpoint` saves to hardcoded relative `weights_best.h5` | 🐛 Open | ✅ Fixed | `models/drop_out.py` now derives `checkpoint_path = self.model_save_path.replace(".h5", "_best.h5")`. Same fix in `models/kl.py`. |
| `CLASS_NAMES` order mismatch (`["AC","H","AD"]` vs alphabetical) | 🐛 Latent | ✅ Fixed | Canonical order `('AC','AD','H')` lives in `src/config.py` and is imported everywhere. No `['AC','H','AD']` literals remain in the active codebase. (Pre-existing `.h5` checkpoints encode the old order — re-train before trusting indices.) Covered by `tests/unit/test_config.py`. |
| `DropOut.py` docstring still references `model_training.py` | 🐛 Open | ✅ Fixed | `models/drop_out.py` docstring now reads `python -m models.drop_out`. |
| Bug-1 "ImageDataGenerator not imported" | ✓ Fixed | ✅ | Import present at `models/drop_out.py:26`. |
| `tfp` import commented out in `Kl.py` (was a `NameError` waiting to fire) | 🐛 Open | ✅ Fixed | `models/kl.py:6` imports `tensorflow_probability as tfp`. |
| `MyCallback` in `Kl.py` keyed on `'acc'`/`'val_acc'` (Keras-1 names → `KeyError` in TF 2.x) | 🐛 Open | ✅ Fixed | Replaced with `TrainingProgressCallback` using `.get("accuracy", …)` / `.get("val_accuracy", …)`. |
| `Kl.py` `model.fit_generator` deprecated | 🐛 Open | ✅ Fixed | `models/kl.py` now uses `model.fit`. |
| `Kl.py` `__main__` block broken (no ctor args, no return) | 🐛 Open | ✅ Fixed | `models/kl.py` `__main__` constructs with real kwargs and `start_train` returns `History`. |
| `uncertainty_analysis.py` line-35 `out_al = np.median(list_epi)` copy-paste | 🐛 Open | ✅ Fixed | `src/uncertainty_analysis.py:35` now reads `out_al = np.median(list_ale)`. The aleatoric outlier-replacement value is now correctly computed from `list_ale` (aleatoric) instead of `list_epi` (epistemic). |
| DeepZoom launcher used Windows-only `cmd /k` → `sh: cmd: comando non trovato` on Linux | 🐛 New (v8.2.0) | ✅ Fixed | `ui_pyqt5.py:_launch_deepzoom` now builds `python <script> <svs>` directly on Linux (path-quoted for spaces) and only prepends `cmd /k` under Windows. |
| App menu bar invisible on Linux (GNOME/Unity/KDE/Wayland) | 🐛 New (v8.2.0) | ✅ Fixed | `ui_pyqt5.py` `__main__` now sets `AA_DontUseNativeMenuBar` and `viewer.menuBar().setNativeMenuBar(False)` on Linux, so the File/Analysis/View/Options/Help bar renders inside the window. |
| Wrong Wayland window icon (shown the default Wayland placeholder instead of `target.ico`) | 🐛 New (v8.5.0) | ✅ Fixed | `.ico` files don't load under Qt/Wayland; converted `target.ico` → `target.png` with ImageMagick and `APP_ICON` now picks `icons/target.png` on Linux (`sys.platform.startswith("linux")`). |
| Generic Wayland mouse cursor instead of the KDE/Plasma theme cursor | 🐛 New (v8.5.0) | ✅ Fixed | `ui_pyqt5.py` `__main__` exports `QT_QPA_PLATFORM=xcb` so Qt runs under XWayland, which honours the user's cursor theme. |
| `StartAnalysis.get_thumb` raised `IndexError: tuple index out of range` when `lev_sec` was out of range | 🐛 Latent | ✅ Fixed | Now clamps `lev_sec` against `level_count`. Regression captured in `tests/analysis/test_start_analysis.py::test_openSvs_clamps_out_of_range_lev_sec_instead_of_IndexError` and `tests/unit/test_tile_partition.py`. |
| `tileGen(state=0)` could hit `Invalid address` ValueError | 🐛 Latent | ✅ Fixed | Regression captured in `tests/analysis/test_start_analysis.py`. |

---

## 7. Git state and governance

| Aspect | State |
|---|---|
| Current branch | `master` (single local branch). |
| Remote | `origin = https://github.com/Pier45/WSI_analysis.git`. |
| `.gitignore` | Present, fairly complete. `.vscode/` is now correctly ignored and untracked (since v8.5.0). Redundantly lists `.python-version` (line 88) even though `.python-version` IS tracked. `.opencode/` is not yet ignored. |
| `.gitattributes` | LF normalisation + suppresses JS/TS from linguist stats. `[lfs]` section in `.git/config` indicates LFS was initialised but `.gitattributes` defines **no LFS filter patterns** — multi-MB `.txt` data blobs in `archive/` are committed as normal blobs, not via LFS. |
| `LICENSE` | MIT, Copyright (c) 2026 Piero Policastro — present and tracked. |
| `AGENTS.md` | present — graphify-skill invocation rules. |
| `.github/` | ❌ No directory — no CI workflows, no issue/PR templates, no CODEOWNERS, no dependabot.yml. |
| Pre-commit / signed commits | ❌ None. Git hooks are all `.sample` defaults. |
| CI | ❌ None — but the `tests/` suite is CI-ready: `pytest` is configured with strict markers/strict config and markers gate heavy deps (`openslide`, `tf`, `gui`) so a default CI run is just the fast `unit/*` tests. |
| Tests | ✅ Real `tests/` suite now exists (see §4). Fast-Lane unit + pytest-qt offscreen GUI tests. |

---

## Quick-reference paths

```
WSI_analysis/
├── ui_pyqt5.py                    # entry point 1 ("Bayesian Analyzer" GUI) — cross-platform DeepZoom launch + Linux/KDE fixes
├── ui_dataclean.py                # entry point 2 (5-tab GUI — package-qualified imports)
├── main.py                        # ⚠ dead 4-line uv init stub
│
├── models/                        # Bayesian model package
│   ├── __init__.py                # re-exports BayesianDropoutCNN, ModelKl, BayesianModel
│   ├── base.py                    # BayesianModel Protocol (unified ctor + train() + start_train())
│   ├── drop_out.py                # MC-Dropout CNN (BayesianDropoutCNN) — unified signature
│   └── kl.py                      # KL-Flipout CNN (ModelKl) — tfp imported; fit_generator→fit; __main__ fixed
│
├── src/                           # core logic package
│   ├── __init__.py                # re-exports CLASS_NAMES, N_CLASSES, INPUT_SHAPE
│   ├── config.py                  # single source of truth for class order / shape
│   ├── classification.py          # Classification — uses CLASS_NAMES from config
│   ├── multi_processing_analysis.py  # StartAnalysis OpenSlide tiling (lev_sec clamped, IndexedError fixed)
│   ├── uncertainty_analysis.py    # Th thresholding class (line-35 bug fixed)
│   ├── performance_widget.py      # PerformanceTab confusion-matrix widget (renamed from test_widget.py)
│   ├── progress_bar.py            # Actions Qt helper (14 lines)
│   ├── qt_workers.py              # shared WorkerSignals/Worker/WorkerLong (NEW — extracted from the two GUIs)
│   └── deepzoom/deepzoom_server.py  # clean Flask sub-app
│
├── tests/                         # ✓ NEW real pytest suite (v8.0.55+)
│   ├── conftest.py                #   root fixtures + marker gating
│   ├── analysis/test_start_analysis.py   # -m openslide — OpenSlide regressions
│   ├── gui/test_actions_factory.py       # -m gui — pytest-qt, action contract tests
│   └── unit/                            # default — runs in CI
│       ├── test_config.py               # CLASS_NAMES / N_CLASSES / INPUT_SHAPE guards
│       └── test_tile_partition.py        # manage_process shape + value regression
│
├── icons/                         # 14 PyQt5 GUI icons (.ico + Linux .png variants — target.png for Wayland)
├── archive/                       # legacy + sample data (out of the live tree)
│   ├── dataset_creation.py         # TF1 Colab-era 5-class pipeline
│   ├── dictionary_5_js.txt         # ~1 MB sample tile dict (MC=5)
│   ├── new_train_js.txt            # 9.5 MB sample tile dict (train)
│   ├── new_val_js.txt              # 4.0 MB sample tile dict (val)
│   └── original_requirements.txt   # 2019 TF 1.15 requirements
│
├── data/                          # real patient WSI datasets
├── docs/                          # PROJECT_MAP.md, BUG_REPORT.md, sasa.txt
├── img/                           # 6 thesis figures + 3 stray UI assets
├── styles/                        # stile.txt + stileor.css (near-duplicate)
├── graphify-out/                  # graphify outputs (refreshed 2026-07-17: 809 nodes / 1230 edges / 82 communities)
│
├── pyproject.toml                 # full toolchain: deps + [tool.uv] + [dependency-groups].dev + [tool.pytest] + [tool.ruff]
├── uv.lock                        # lockfile (190 KB, win32 + linux resolutions)
├── Dockerfile                     # multi-stage, only launches ui_dataclean.py
├── docker-compose.yaml            # 2 services, hardcoded personal WSL path
├── .gitignore                     # now correctly ignores .vscode/
└── .gitattributes                 # has [lfs] in .git/config but no LFS filter patterns
```
