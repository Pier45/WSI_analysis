# Graph Report - WSI_analysis  (2026-08-11)

## Corpus Check
- 57 files · ~1,353,310 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 940 nodes · 1500 edges · 82 communities (57 shown, 25 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 57 edges (avg confidence: 0.66)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a8ebab39`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- OpenSeadragon Viewer
- PyQt5 Image Viewer
- Multi-Process Analysis
- jQuery Library
- DataClean Tab Widget
- Archive & Config
- Bayesian Dropout CNN
- OpenSeadragon Mouse Events
- Performance Widget
- KL Bayesian VGG Model
- OpenSeadragon Controls Autohide
- OpenSeadragon Tile Drawing
- Qt Workers
- DataClean UI Builders
- OpenSeadragon Touch Events
- Classification Module
- DataClean Long-Running Workers
- jQuery Ajax & Animation
- jQuery Selector Engine
- Base Bayesian Model
- TFRecord Dataset Creation
- DeepZoom Server
- DataClean Matplotlib Canvas
- Config & Init
- Uncertainty Analysis TH
- OpenSeadragon Scalebar
- jQuery DOM Manipulation
- OpenSeadragon Pointer Events
- DataClean Main Window
- DataClean Uncertainty Runner
- Icons Assets
- DataClean Threshold Mode
- Figure Images
- OpenSeadragon Fade
- OpenSeadragon Zoom
- jQuery Size Helpers
- OpenSeadragon Pointer Tracking
- OpenSeadragon Wheel Events
- OpenSeadragon Strip Scroll
- OpenSeadragon DZI Parser
- Project Map Doc
- Models Package Init
- OpenSeadragon Config
- OpenSeadragon XML10 Parser
- OpenSeadragon Stop Fade
- Graphify Skill
- Bug Report Doc
- WSI Analysis Package
- ._start_analysis
- start_tile_threads
- ._select_dataset_folder
- Sample Tile Dictionary (Train)
- ._tile_dataset
- defaultDisplay
- AGENTS.md
- Original TF 1.15 Requirements (2019)
- DeepZoom Web Viewer (Flask)
- ._create_clean_dataset
- ._update_training_log
- CLASS_NAMES Order Bug (AC/H/AD vs alphabetical)
- DropOut.py Constructor Signature Mismatch Bug
- Module Interaction Diagram (PROJECT_MAP)
- Structural Smells Inventory
- Aleatoric Uncertainty
- Class AC (Adenocarcinoma)
- Class AD (Adenoma)
- Class H (Healthy)
- Colorectal WSI Dataset (Leeds, 27 patients)
- Downscale to 64x64 (Equivalent accuracy, lower cost)
- Epistemic Uncertainty
- KL-Divergence Variational Inference (Flipout)
- Patient-wise Data Splits
- Tile Extraction (256x256, level 0, no overlap)
- Prediction Uncertainty Quantification
- WorkerSignals
- test_actions_factory.py
- ui_dataclean.py
- MainWindow
- UncertaintyTab
- about_dialogs.py
- actions.py

## God Nodes (most connected - your core abstractions)
1. `ImageViewer` - 36 edges
2. `StartAnalysis` - 30 edges
3. `AnalyzerState` - 23 edges
4. `CleaningTab` - 21 edges
5. `Classification` - 20 edges
6. `TrainingTab` - 18 edges
7. `PerformanceTab` - 18 edges
8. `make_action()` - 17 edges
9. `GetTilesTab` - 17 edges
10. `DataCleanState` - 16 edges

## Surprising Connections (you probably didn't know these)
- `ImageViewer` --uses--> `StartAnalysis`  [INFERRED]
  gui/analyzer/main_window.py → src/multi_processing_analysis.py
- `ImageViewer` --uses--> `Actions`  [INFERRED]
  gui/analyzer/main_window.py → src/progress_bar.py
- `_import_viewer()` --indirect_call--> `ImageViewer`  [INFERRED]
  tests/gui/test_actions_factory.py → gui/analyzer/main_window.py
- `DataCleanState` --uses--> `Th`  [INFERRED]
  gui/dataclean/state.py → src/uncertainty_analysis.py
- `CleaningTab` --uses--> `Th`  [INFERRED]
  gui/dataclean/tabs/tab_cleaning.py → src/uncertainty_analysis.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Two Bayesian Uncertainty Approaches (MC-Dropout + KL-Flipout)** — readme_mc_dropout, readme_kl_divergence_vi [EXTRACTED 1.00]
- **Three-Class Tile Set (AC/AD/H) - docs** — readme_class_ac, readme_class_ad, readme_class_h [EXTRACTED 1.00]
- **Three GUI/Server Entry Points** — readme_bayesian_analyzer_gui, readme_dataclean_gui, readme_deepzoom_viewer [EXTRACTED 1.00]
- **Thesis methodology-to-application pipeline figures (dataset -> uncertainty concept -> GUI -> output maps)** —  [INFERRED 0.75]
- **Uncertainty quantification conceptualisation and its concrete output** —  [INFERRED 0.85]
- **Three-Class Tile Set (AC/AD/H)** — icons_ac, icons_ad, icons_h [EXTRACTED 1.00]

## Communities (82 total, 25 thin omitted)

### Community 0 - "OpenSeadragon Viewer"
Cohesion: 0.03
Nodes (30): beginZoomingIn(), beginZoomingOut(), clearTrackedPointers(), configureFromObject(), configureFromXML(), configureFromXml10(), doZoom(), inTo() (+22 more)

### Community 1 - "PyQt5 Image Viewer"
Cohesion: 0.11
Nodes (8): Bayesian Analyzer application — formerly ``ui_pyqt5.py``.  Public API ----------, ImageViewer, QMainWindow, Main application window for the Bayesian Analyzer., Display a critical dialog when a background worker raises an exception., show_worker_error(), main(), Bayesian Analyzer — thin launcher.  The real application lives under :mod:`gui.a

### Community 2 - "Multi-Process Analysis"
Cohesion: 0.13
Nodes (12): Start the Bayesian classification in a background thread.  Contains the first-ru, User-tunable constants for the Bayesian Analyzer GUI.  These were originally mod, DeepZoom viewer launcher + info / about dialogs for the Bayesian Analyzer.  The, Mutable shared state for the Bayesian Analyzer helpers.  Mirrors :class:`gui.dat, Tile-creation background workers for the Bayesian Analyzer.  Historical design:, QRunnable, Actions, Shared Qt thread-pool plumbing for the two GUI entry points.  Both ``ui_pyqt5.py (+4 more)

### Community 3 - "jQuery Library"
Cohesion: 0.05
Nodes (4): NOTE: This can be skipped if there are no unmatched elements (i.e., `matchedCoun, TODO: Now that all calls to _data and _removeData have been replaced, TODO: identify versions, TODO: identify versions

### Community 4 - "DataClean Tab Widget"
Cohesion: 0.13
Nodes (7): CleaningTab, QWidget, Load uncertainty values from the JSON, refresh the histograms., After the histograms are loaded, enable the auto/manual radios., Compute Otsu + new thresholds and draw vertical lines on the total histogram., Copy the surviving tiles into the chosen folder under per-class subdirs., Tab 4 — histograms, threshold selection, clean-dataset export.

### Community 5 - "Archive & Config"
Cohesion: 0.50
Nodes (5): Hardcoded WSL Path in Compose, Docker Compose (wsi-clean, wsi-analysis services), Bayesian Analyzer GUI (ui_pyqt5), Dataclean Tool GUI (ui_dataclean, 5 tabs), Qt Stylesheet (stile.txt)

### Community 6 - "Bayesian Dropout CNN"
Cohesion: 0.10
Nodes (16): BayesianDropoutCNN, ConvBlockConfig, Callback, History, ImageDataGenerator, Model, Bayesian dropout CNN for histological-tile classification (AC / AD / H).  Monte, Convolutional neural network with Monte Carlo Dropout for Bayesian     uncertain (+8 more)

### Community 7 - "OpenSeadragon Mouse Events"
Cohesion: 0.11
Nodes (30): capturePointer(), getMouseAbsolute(), getMouseRelative(), getPointRelativeToAbsolute(), handleMouseEnter(), handleMouseExit(), handleMouseMove(), handlePointerStop() (+22 more)

### Community 8 - "Performance Widget"
Cohesion: 0.16
Nodes (9): ndarray, QTableWidget, PerformanceTab, QFrame, QWidget, QHLine, Populate the single-patient confusion-matrix table., Populate the aggregate confusion-matrix table. (+1 more)

### Community 9 - "KL Bayesian VGG Model"
Cohesion: 0.07
Nodes (22): BayesianModel, History, Shared interface for the Bayesian uncertainty models.  Both ``BayesianDropoutCNN, Construct a model for training and uncertainty estimation.      Parameters (cons, Build, compile and train the model, returning the Keras history., Backward-compatible alias for :meth:`train`., Bayesian uncertainty models package.  Re-exports the two model classes for conve, ModelKl (+14 more)

### Community 10 - "OpenSeadragon Controls Autohide"
Cohesion: 0.12
Nodes (19): abortControlsAutoHide(), beginControlsAutoHide(), drawersNeedUpdate(), drawOverlays(), getOverlayObject(), _getSafeElemSize(), loadOverlays(), onBlur() (+11 more)

### Community 11 - "OpenSeadragon Tile Drawing"
Cohesion: 0.14
Nodes (18): blendTile(), compareTiles(), drawDebugInfo(), drawTiles(), getTile(), isCovered(), loadTile(), offsetForRotation() (+10 more)

### Community 12 - "Qt Workers"
Cohesion: 0.29
Nodes (6): Any, QObject, _accepts_kwarg(), Signals emitted by :class:`WorkerLong` during its lifecycle.      ``intermediate, True iff *fn* accepts a keyword argument named *name*.      Returns True when *f, WorkerSignals

### Community 13 - "DataClean UI Builders"
Cohesion: 0.18
Nodes (6): GetTilesTab, QWidget, Open a folder dialog and populate the QListWidgets with files in         each pe, Launch per-dataset tiling workers (one worker per class sub-folder)., Tab 1 — pick source folders and launch per-dataset tiling workers., QHBoxLayout

### Community 14 - "OpenSeadragon Touch Events"
Cohesion: 0.15
Nodes (17): abortTouchContacts(), getCaptureEventParams(), getCenterPoint(), getStandardizedButton(), handleMouseUp(), handleTouchEnd(), onMouseUp(), onMouseUpCaptured() (+9 more)

### Community 15 - "Classification Module"
Cohesion: 0.09
Nodes (17): Classification, GREAT NOTE: LIST ARE index-1 for the 0 index, Analyze the selected folder, finding all the png files, Create the dict with the key the number of the tile, to each key correspond anot, This method read the image, modify it as numpy array and at the end control if s, Load the model and analyze the tile, the dictionary is updated with the predicte, _grid_dictionary(), _make_classification_with_thumbnail() (+9 more)

### Community 16 - "DataClean Long-Running Workers"
Cohesion: 0.16
Nodes (13): HorizontalLine, Reusable Qt widgets used across the Datacleaning tabs., Decorative horizontal separator., User-tunable defaults and paths for the Datacleaning application.  These were or, DataCleanState, Shared mutable state for the Datacleaning tabs.  The original ``MainTabWidget``, Mutable shared state across the five Datacleaning tabs.      Attributes are inte, Datacleaning application tabs.  Each tab is a self-contained ``QWidget`` that ta (+5 more)

### Community 17 - "jQuery Ajax & Animation"
Cohesion: 0.25
Nodes (9): ajaxConvert(), ajaxHandleResponses(), Animation(), createFxNow(), createTween(), defaultPrefilter(), done(), propFilter() (+1 more)

### Community 18 - "jQuery Selector Engine"
Cohesion: 0.20
Nodes (12): addCombinator(), condense(), createPositionalPseudo(), elementMatcher(), markFunction(), matcherFromGroupMatchers(), matcherFromTokens(), multipleContexts() (+4 more)

### Community 19 - "Base Bayesian Model"
Cohesion: 0.25
Nodes (4): Divide the wsi in tiles, thanks to get_tile, if the test with fold managere is f, Create the folders where the thumbnail and the tiles of the image are         st, Create the thumbnail of the image, ready for the classification phase., StartAnalysis

### Community 20 - "TFRecord Dataset Creation"
Cohesion: 0.33
Nodes (3): _bytes_feature(), CreationTFRecord, _int64_feature()

### Community 21 - "DeepZoom Server"
Cohesion: 0.27
Nodes (7): BytesIO, index(), load_slide(), PILBytesIO, Classic PIL doesn't understand io.UnsupportedOperation., slugify(), tile()

### Community 22 - "DataClean Matplotlib Canvas"
Cohesion: 0.22
Nodes (6): FigureCanvasQTAgg, MatplotlibCanvas, QFrame, Decorative vertical separator., Matplotlib canvas for rendering inline histograms., VerticalLine

### Community 23 - "Config & Init"
Cohesion: 0.20
Nodes (7): analysis(), _expected_partition(), Unit tests for ``StartAnalysis.manage_process()``.  This method computes the x-a, A StartAnalysis with the minimum attributes manage_process touches.      We don', Reference implementation of manage_process() used as the oracle.      Kept indep, Return-shape contract: always a 5-tuple of equal-length lists., TestManageProcessShape

### Community 25 - "OpenSeadragon Scalebar"
Cohesion: 0.44
Nodes (7): getScalebarSizeAndText(), getScalebarSizeAndTextForMetric(), getSignificand(), getWithUnit(), log10(), normalize(), roundSignificand()

### Community 26 - "jQuery DOM Manipulation"
Cohesion: 0.38
Nodes (7): buildFragment(), disableScript(), domManip(), getAll(), remove(), restoreScript(), setGlobalEval()

### Community 27 - "OpenSeadragon Pointer Events"
Cohesion: 0.12
Nodes (18): _build_partition_args(), _get_cached_generator(), Build the per-partition argument tuples for ``_tile_partition_worker``.      Pur, Per-process cached ``DeepZoomGenerator`` for *file_path*.      Each worker proce, Module-level worker for ``ProcessPoolExecutor`` — extracts one x-range     of ti, _tile_partition_worker(), _fields(), _kwargs() (+10 more)

### Community 28 - "DataClean Main Window"
Cohesion: 0.20
Nodes (9): Analysis-tier tests for ``StartAnalysis`` that exercise the OpenSlide / DeepZoom, Regression for the IndexError: tuple index out of range.      Original bug: ui_p, When ``lev_sec`` is in range, the produced thumbnail must come from     ``list_l, Regression for the ``Invalid address`` ValueError.      ``tile_gen(state=0)`` re, Regression: a small fixture (``level_count=2``) used to crash hard in     ``tile, test_get_thumb_uses_a_valid_openslide_level(), test_lev_sec_under_level_count_does_not_raise_in_get_thumb(), test_openSvs_clamps_out_of_range_lev_sec_instead_of_IndexError() (+1 more)

### Community 29 - "DataClean Uncertainty Runner"
Cohesion: 0.22
Nodes (18): _adjust_scroll_bar(), enable_view_actions(), fit_to_window(), normal_size(), print_image(), QMainWindow, Image display, printing, and zoom helpers for the Bayesian Analyzer.  These were, Toggle the eight View-menu overlay actions on/off. (+10 more)

### Community 30 - "Icons Assets"
Cohesion: 0.70
Nodes (5): AC Class Icon (Adenocarcinoma), AD Class Icon (Adenoma), Folder Icon, H Class Icon (Healthy), Target App Icon

### Community 32 - "Figure Images"
Cohesion: 0.67
Nodes (4): Figure 4.2 - Tile extraction & dataset composition diagram, Figure 4.3 - Bayesian uncertainty quantification concept (MC Dropout aleatoric/epistemic), Figure 5.1 - WSI Analysis PyQt5 GUI screenshot, Figures 5.2 & 5.3 - WSI Analysis result overlays / uncertainty maps output

### Community 33 - "OpenSeadragon Fade"
Cohesion: 0.50
Nodes (4): beginFading(), outTo(), scheduleFade(), updateFade()

### Community 34 - "OpenSeadragon Zoom"
Cohesion: 0.07
Nodes (29): 1. WSI Analysis GUI, 2. Data Cleaning & Training GUI, 3. DeepZoom Server, Apple Silicon / ARM hosts, Bare `docker run` (fallback, no compose), Build (one image, two entry points), Data Cleaning via Uncertainty, Dataset (+21 more)

### Community 35 - "jQuery Size Helpers"
Cohesion: 0.67
Nodes (3): augmentWidthOrHeight(), curCSS(), getWidthOrHeight()

### Community 36 - "OpenSeadragon Pointer Tracking"
Cohesion: 0.08
Nodes (24): 1. Top-level directory layout, 2. What the project does, 3. Package layout and modules, 4. Build / dependency / test / config files, 5. Remaining structural smells, 6. Bug status, 7. Git state and governance, Domain (+16 more)

### Community 37 - "OpenSeadragon Wheel Events"
Cohesion: 0.17
Nodes (7): ``numx_stop[-1]`` must never exceed ``numtotx`` — otherwise the         ``range(, ``_folder_exists`` + ``Classification.select_folder`` parse the         ``p_<xs>, ``manage_process`` uses 1-based tile indices (``start_idx`` starts         at 1,, Adjacent partitions must touch: stop[i] == start[i+1]. Otherwise a         colum, Cross-check the production method against the reference oracle., The branch ``n_core >= numtotx`` collapses to 1 partition so we         don't ov, TestManageProcessValues

### Community 38 - "OpenSeadragon Strip Scroll"
Cohesion: 0.17
Nodes (11): Bug 1 — `ImageDataGenerator` not imported, Bug 2 — Parameter name mismatch between caller and constructor, Bug 3 — `CLASS_NAMES` order may not match downstream code, Bug 4 — `history_save_path` assumes `.h5` extension, Bug 5 — `ModelCheckpoint` saves to a hardcoded relative path, Bug 6 — Batch-level accuracy key may be version-dependent, Bug Report — DropOut.py, 🔴 Critical Bugs (+3 more)

### Community 39 - "OpenSeadragon DZI Parser"
Cohesion: 0.33
Nodes (4): display_image(), Render *image* in the central label, scaling to fit the screen if needed., Open the SVS file, generate the thumbnail, cache tile metadata., QImage

### Community 41 - "Models Package Init"
Cohesion: 0.15
Nodes (5): QWidget, Instantiate the chosen model and run it inside a WorkerLong., Append epoch rows to the training log label.          The Keras ``TrainingProgre, Tab 2 — pick dropout vs KL, set epochs / batch size, run training., TrainingTab

### Community 42 - "OpenSeadragon Config"
Cohesion: 0.19
Nodes (7): Datacleaning application — formerly ``ui_dataclean.py``.  Public API ----------, MainWindow, QMainWindow, Main application window for Bayesian Datacleaning., QAction, main(), Bayesian Datacleaning Application — thin launcher.  The real application lives u

### Community 43 - "OpenSeadragon XML10 Parser"
Cohesion: 0.33
Nodes (3): Manage the starting and ending point for the reading phase of the SVS file., Start the tile-extraction workers and block until they finish.          Historic, Call this function to divide the slice in tiles, it manage the dimension and the

### Community 44 - "OpenSeadragon Stop Fade"
Cohesion: 0.22
Nodes (8): on_analysis_complete(), QMainWindow, QThreadPool, Run the Bayesian classification in a background thread (or load existing results, Called when the analysis thread finishes successfully., Prompt the user to pick a ``.h5`` model file and cache the path.      Returns th, select_model(), start_analysis()

### Community 49 - "._start_analysis"
Cohesion: 0.24
Nodes (9): _get_screen_size(), Return the primary screen dimensions as ``(width, height)``.      Falls back to, create_menus(), populate_toolbar(), QMainWindow, Menu bar + toolbar construction for the Bayesian Analyzer.  Pure UI wiring — no, Build the five top-level menus and attach them to ``parent.menuBar()``., Fill the main toolbar with the action/separator sequence used by the Analyzer. (+1 more)

### Community 50 - "start_tile_threads"
Cohesion: 0.17
Nodes (11): _analysis_tile_worker(), create_tiles(), folder_exists(), on_tile_worker_finished(), QThreadPool, Launch one :class:`WorkerLong` that drives a :class:`ProcessPoolExecutor`     ov, Slot connected to the host tile worker's ``finished`` signal.      Runs on the m, Module-level (picklable) tile-extraction worker for the analysis path.      Mirr (+3 more)

### Community 51 - "._select_dataset_folder"
Cohesion: 0.10
Nodes (11): Project-wide configuration constants.  Single source of truth for class names, i, Source package — core logic (classification, uncertainty, widgets, config)., Unit tests for ``src.config`` — pure Python, no optional deps.  These run in any, Lock the canonical class/shape contract — changing these is a breaking     chang, Canonical order is alphabetical, defining argmax index., Tuples are immutable and hashable — guards against accidental         in-place m, ``N_CLASSES`` must stay in sync with ``len(CLASS_NAMES)``., The Bayesian CNNs are trained on 64x64 RGB tiles — any other shape         is an (+3 more)

### Community 52 - "Sample Tile Dictionary (Train)"
Cohesion: 0.83
Nodes (4): Sample Tile Dictionary (MC=5), Sample Tile Dictionary (Train), Sample Tile Dictionary (Validation), Monte Carlo Dropout (Gal & Ghahramani 2015)

### Community 53 - "._tile_dataset"
Cohesion: 0.09
Nodes (18): create_actions(), make_action(), QMainWindow, Factory used by every menu/toolbar action in the Analyzer.      Wrapping ``QActi, Populate every ``_*_act`` attribute on *parent*.      Called from ``ImageViewer., _import_viewer(), GUI tests for ``ui_pyqt5.ImageViewer._make_action`` (the Qt action factory intro, We need a real on-disk image for QIcon to load; PyQt5 silently         produces (+10 more)

### Community 54 - "defaultDisplay"
Cohesion: 0.67
Nodes (3): actualDisplay(), defaultDisplay(), showHide()

### Community 58 - "._create_clean_dataset"
Cohesion: 0.15
Nodes (8): qapp(), _QtBot, GUI-tier-specific pytest configuration.  Provides a minimal ``qapp`` and ``qtbot, Session-scoped QApplication fallback when pytest-qt is absent.      pytest-qt别名名, Context manager returned by our qtbot.waitSignal fallback., Minimal subset of pytest-qt's qtbot: just ``waitSignal``., Fallback qtbot providing ``waitSignal`` only — enough for our tests., _WaitSignalCtx

### Community 75 - "WorkerSignals"
Cohesion: 0.22
Nodes (9): Config, Path, fixtures_dir(), pytest_configure(), Top-level pytest configuration shared by every test subfolder.  This file: * Boo, Register markers (declared in pyproject.toml) and add custom ini-style     optio, Absolute path to ``tests/fixtures`` — create it on demand so a missing     folde, Return the path to a small ``.svs`` test fixture, or ``None``.      Tests that n (+1 more)

### Community 76 - "test_actions_factory.py"
Cohesion: 0.22
Nodes (7): about_deep_zoom(), open_browser(), open_deep_zoom(), QMainWindow, QThreadPool, Start the DeepZoom Flask server in a separate process and open the     browser o, Open the DeepZoom URL in the default browser after a short delay.

### Community 77 - "ui_dataclean.py"
Cohesion: 0.38
Nodes (7): getPointerType(), handlePointerMove(), handlePointerUp(), onPointerMove(), onPointerMoveCaptured(), onPointerUp(), onPointerUpCaptured()

### Community 78 - "MainWindow"
Cohesion: 0.11
Nodes (10): MainTabWidget, QWidget, MainTabWidget — coordinator of the 5 Datacleaning tabs.  Holds the shared :class, Central widget that coordinates all application tabs.      Manages shared state, Connect each tab's worker_started signal to the shared handlers., Connect a worker to the shared handlers and start it.          For a training wo, Main window of the Datacleaning application.  Owns the menu bar (File / About) a, Tab 5 — wraps :class:`PerformanceTab` with shared-state awareness.      ``Perfor (+2 more)

### Community 79 - "UncertaintyTab"
Cohesion: 0.27
Nodes (4): QWidget, Spin up a :class:`WorkerLong` running MC classification., Tab 3 — run Monte-Carlo classification on a chosen dataset., UncertaintyTab

### Community 80 - "about_dialogs.py"
Cohesion: 0.40
Nodes (3): about(), QMainWindow, About dialog for the Bayesian Analyzer.

### Community 81 - "actions.py"
Cohesion: 0.40
Nodes (3): Qt action factory + bulk action construction for the Bayesian Analyzer.  The tes, Check the right MC action; uncheck the rest., set_monte_carlo()

## Knowledge Gaps
- **71 isolated node(s):** `wsi-analysis`, `graphify`, `Overview`, `Why Bayesian Networks?`, `Dataset` (+66 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `StartAnalysis` connect `Base Bayesian Model` to `PyQt5 Image Viewer`, `Multi-Process Analysis`, `OpenSeadragon Wheel Events`, `OpenSeadragon DZI Parser`, `._update_training_log`, `OpenSeadragon XML10 Parser`, `DataClean UI Builders`, `DataClean Long-Running Workers`, `start_tile_threads`, `Config & Init`, `OpenSeadragon Pointer Events`, `DataClean Main Window`, `DataClean Uncertainty Runner`?**
  _High betweenness centrality (0.086) - this node is a cross-community bridge._
- **Why does `ImageViewer` connect `PyQt5 Image Viewer` to `Multi-Process Analysis`, `OpenSeadragon DZI Parser`, `test_actions_factory.py`, `OpenSeadragon Stop Fade`, `about_dialogs.py`, `._start_analysis`, `start_tile_threads`, `actions.py`, `Base Bayesian Model`, `._tile_dataset`, `DataClean Uncertainty Runner`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Why does `Classification` connect `Classification Module` to `DataClean Long-Running Workers`, `Multi-Process Analysis`, `OpenSeadragon Stop Fade`, `UncertaintyTab`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `ImageViewer` (e.g. with `AnalyzerState` and `StartAnalysis`) actually correct?**
  _`ImageViewer` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `StartAnalysis` (e.g. with `ImageViewer` and `GetTilesTab`) actually correct?**
  _`StartAnalysis` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Classification` (e.g. with `UncertaintyTab` and `TestOverlayAxisAlignment`) actually correct?**
  _`Classification` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `wsi-analysis`, `graphify`, `Overview` to the rest of the system?**
  _71 weakly-connected nodes found - possible documentation gaps or missing edges._