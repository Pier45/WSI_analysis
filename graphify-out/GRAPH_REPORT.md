# Graph Report - .  (2026-07-12)

## Corpus Check
- Large corpus: 77 files · ~1,318,185 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 631 nodes · 1046 edges · 63 communities (46 shown, 17 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 89 edges (avg confidence: 0.6)
- Token cost: 19,900 input · 4,450 output

## Community Hubs (Navigation)
- OpenSeadragon tile viewer core
- jQuery internals (DeepZoom static)
- Docs, bugs & sample data
- OpenSeadragon pointer/mouse handlers
- KL-Flipout Bayesian model
- Multiprocessing rationale
- UI launchers & progress bar
- Bayesian Analyzer GUI (ui_pyqt5)
- OpenSeadragon overlays/controls
- MC-Dropout Bayesian model
- OpenSeadragon tile loading
- Performance/confusion-matrix widget
- OpenSeadragon touch handlers
- models package init
- Classification & uncertainty overlay
- MainTabWidget training slots
- MainTabWidget UI builders
- Qt helpers (lines, matplotlib canvas)
- OpenSeadragon mouse-move handlers
- jQuery Sizzle selectors
- Bayesian model Protocol (base.py)
- Uncertainty thresholding (Th)
- Archive: TF1 dataset creation
- DeepZoom Flask server
- Bayesian Analyzer viewer slots
- jQuery AJAX/fx internals
- OpenSeadragon scalebar plugin
- Bayesian Analyzer init/menus
- Bayesian Analyzer tiling/analysis
- Training progress callbacks
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61

## God Nodes (most connected - your core abstractions)
1. `MainTabWidget` - 54 edges
2. `ImageViewer` - 38 edges
3. `StartAnalysis` - 29 edges
4. `Classification` - 26 edges
5. `PerformanceTab` - 24 edges
6. `BayesianDropoutCNN` - 19 edges
7. `ModelKl` - 19 edges
8. `Th` - 19 edges
9. `getMouseAbsolute()` - 15 edges
10. `LongRunningWorker` - 15 edges

## Surprising Connections (you probably didn't know these)
- `HorizontalLine` --uses--> `BayesianDropoutCNN`  [INFERRED]
  ui_dataclean.py → models/drop_out.py
- `LongRunningWorker` --uses--> `BayesianDropoutCNN`  [INFERRED]
  ui_dataclean.py → models/drop_out.py
- `MainTabWidget` --uses--> `BayesianDropoutCNN`  [INFERRED]
  ui_dataclean.py → models/drop_out.py
- `MainWindow` --uses--> `BayesianDropoutCNN`  [INFERRED]
  ui_dataclean.py → models/drop_out.py
- `MatplotlibCanvas` --uses--> `BayesianDropoutCNN`  [INFERRED]
  ui_dataclean.py → models/drop_out.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Two Bayesian Uncertainty Approaches (MC-Dropout + KL-Flipout)** — readme_mc_dropout, readme_kl_divergence_vi [EXTRACTED 1.00]
- **Three-Class Tile Set (AC/AD/H) - docs** — readme_class_ac, readme_class_ad, readme_class_h [EXTRACTED 1.00]
- **Three GUI/Server Entry Points** — readme_bayesian_analyzer_gui, readme_dataclean_gui, readme_deepzoom_viewer [EXTRACTED 1.00]
- **Thesis methodology-to-application pipeline figures (dataset -> uncertainty concept -> GUI -> output maps)** —  [INFERRED 0.75]
- **Uncertainty quantification conceptualisation and its concrete output** —  [INFERRED 0.85]
- **Three-Class Tile Set (AC/AD/H)** — icons_ac, icons_ad, icons_h [EXTRACTED 1.00]

## Communities (63 total, 17 thin omitted)

### Community 0 - "OpenSeadragon tile viewer core"
Cohesion: 0.03
Nodes (11): TODO: get rid of this.  I can't see how it's required at all.  Looks, TODO: Determine what this function is intended to do and if it's actually, TODO: Determine what this function is intended to do and if it's actually, TODO: Figure out why this is used on the public API and if a more useful, TODO: Figure out why this is used on the public API and if a more useful, TODO: What is this for? Future keyboard navigation support?, NOTE: This event is only fired when the drawer is using a <canvas>., TODO: Add a method 'one' which automatically unbinds a listener after the first (+3 more)

### Community 1 - "jQuery internals (DeepZoom static)"
Cohesion: 0.05
Nodes (4): NOTE: This can be skipped if there are no unmatched elements (i.e., `matchedCoun, TODO: Now that all calls to _data and _removeData have been replaced, TODO: identify versions, TODO: identify versions

### Community 2 - "Docs, bugs & sample data"
Cohesion: 0.09
Nodes (27): Sample Tile Dictionary (MC=5), Sample Tile Dictionary (Train), Sample Tile Dictionary (Validation), Original TF 1.15 Requirements (2019), Hardcoded WSL Path in Compose, Docker Compose (wsi-clean, wsi-analysis services), CLASS_NAMES Order Bug (AC/H/AD vs alphabetical), DropOut.py Constructor Signature Mismatch Bug (+19 more)

### Community 3 - "OpenSeadragon pointer/mouse handlers"
Cohesion: 0.14
Nodes (24): capturePointer(), getMouseAbsolute(), getPointerType(), handleMouseEnter(), handleMouseExit(), handlePointerMove(), handlePointerUp(), isParentChild() (+16 more)

### Community 4 - "KL-Flipout Bayesian model"
Cohesion: 0.10
Nodes (11): Callback, History, ImageDataGenerator, Model, Return the training :class:`ImageDataGenerator`, with optional augmentation., Create and return ``(train_generator, validation_generator)``.          Both gen, Constructs a Flipout Bayesian VGG-style model.          Args:             input_, Network block for VGG. (+3 more)

### Community 5 - "Multiprocessing rationale"
Cohesion: 0.13
Nodes (9): Test if the folder alredy exist, if true return 1 and the thread will stop, Divide the wsi in tiles, thanks to get_tile, if the test with fold managere is f, Divide the wsi in tiles, thanks to get_tile, if the test with fold managere is f, Manage the starting and ending point for the reading phase of the SVS file., Start the theads, in this way the process is faster., Create the folders where put the thumbnail and the tiles of the image., Create the thumbnail of the image, ready for the classification phase., Call this function to divide the slice in tiles, it manage the dimension and the (+1 more)

### Community 6 - "UI launchers & progress bar"
Cohesion: 0.13
Nodes (11): Actions, QObject, QRunnable, Bayesian Analyzer — main application window.  Entry point:     python ui_pyqt, Thread wrapper with full lifecycle signals.      Injects a ``progress_callback, Start the deepzoom Flask server in a separate thread and open the         brows, Signals emitted by :class:`WorkerLong` during its lifecycle., Fire-and-forget thread wrapper.      Runs *fn* in a thread-pool thread without (+3 more)

### Community 7 - "Bayesian Analyzer GUI (ui_pyqt5)"
Cohesion: 0.13
Nodes (6): ImageViewer, QMainWindow, Main application window for the Bayesian Analyzer.      Workflow     --------, Display a critical dialog when a background worker raises an exception., Open the deepzoom URL in the default browser after a short delay., Open the print dialog and print the currently displayed pixmap.

### Community 8 - "OpenSeadragon overlays/controls"
Cohesion: 0.12
Nodes (19): abortControlsAutoHide(), beginControlsAutoHide(), drawersNeedUpdate(), drawOverlays(), getOverlayObject(), _getSafeElemSize(), loadOverlays(), onBlur() (+11 more)

### Community 9 - "MC-Dropout Bayesian model"
Cohesion: 0.16
Nodes (12): BayesianDropoutCNN, ConvBlockConfig, History, ImageDataGenerator, Model, Convolutional neural network with Monte Carlo Dropout for Bayesian     uncertai, Return the training :class:`ImageDataGenerator`, with optional augmentation., Create and return ``(train_generator, validation_generator)``.          Both g (+4 more)

### Community 10 - "OpenSeadragon tile loading"
Cohesion: 0.14
Nodes (18): blendTile(), compareTiles(), drawDebugInfo(), drawTiles(), getTile(), isCovered(), loadTile(), offsetForRotation() (+10 more)

### Community 11 - "Performance/confusion-matrix widget"
Cohesion: 0.24
Nodes (4): PerformanceTab, QFrame, QWidget, QHLine

### Community 12 - "OpenSeadragon touch handlers"
Cohesion: 0.15
Nodes (17): abortTouchContacts(), getCaptureEventParams(), getCenterPoint(), getStandardizedButton(), handleMouseUp(), handleTouchEnd(), onMouseUp(), onMouseUpCaptured() (+9 more)

### Community 13 - "models package init"
Cohesion: 0.18
Nodes (8): Bayesian dropout CNN for histological-tile classification (AC / AD / H).  Mont, Bayesian uncertainty models package.  Re-exports the two model classes for conve, ModelKl, Variational inference (KL-divergence) Bayesian CNN for histological-tile classif, Bayesian CNN using KL-divergence variational inference (TF-Probability     Flipo, Project-wide configuration constants.  Single source of truth for class names, i, Source package — core logic (classification, uncertainty, widgets, config)., Bayesian Datacleaning Application Application for cleaning medical image datase

### Community 14 - "Classification & uncertainty overlay"
Cohesion: 0.17
Nodes (6): Classification, GREAT NOTE: LIST ARE index-1 for the 0 index, Analyze the selected folder, finding all the png files, Create the dict with the key the number of the tile, to each key correspond anot, This method read the image, modify it as numpy array and at the end control if s, Load the model and analyze the tile, the dictionary is updated with the predicte

### Community 16 - "MainTabWidget UI builders"
Cohesion: 0.17
Nodes (7): QHBoxLayout, QWidget, Builds the 'Get Tiles' tab for dataset folder selection and tiling., Builds the 'Training' tab for model configuration and launch., Builds the 'Uncertainty analysis' tab for MC Dropout classification., Costruisce il tab 'Data cleaning' con istogrammi e selezione soglia., Crea una riga orizzontale con etichetta e pulsante.

### Community 17 - "Qt helpers (lines, matplotlib canvas)"
Cohesion: 0.17
Nodes (8): FigureCanvasQTAgg, HorizontalLine, MatplotlibCanvas, QFrame, Decorative horizontal separator., Decorative vertical separator., Matplotlib canvas for rendering inline histograms., VerticalLine

### Community 18 - "OpenSeadragon mouse-move handlers"
Cohesion: 0.17
Nodes (13): getMouseRelative(), getPointRelativeToAbsolute(), handleMouseMove(), handlePointerStop(), handleTouchMove(), handleWheelEvent(), onMouseMove(), onMouseMoveCaptured() (+5 more)

### Community 19 - "jQuery Sizzle selectors"
Cohesion: 0.20
Nodes (12): addCombinator(), condense(), createPositionalPseudo(), elementMatcher(), markFunction(), matcherFromGroupMatchers(), matcherFromTokens(), multipleContexts() (+4 more)

### Community 20 - "Bayesian model Protocol (base.py)"
Cohesion: 0.20
Nodes (7): BayesianModel, History, Shared interface for the Bayesian uncertainty models.  Both ``BayesianDropoutCNN, Construct a model for training and uncertainty estimation.      Parameters (cons, Build, compile and train the model, returning the Keras history., Backward-compatible alias for :meth:`train`., Protocol

### Community 22 - "Archive: TF1 dataset creation"
Cohesion: 0.33
Nodes (3): _bytes_feature(), CreationTFRecord, _int64_feature()

### Community 23 - "DeepZoom Flask server"
Cohesion: 0.27
Nodes (7): BytesIO, index(), load_slide(), PILBytesIO, Classic PIL doesn't understand io.UnsupportedOperation., slugify(), tile()

### Community 24 - "Bayesian Analyzer viewer slots"
Cohesion: 0.24
Nodes (5): QImage, Prompt the user to select an ``.svs`` file, generate a thumbnail,         and k, Open the SVS file with :class:`StartAnalysis`, generate the         thumbnail a, Render *image* in the central label, scaling to fit the screen if needed., Load and display a result image.          Parameters         ----------

### Community 25 - "jQuery AJAX/fx internals"
Cohesion: 0.25
Nodes (9): ajaxConvert(), ajaxHandleResponses(), Animation(), createFxNow(), createTween(), defaultPrefilter(), done(), propFilter() (+1 more)

### Community 26 - "OpenSeadragon scalebar plugin"
Cohesion: 0.44
Nodes (7): getScalebarSizeAndText(), getScalebarSizeAndTextForMetric(), getSignificand(), getWithUnit(), log10(), normalize(), roundSignificand()

### Community 27 - "Bayesian Analyzer init/menus"
Cohesion: 0.25
Nodes (3): _get_screen_size(), Return the primary screen dimensions as ``(width, height)``.      Falls back t, Set the Monte Carlo sample count and uncheck the other options.

### Community 28 - "Bayesian Analyzer tiling/analysis"
Cohesion: 0.25
Nodes (4): Safely close the progress dialog if it is open., Return ``True`` if *name* already exists inside the working directory., Worker function: create PNG tiles for one process partition.          Paramete, Called when the analysis thread finishes successfully.

### Community 29 - "Training progress callbacks"
Cohesion: 0.29
Nodes (3): Callback, Emits per-batch and per-epoch training metrics via Qt signals.      Parameters, TrainingProgressCallback

### Community 30 - "Community 30"
Cohesion: 0.38
Nodes (7): buildFragment(), disableScript(), domManip(), getAll(), remove(), restoreScript(), setGlobalEval()

### Community 31 - "Community 31"
Cohesion: 0.38
Nodes (3): MainWindow, QMainWindow, Main application window for Bayesian Datacleaning.

### Community 32 - "Community 32"
Cohesion: 0.29
Nodes (3): Open the progress dialog with the given *title*., Launch one :class:`WorkerLong` per process partition to create tiles., Run the Bayesian classification in a background thread (if not already done).

### Community 33 - "Community 33"
Cohesion: 0.33
Nodes (4): LongRunningWorker, QRunnable, Generic worker for long-running operations on background threads.      Accepts, Avvia il training del modello selezionato in un thread separato.

### Community 36 - "Community 36"
Cohesion: 0.70
Nodes (5): AC Class Icon (Adenocarcinoma), AD Class Icon (Adenoma), Folder Icon, H Class Icon (Healthy), Target App Icon

### Community 39 - "Community 39"
Cohesion: 0.67
Nodes (4): Figure 4.2 - Tile extraction & dataset composition diagram, Figure 4.3 - Bayesian uncertainty quantification concept (MC Dropout aleatoric/epistemic), Figure 5.1 - WSI Analysis PyQt5 GUI screenshot, Figures 5.2 & 5.3 - WSI Analysis result overlays / uncertainty maps output

### Community 40 - "Community 40"
Cohesion: 0.50
Nodes (3): @opencode-ai/plugin, dependencies, @opencode-ai/plugin

### Community 41 - "Community 41"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 42 - "Community 42"
Cohesion: 0.50
Nodes (4): beginFading(), outTo(), scheduleFade(), updateFade()

### Community 43 - "Community 43"
Cohesion: 0.50
Nodes (4): beginZoomingIn(), beginZoomingOut(), doZoom(), scheduleZoom()

### Community 46 - "Community 46"
Cohesion: 0.67
Nodes (3): actualDisplay(), defaultDisplay(), showHide()

### Community 47 - "Community 47"
Cohesion: 0.67
Nodes (3): augmentWidthOrHeight(), curCSS(), getWidthOrHeight()

### Community 48 - "Community 48"
Cohesion: 0.67
Nodes (3): clearTrackedPointers(), startTracking(), stopTracking()

### Community 49 - "Community 49"
Cohesion: 0.67
Nodes (3): loadPanels(), onStripDrag(), onStripScroll()

### Community 50 - "Community 50"
Cohesion: 0.67
Nodes (3): processDZI(), processDZIResponse(), processDZIXml()

### Community 51 - "Community 51"
Cohesion: 0.67
Nodes (3): QObject, Signals emitted by LongRunningWorker to communicate with the main thread., WorkerSignals

## Knowledge Gaps
- **15 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `@opencode-ai/plugin`, `wsi-analysis`, `Aleatoric Uncertainty` (+10 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **17 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MainTabWidget` connect `MainTabWidget training slots` to `Community 33`, `Community 34`, `Community 35`, `Multiprocessing rationale`, `Community 37`, `Community 38`, `MC-Dropout Bayesian model`, `Performance/confusion-matrix widget`, `Community 44`, `models package init`, `Classification & uncertainty overlay`, `MainTabWidget UI builders`, `Qt helpers (lines, matplotlib canvas)`, `Uncertainty thresholding (Th)`, `Community 57`, `Community 58`, `Community 31`?**
  _High betweenness centrality (0.077) - this node is a cross-community bridge._
- **Why does `StartAnalysis` connect `Multiprocessing rationale` to `Community 33`, `UI launchers & progress bar`, `Bayesian Analyzer GUI (ui_pyqt5)`, `Community 44`, `models package init`, `MainTabWidget training slots`, `Qt helpers (lines, matplotlib canvas)`, `Community 51`, `Bayesian Analyzer viewer slots`, `Bayesian Analyzer tiling/analysis`, `Community 31`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Why does `Classification` connect `Classification & uncertainty overlay` to `Community 32`, `Community 33`, `Community 34`, `UI launchers & progress bar`, `Bayesian Analyzer GUI (ui_pyqt5)`, `models package init`, `MainTabWidget training slots`, `Qt helpers (lines, matplotlib canvas)`, `Community 51`, `Community 31`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `MainTabWidget` (e.g. with `BayesianDropoutCNN` and `ModelKl`) actually correct?**
  _`MainTabWidget` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `ImageViewer` (e.g. with `StartAnalysis` and `Classification`) actually correct?**
  _`ImageViewer` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `StartAnalysis` (e.g. with `HorizontalLine` and `LongRunningWorker`) actually correct?**
  _`StartAnalysis` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `Classification` (e.g. with `HorizontalLine` and `LongRunningWorker`) actually correct?**
  _`Classification` has 11 INFERRED edges - model-reasoned connections that need verification._