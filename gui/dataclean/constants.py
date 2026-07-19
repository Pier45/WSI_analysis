"""User-tunable defaults and paths for the Datacleaning application.

These were originally module-level constants in ``ui_dataclean.py``; they
have been hoisted into this module so that any tab widget or the main
window can import them without recreating a circular import through
``MainTabWidget``.
"""

from __future__ import annotations

DEFAULT_TRAIN_PATH = "test/train"
DEFAULT_VAL_PATH = "test/val"
DEFAULT_MODEL_PATH = "Model_1_85aug.h5"
DEFAULT_WORK_PATH = "test"
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 100
DEFAULT_MONTE_CARLO_SAMPLES = 5
DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_LEVEL = 0

# Manual threshold editor bounds (exclusive). Outside this range the value
# is silently rejected — see ``CleaningTab._apply_manual_threshold``.
MANUAL_THRESHOLD_MIN = 0.1
MANUAL_THRESHOLD_MAX = 1.0

# Lower-cased canonical class folder names. The Get-Tiles tab scans each
# selected dataset folder for sub-folders matching these (case-insensitive)
# and populates the QListWidgets accordingly.
KNOWN_CLASSES = {"ac", "ad", "h"}

APP_ICON_PATH = "icons/target.png"
APP_STYLE_PATH = "styles/stileor.css"

TUTORIAL_MESSAGE = (
    "The program is divided into tabs; follow the tab sequence to ensure everything works correctly.\n\n"
    "Tab 'Get tiles': select the working folder, then the train/val/test folders "
    "(SVS files must be organised into subfolders AC, AD, H). Press Start.\n\n"
    "Tab 'Training': configure the model parameters and start training.\n\n"
    "Press OK to continue."
)
