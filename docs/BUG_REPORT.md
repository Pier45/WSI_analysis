# Bug Report — DropOut.py

## 🔴 Critical Bugs

### Bug 1 — `ImageDataGenerator` not imported
- **Lines affected:** 173, 181, 191 (usage) / 28–30 (commented-out imports)
- **Issue:** `ImageDataGenerator` is called but its import is commented out.
- **Impact:** `NameError: name 'ImageDataGenerator' is not defined` — crashes immediately when training.
- **Fix:** Uncomment the import:
  ```python
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  ```

### Bug 2 — Parameter name mismatch between caller and constructor
- **Lines affected:** `__init__` (lines 141–164) vs `ui_dataclean.py:736–740`
- **Issue:** The caller passes `n_model`, `b_dim`, `aug` but the constructor expects `model_save_path`, `batch_size`, `augment`.
- **Impact:** `TypeError: __init__() got an unexpected keyword argument 'n_model'` — crashes when instantiated from the UI.
- **Constructor params:** `model_save_path`, `epochs`, `path_train`, `path_val`, `batch_size`, `augment`
- **Caller params:** `n_model`, `epochs`, `path_train`, `path_val`, `b_dim`, `aug`
- **Fix:** Rename constructor parameters to match the established interface (same as `ModelKl` in `Kl.py`):
  ```python
  def __init__(self, n_model, epochs, path_train, path_val, b_dim, augment=0)
  ```

---

## 🟡 Medium Bugs

### Bug 3 — `CLASS_NAMES` order may not match downstream code
- **Line affected:** 48
- **Issue:** `CLASS_NAMES = ["AC", "H", "AD"]` overrides the default alphabetical order. Downstream code in `uncertainty_analysis.py` uses `{'AC': 0, 'AD': 0, 'H': 0}` (alphabetical), so class indices may be misinterpreted.
- **Fix:** Change to alphabetical order:
  ```python
  CLASS_NAMES = ["AC", "AD", "H"]
  ```

### Bug 4 — `history_save_path` assumes `.h5` extension
- **Line affected:** 151
- **Issue:** `model_save_path.replace(".h5", "_history.json")` silently fails for non-`.h5` extensions (e.g., `.keras`, `.hdf5`), producing a malformed filename.
- **Fix:**
  ```python
  base, _ = os.path.splitext(model_save_path)
  self.history_save_path = base + "_history.json"
  ```

### Bug 5 — `ModelCheckpoint` saves to a hardcoded relative path
- **Lines affected:** 313–319
- **Issue:** `filepath="weights_best.h5"` saves to the current working directory instead of alongside `self.model_save_path`.
- **Fix:**
  ```python
  ModelCheckpoint(
      filepath=os.path.join(os.path.dirname(self.model_save_path) or ".", "weights_best.h5"),
      ...
  )
  ```

---

## 🟡 Low Bugs

### Bug 6 — Batch-level accuracy key may be version-dependent
- **Line affected:** 97
- **Issue:** `logs.get("accuracy", ...)` — some TF/Keras versions use `"acc"` for batch-level logs. Epoch-level (line 102) uses `"accuracy"` which is stable.
- **Note:** Needs testing against the installed TF version.

---

## 🟢 Minor Improvements

| # | Issue | Fix |
|---|-------|-----|
| 7 | Unused import: `field` from `dataclasses` (line 19) | Remove `field` from the import |
| 8 | Docstring references `model_training.py` (line 11) but file is `DropOut.py` | Update docstring |
| 9 | Missing explicit `class_mode="categorical"` in `flow_from_directory` kwargs | Add `class_mode="categorical"` to `flow_kwargs` (line 193) |
| 10 | `hist_df.to_json(fp)` with file pointer is deprecated in newer pandas | Use `hist_df.to_json(self.history_save_path)` with path string instead |
