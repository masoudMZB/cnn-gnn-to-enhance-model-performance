# Changelog

## [1.0.0] - $(date +'%Y-%m-%d')
### Changed
- Converted `node_features.ipynb` to `node_features.py`. Cleaned up the script format and encapsulated parts into functions to ensure it runs correctly.
- Converted `Draw_plots.ipynb` to `Draw_plots.py`. Encapsulated the logic correctly, including plotting utilities.

### Added
- Added `test_node_features.py` to write basic unit tests for GCN model creation and PCA feature processing in `node_features.py`.
- Added `test_Draw_plots.py` to write basic unit tests for the functions `select_best_run_final_epoch` and `mean_performance` in `Draw_plots.py`.
