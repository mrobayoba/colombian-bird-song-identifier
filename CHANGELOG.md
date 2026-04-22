# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

### Added
- Notebook-first processing pipeline in [pre-processing.ipynb](pre-processing.ipynb):
  - Step 1 setup and runtime API key input
  - Step 2 Google Drive mount and temporary metadata paths
  - Step 3 metadata fetch from Xeno-Canto API v3 with raw page caching
  - Step 4 species/recording filtering logic integrated into notebook
  - Step 5 concurrent MP3 download to Drive
- Preprocessing planning document in [PREPROCESSING_CONTEXT.md](PREPROCESSING_CONTEXT.md) focused on chunking, mel-spectrograms, and tensor export strategy.

### Changed
- Authentication prompt now explicitly instructs users to generate API keys from the Xeno-Canto account page.
- Runtime workflow now stores intermediate JSON metadata under /content/data (temporary) and audio under Google Drive (persistent).
- [README.md](README.md) rewritten to reflect the active notebook workflow and current repository structure.

### Deprecated
- Legacy script-based pipeline marked as retired:
  - [fetch_metadata.py.retired](fetch_metadata.py.retired)
  - [filter_species.py.retired](filter_species.py.retired)

## [0.1.0] - Initial project state

### Added
- Initial metadata fetch and filtering scripts.
- Base project scaffolding and dataset artifacts.
