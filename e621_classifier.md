# e621 Classifier Windows Service

This script can be run as a Windows Service to watch a set of folders and automatically
add e621 classification tags to every eligible image using the JTP-3 model.

## Installation Instructions

1) Perform these steps as normal for installing tail-tagger (see [README.md]):
   * Run `setup.bat`.
   * Run `venv\Scripts\activate.bat`
   * Download the JTP-3 model and place it in correct classifier folder.
2) Build the service package:
   * `python -m pip install -r requirements-service.txt`
   * Download EverythingSDK3 and unzip it into `everything_sdk`
   * `pyinstaller --noconfirm e621-service.spec`
3) Deploy the service package:
   * Stop the service if it already exists.
   * Copy the `dist\e621-service` directory tree to `c:\ProgramData\e621 Classifier`
   * Make any desired edits to the config in `c:\ProgramData\e621 Classifier\e621-classifier.toml`
   * If service not already installed: `c:\ProgramData\e621 Classifier\e621-classifier install`
   * Start service: `c:\ProgramData\e621 Classifier\e621-classifier start`