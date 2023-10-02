# urop-surgical-guide-analysis

## Description

This repository contains code to analyse Bill King's sugrical guide dataset. The `run.py` file handles all the magic:
- extracting images of the STL cross-sections
- segmenting the photos
- aligning photos to corresponding cross-sections
- analysing the results
- 
Please note that I have already run this code on the dataset. The folder containing the results is at <link!>

There are also 2 other files, aimed at visualising and understanding the resutls.
- `lookatheatmaps.py` allows you to look through plots of the heatmaps compared with the original photo and cad file, by default in descending order of error magnitude:
![alt text](https://github.com/suspicious-salmon/urop-surgical-guide-analysis/blob/main/readme-images/lookatheatmaps_demo.png?raw=true)
- `get_biggest_dft.py` helps with finding parts which might have high errors. It sorts the csv file in Bill King's dataset, providing the highest-DFT (difference from target) parts for each feature category.

## Getting Started

### Dependencies

I ran the code in Windows 11 Anaconda, Python 3.8.17 and using the modules contained in `environment.yml`.
Also needed (for pyimagej) is Java OpenJDK, mine was version 20. Set your JAVA_HOME environment variable to JDK's root directory (e.g. mine was `C:\Program Files\Java\jdk-20`)

### Quickstart

Set up the environment by, in anaconda terminal in repository folder, executing `conda env create -f environment.yml` (this might take a while to install everything). It will create an environment called my_surgical_guide_env, or whatever you change the first line to in `environment.yml`.

Then, edit `run.py`:
- Change `root_dataset_folder` to the directory where the folder containing Bill King's dataset is.
- Change `root_output_folder` to the directory where you would like your results to be stored. Be warned: for 258 surgical parts, this needs about 23GB of space!

Now try executing `run.py` in the environment you just made - it should work.
If you see the message `[ERROR] Cannot create plugin: org.scijava.plugins.scripting.javascript.JavaScriptScriptLanguage`, don't worry, this is expected and the code should run fine anyway.

For each new run, use an empty `root_output_folder`.
