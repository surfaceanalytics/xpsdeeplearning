# Data set simulation

This tool contains tools for creating large data sets of synthetic (yet realistic) transition-metal XP spectra base on reference data. Exemplary JSON files containing the parameters used during simulation available in the [`params`](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/simulation/params) subfolder. The reference data that is used as input for the simulation is available inside [`data`](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/data).

# Usage
## Data simulation
```console
user@box:~$ simulate --help
Usage: simulate [OPTIONS]

  The CLI entrypoint for the convert function

Options:
  --param-file TEXT               The path to the input parameter file to
                                  read.  [required]
  --reload-from-previous-folder TEXT
                                  The path to a previous run which is to be
                                  continued.
  --help                          Show this message and exit.


