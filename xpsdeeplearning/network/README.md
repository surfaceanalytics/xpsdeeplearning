# Deep learning for XPS spectra

This tool contains a supervised machine learning framework based on artificial convolutional neural networks that is able to accurately perform automated quantitative analysis of the phases present in such XP spectra.

# Usage
## Train from the command line
```console
user@box:~$ train --help
Usage: train [OPTIONS]

  Train a CNN on new data.

Options:
  --param-file TEXT  The path to the training parameter file.  [required]
  --help             Show this message and exit.

```console
user@box:~$ predict --help
Usage: predict [OPTIONS]

  Predict using an existing classifier.

Options:
  --param-file TEXT  The path to the parameter file for this experiment.
                     [required]
  --clf-path TEXT    The path to the existing classifier.  [required]
  --help             Show this message and exit.
