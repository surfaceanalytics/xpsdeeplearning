# Example training and prediction parameters

These JSON files can be used as examples for training and prediction from the
command line. After installing `xpsdeeplearning`, you can download examplary
dataset and trained model for Ni spectra from [`KEEPER`](https://keeper.mpdl.mpg.de/d/25ebee640ba54622864a/).

You can then test the training
```shell
train --param-file train_params.json
```
and prediction pipelines:
```shell
predict --param-file predict_params.json --clf-path <path-to-clf>
```
where `path-to-clf` is the path to a trained classifier model.

In case you want to run the cli tools on different data, please change the
JSON files appropriately.