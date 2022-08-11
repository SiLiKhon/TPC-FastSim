# TPC-FastSim

## Minimalistic environment setup (CentOS 7)

The code was tested with python 3.6. The minimal set of packages is present in `requirements_minimal.txt`. Installing all these can be done with:

```bash
yum -y install python3-3.6.8-18.el7
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements_minimal.txt
```

## Runing the training and/or evaluation of the model
The entry point to model training and evaluation is the script called `run_model_v4.py`. It has the following arguments:

argument | description
-|-
`--checkpoint_name CHECKPOINT_NAME` | Name to identify a trained model instance with. If starting to train a new model, this will create a directory `saved_models/CHECKPOINT_NAME` in which the configuration and the states of the model at various epochs will be saved.
`--config CONFIG` | Configuration file to setup the model and training. This parameter is obligatory when starting to train a **new** model. This parameter should be omitted when resuming a previously stopped training process.<br />Some pre-defined model configs can be found in `models/configs`. At the moment of writing this documentation, the current best model (with 6 input parameters, taking the pad type and transverse momentum into account) is defined with the configuration file `models/configs/v4plus_full_fc_8x16_kinked_GP50_js.yaml`. The previous best model (with 4 input parameters) is `models/configs/baseline_fc_8x16_kinked.yaml` (this is the one used in the publication [Eur. Phys. J. C 81, 599 (2021)](https://doi.org/10.1140/epjc/s10052-021-09366-4)).
`--gpu_num GPU_NUM` | GPU index to pick a single GPU on a multi-GPU machine. Provide an empty string (`''`) in case of running this code on a non-GPU machine.
`--prediction_only` | Boolean flag to switch to the model evaluation mode. Running with this option would produce evaluation plots like Fig.3 from [Eur. Phys. J. C 81, 599 (2021)](https://doi.org/10.1140/epjc/s10052-021-09366-4). The plots will be saved under `saved_models/CHECKPOINT_NAME/prediction_XXXXX`, where `XXXXX` is the epoch number picked for evaluation (the latest one available).

An example command to run model training:
```bash
python3 run_model_v4.py --config models/configs/v4plus_full_fc_8x16_kinked_GP50_js.yaml --checkpoint_name test_run --gpu_num ''
```

An example command to run model evaluation:
```bash
python3 run_model_v4.py --checkpoint_name test_run --gpu_num '' --prediction_only
```
