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

## Model quality monitoring during training

As the training goes, the model gets evaluated every `save_every` epochs (as defined in the model config). The evaluation results are written in the [TensorBoard](https://www.tensorflow.org/tensorboard/) format in the `logs/CHECKPOINT_NAME` folder. Some simple quantities like the generator and discriminator losses are written every epoch. TensorFlow provides a tool — the TensorBoard server — to interactively monitor this information in a web browser. The TensorBoard server is included as a dependency in the `requirements_minimal.txt`, so it should already be installed on your machine if you followed the instructions above.

In case you run everything on your local machine, it should be sufficient to just run:
```bash
tensorboard --logdir=logs/
```
This should start a server that's going to be accessible via http://localhost:6006/ in your browser locally. If you run everything on a remote machine accessed via SSH, you'll also need to make use of the SSH port forwarding to be able to acces the server on your local machine web browser. This can be done with the `-L <LOCAL_PORT>:<HOST>:<PORT>` SSH option, which forwards all local connections to the `<LOCAL_PORT>` to the `<HOST>:<PORT>` from the remote machine. For example, you can make an SSH connection to your `REMOTE_MACHINE` with:
```bash
ssh -L 4321:localhost:6006 username@REMOTE_MACHINE
```
after which opening http://localhost:4321/ in your browser locally would be forwarded through SSH as if you opened http://localhost:6006/ on `REMOTE_MACHINE`. The port 6006 is the default TensorBoard port, but it can be configured to be any other using the `--port` argument of the `tensorboard`.

Once you configure (if necessary) the port forwarding, start the TensorBoard server and access it through the web browser, you should see a page with two tabs: `SCALARS` and `IMAGES`. The `SCALARS` tab contains the generator and discriminator losses, along with a quantity called `chi2`. This `chi2` quantity is a sum of squared discrepancy-over-error terms, where discrepancies are calculated between the data and model prediction for the upper and lower bands in each bin of profiles like Fig.3 from [Eur. Phys. J. C 81, 599 (2021)](https://doi.org/10.1140/epjc/s10052-021-09366-4) (excluding the amplitude profiles). The `chi2` quantity is not technically a chi-squared due to the correlations between different terms, but it does reflect the overall agreement of the model (the lower `chi2` the better). The `IMAGES` tab should contain validation histograms and profiles and example responses generated.
