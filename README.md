### Requirements

```
pip install -r requirements.txt
```

### Quickstart

#### Downloading java-small dataset
The java-small dataset is downloaded and cached in the `data/` directory by default, along with a custom loading script. If you want to use a different directory, simply modify the `CHECKPOINT_PATH` in the `Config` file and copy the script to the desired location.

#### Training a model
You have the option of selecting between two models for training: `GPT-2` and `RoBERTa`. This can be done using the `-m` option ([`gpt`, `roberta`]). Additionally, prior to commencing training, you can make adjustments to the configuration hyper-parameters by modifying the `config.py` file.
The training objective for `RoBERTa` is mlm, whereas for `GPT-2` it is next token prediction.

To train a model run:
```
python train.py -m [MODEL] -c [SAVE_PATH]
```