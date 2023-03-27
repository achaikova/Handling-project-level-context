from argparse import ArgumentParser

from config import Config
from model import Model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint", dest="checkpoint_path",
                        help="path to store a model checkpoint", required=False)
    parser.add_argument("-m", "--model", dest="model",
                        help="model to train", choices=['gpt', 'roberta'], required=True)
    args = parser.parse_args()

    config = Config.get_default_config(args)
    model = Model(config)

    model.train()
