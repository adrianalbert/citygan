import torch

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

# Mean [0.03400000184774399, 164.42999267578125, 1.3049999475479126, 0.8849999904632568, 0.08799999952316284]
# Stdv [0.16899999976158142, 1040.2120361328125, 16.354999542236328, 0.3100000023841858, 0.2150000035762787]

normalize_channels = ([0, 164.42, 1.304, 0, 0],
                      [1, 1040.413, 16.354, 1, 1])

def fn_filter_contains(fnames, any_of=[]):
    return [sum([x in f for x in any_of])>0 for f in fnames]

def main(config):
    prepare_dirs_and_logger(config)

    torch.manual_seed(config.random_seed)
    if len(config.gpu_ids) > 0:
        torch.cuda.manual_seed(config.random_seed)

    if config.is_train:
        data_path = config.data_path
        batch_size = config.batch_size
        do_shuffle = True
    else:
        if config.test_data_path is None:
            data_path = config.data_path
        else:
            data_path = config.test_data_path
        batch_size = config.sample_per_image
        do_shuffle = False

    if config.src_names is not None:
        config.src_names = config.src_names.split(",")

    if config.load_attributes is not None:
        config.load_attributes = config.load_attributes.split(",")

    if config.filter_by_pop is not None:
        fn_filter = lambda fnames: fn_filter_contains(fnames, any_of=config.filter_by_pop)
    else:
        fn_filter = None

    normalize = config.normalize if not config.normalize_channels else normalize_channels

    data_loader = get_loader(
        data_path, config.split, batch_size, config.input_scale_size, num_workers=config.num_worker, shuffle=do_shuffle, load_attributes=config.load_attributes, flips=config.flips, rotate_angle=config.rotate_angle, take_log=config.take_log, normalize=normalize, use_channels=config.use_channels, fn_filter=fn_filter)

    trainer = Trainer(config, data_loader)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
