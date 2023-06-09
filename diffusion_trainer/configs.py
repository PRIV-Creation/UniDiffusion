import yaml
from argparse import ArgumentParser


def str2bool(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
        return True
    elif 'FALSE'.startswith(ua):
        return False
    else:
        raise Exception('Error!')


class Config:

    def __init__(self):
        self.config_parser = self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # Base configs
        self.parser.add_argument('-c', '--config', default='', type=str, nargs='+', metavar='FILE', help='YAML config file specifying default arguments')
        self.parser.add_argument('--output_dir', default='output_dir/', type=str, help='Path to experiment output directory')

        # Data configs
        self.parser.add_argument('--transform_type', default='encodetransforms', type=str, help='Type of dataset trans')
        self.parser.add_argument('--train_dataset_path', default=None, type=str)
        self.parser.add_argument('--test_dataset_path', default='../test', type=str)
        self.parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training.')
        self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference.')
        self.parser.add_argument('--workers', default=0, type=int, help='Number of train dataloader workers.')
        self.parser.add_argument('--test_workers', default=0, type=int,
                                 help='Number of test/inference dataloader workers.')
        self.parser.add_argument('--auto_resume', default='False', type=str2bool,
                                 help='Whether to automatically resume. During training, the last checkpoint will be '
                                      'used to resume while the best checkpoint will be used in '
                                      'inference/evaluation/editing process.')

        # Model configs
        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to model checkpoint.')
        self.parser.add_argument('--stylegan_weights', default="", type=str, help='Path to StyleGAN model weights.')
        self.parser.add_argument('--encoder_type', default='Encoder4Editing', type=str, help='Which encoder to use')
        self.parser.add_argument('--encoder_backbone', default='vit_small', type=str, help='Which encoder to use')
        self.parser.add_argument('--encoder_backbone_weight', default='', type=str, help='Which encoder to use')
        self.parser.add_argument('--start_from_latent_avg', action='store_true', help='Whether to add average latent '
                                                                                      'vector to generate codes from '
                                                                                      'encoder.')
        self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')
        self.parser.add_argument('--input_nc', default=3, type=int, help='number of channels of the first encoder layer')
        self.parser.add_argument('--layers', default=50, type=int, help='Number of layers of backbone')

    def parse(self):
        opts = self.config_parser.parse_args()
        if opts.config:
            for config in opts.config:
                with open(config, 'r') as f:
                    cfg = yaml.safe_load(f)
                    self.parser.set_defaults(**cfg)
        opts = self.parser.parse_args()
        return opts
