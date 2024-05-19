import yaml
import argparse

def import_yaml(fpath):
	with open(fpath) as stream:
		config = yaml.load(stream, Loader=yaml.FullLoader)
	return config

def argument_parser():
    parser = argparse.ArgumentParser(description='Argument Parser for RenderDB')

    ##### Configurations #####
    parser.add_argument('--config',
                        type=str,
                        default='./config/config.yaml',
                        help='config path direction')

    args = parser.parse_args()
    return args