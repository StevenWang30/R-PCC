import sys
import yaml
from easydict import EasyDict


def sys_size(data):
    return sys.getsizeof(data)


def bit_size(data):
    return len(data)


def np_size(data):
    return data.nbytes


def load_compressor_cfg(yaml_file):
    with open(yaml_file, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    compressor_config = EasyDict(config)
    return compressor_config


def write_bitstream(bitstream, info):
    with bz2.open("myfile.bz2", "wb") as f:
        # Write compressed data to file
        unused = f.write(data)
    with bz2.open("myfile.bz2", "rb") as f:
        # Decompress data from file
        content = f.read()