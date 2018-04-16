import tensorflow as tf

import glob

try:
    with open('config.json') as f:
        config = eval(f.read())
except SyntaxError:
    print('Unable to open the config.json file. Terminating...')
except IOError:
    print('Unable to find the config.json file. Terminating...')

def is_file_prefix(attr):
    return bool(glob.glob(get(attr) + '*'))

def get(attr, root=config):
    node = root
    for part in attr.split('.'):
        node = node[part]
    return node
