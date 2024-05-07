import argparse
import time

parser = argparse.ArgumentParser(description='Collaborative_HAR')
parser.add_argument('--save', default='save/unimib/default-{}'.format(time.time()),
                    type=str, metavar='SAVE',
                    help='path to the experiment logging directory'
                         '(default: save/debug)')
parser.add_argument('--model_name', type=str, default='uci', choices=['uci', 'wisdm', 'unimib', 'pamap2', 'oppo', 'usc', 'weakly', 'resnet_uci', 'resnet_wisdm', 'resnet_unimib', 'resnet_pamap2', 'resnet_oppo', 'resnet_usc', 'resnet_weakly'])

args = parser.parse_args()



