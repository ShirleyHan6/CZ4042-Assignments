import argparse
import pickle

from configs import OUTPUT_DIR, BASE_DIR
from helper.utils import plot_train_and_test
from train_image_classifier import train_image_classifier


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--epoch', type=int, default=800, help='epoch number for training')
    parser_train.add_argument('--bs', type=int, default=128, help='batch size for training and testing')
    parser_train.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser_train.add_argument('--momentum', type=float, default=0, help='momentum')
    parser_train.add_argument('--optimizer', type=str, default='sgd', help='optimizer for training')
    parser_train.add_argument('--output', type=str, default='', help='output name of model and statistic result')
    parser_train.add_argument('--config', type=str, default=str(BASE_DIR / 'configs/image-classifier-best.yaml'))
    parser_train.set_defaults(func=train_model)

    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('file_name', type=str, default='', help='name of the statistic result')
    parser_plot.set_defaults(func=plot)
    return parser.parse_args()


def train_model(args):
    content = train_image_classifier(args)
    # save statistic
    with open(OUTPUT_DIR / '{}-stat-{:s}.pkl'.format(args.output, content['info']['name_seed']), 'wb') as f:
        pickle.dump(content, f)


def plot(args):
    plot_train_and_test(args.file_name)


if __name__ == '__main__':
    args_ = parse_args()
    args_.func(args_)
