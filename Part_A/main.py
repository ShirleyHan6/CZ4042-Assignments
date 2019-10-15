import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from training import Train
from utils import plot_multiple_curves
import argparse

np.random.seed(10)


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def train_epochs(args):
	train = Train(args)
	training_accuracies = []
	testing_accuracies = []
	for i in range(1, args.epochs+1):
		train_accuracy, test_accuracy = train.train_one_epoch()
		training_accuracies.append(train_accuracy)
		testing_accuracies.append(test_accuracy)

	return training_accuracies, testing_accuracies

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Default hyper-parameters:
	parser.add_argument("-mode", type=str, default="train")
	parser.add_argument("-batch_size", type=int, default=32)
	parser.add_argument("-hidden_dim", type=int, default=10)
	parser.add_argument("-learning_rate", type=float, default=1e-2)
	parser.add_argument("-weight_decay", type=float, default=1e-6)
	parser.add_argument("-num_layer", type=int, default=3)

	parser.add_argument("-epochs", type=int, default=5000)
	parser.add_argument("-baseline", type=str2bool, nargs="?", const=True, default=True)
	parser.add_argument("-record_loss_interval", type=int, default=100)
	parser.add_argument("-save_model_interval", type=int, default=100)
	parser.add_argument("-tune_batch_size", type=str2bool, nargs="?", const=True, default=False)
	parser.add_argument("-tune_hidden_dim", type=str2bool, nargs="?", const=True, default=False)
	parser.add_argument("-tune_weight_decay", type=str2bool, nargs="?", const=True, default=False)
	parser.add_argument("-tune_num_layer", type=str2bool, nargs="?", const=True, default=False)
	parser.add_argument("-model_dir", type=str, default="models")
	parser.add_argument("-load_from_check_point", type=str, default="")

	args = parser.parse_args()
	# Baseline:

	training_accuracies = []
	testing_accuracies = []

	if args.baseline:
		training_accuracies, testing_accuracies = train_epochs(args)


	if args.tune_batch_size:
		batch_sizes = [4, 8, 16, 32, 64]
		for i in batch_sizes:
			args.batch_size = i
			training_accuracies, testing_accuracies = train_epochs(args)

	if args.tune_hidden_dim:
		hidden_dims = [5, 10, 15, 20, 25, 30]
		for i in hidden_dims:
			args.hidden_dim = i
			training_accuracies, testing_accuracies = train_epochs(args)

	if args.tune_weight_decay:
		weight_decays = [0, 1e-3, 1e-6, 1e-9, 1e-12]
		for i in weight_decays:
			args.weight_decay = i
			training_accuracies, testing_accuracies = train_epochs(args)

	if args.tune_num_layer:
		num_layers = [3, 4]
		for i in num_layers:
			args.num_layer = i
			training_accuracies, testing_accuracies = train_epochs(args)

	x = np.arange(1, args.epochs+1)


	plot_multiple_curves(x, [training_accuracies], legends_list=['train acc'])
	plot_multiple_curves(x, [testing_accuracies], legends_list=['test acc'])






