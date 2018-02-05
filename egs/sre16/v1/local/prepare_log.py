#!/usr/bin/env python

# Copyright 2017 Pegah Ghahremani

""" This script is used to generate some statistic and plots for training and test data.
"""

import argparse
import os
import sys
import re
import traceback
from itertools import izip
import errno
try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    g_plot = True
except ImportError:
    warnings.warn(
        """This script requires matplotlib and numpy.
        Please install them to generate plots.
        Proceeding with generation of tables.
        If you are on a cluster where you do not have admin rights you could
        try using virtualenv.""")
    g_plot = False


parser = argparse.ArgumentParser(description="""Compute test and train distribution
                                                for classification and plot error
                                                for each class based on prediction
                                                and plot report.pdf in the output dir.""")
parser.add_argument('--train-label', type=str, required=True,
                    help='Training data used to train classifier.'
                         'The file format is utt-id aga e.g. utt1 5.')
parser.add_argument('--test-label', type=str, required=True,
                    help='Test data used to test classifier on that.'
                         'the file format is utt-id age .e.g. utt1 10.')
parser.add_argument('--predicted-label', type=str, required=True,
                    help='The data directory for predicted classes.'
                         'This file is generated using evaluate_age.sh '
                         'and the format is true-shifted-age predicted-shifted-age '
                         ' e.g. 0 3.'
                         'The true and predicted age should be in same base.(i.e. zero-based age)'
                         ' and the original class label is the label in test-label in same line.' )
parser.add_argument("output_dir",
                    help="experiment directory, e.g. exp/report")

def compute_dist(eg2label_file):
    label_dist = [0] * 10000
    max_label = -1
    eg2label_info = open(eg2label_file, 'r')
    for line in eg2label_info:
        label = int(line.split(" ")[1])
        label_dist[label]+=1
        max_label = max(max_label, label)
    return label_dist[1:max_label];

def compute_error(eg2label_test, eg2label_predicted):
    eg2label_test_info = open(eg2label_test, 'r')
    eg2label_predicted_info = open(eg2label_predicted, 'r')
    error = [0] * 1000
    count = [0] * 1000
    max_label=-1
    for line1, line2 in izip(eg2label_test_info, eg2label_predicted_info):
        #assert(line1.split(" ")[0] == line2.split(" ")[0])
        true_label = int(line1.split(" ")[1])
        shifted_true_label = int(line2.split("\t")[0])
        predicted_label = int(line2.split("\t")[1])
        error[true_label] += abs(shifted_true_label - predicted_label)
        count[true_label] += 1
        max_label = max(max_label, true_label)
    for  i in range(max_label):
        if (count[i] > 0):
            error[i] = error[i] / count[i]
    return error[1:max_label]

def generate_plots(data_dist, data_type, output_dir):
    labels = range(0, len(data_dist))
    plt.bar(labels, data_dist)
    plt.title('{0} distribution'.format(data_type))
    plt.xlabel('class label')
    plt.ylabel('Frequency')
    plt.grid(True)
    fig = plt.gcf()
    figfile_name = '{0}/{1}_histogram.pdf'.format(output_dir, data_type)
    fig.suptitle("{0} data Distribution".format(data_type))
    plt.savefig(figfile_name, bbox_inches='tight')

def generate_plot2(train_data, test_data, error, output_dir):
    train_labels = range(0, len(train_data))
    ax = plt.subplot(211)
    ax.bar(train_labels, train_data, width=0.5, color='b', align='center')
    test_labels = range(0, len(test_data))
    ax.bar(test_labels, test_data, width=0.5, color='r', align='center')
    #ax.autoscale(tight=True)
    #ax.legend('Train', 'Test')
    ax2 = plt.subplot(212)
    ax2.bar(test_labels, error, width=0.5, color='g', align='center')
    figfile_name = '{0}/tot_distribution.pdf'.format(output_dir)
    plt.savefig(figfile_name, bbox_inches='tight')

def main():
    args = parser.parse_args()
    try:
        try:
            os.makedirs(args.output_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(args.output_dir):
                pass
            else:
                raise e
        train_dist = compute_dist(args.train_label);
        test_dist = compute_dist(args.test_label);
        predicted_error = compute_error(args.test_label, args.predicted_label)

        #generate_plots(train_dist, 'Train', args.output_dir)
        #generate_plots(test_dist, 'Test', args.output_dir)
        #generate_plots(predicted_error, 'Error', args.output_dir)

        generate_plot2(train_dist, test_dist, predicted_error, args.output_dir)
    except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    main()
