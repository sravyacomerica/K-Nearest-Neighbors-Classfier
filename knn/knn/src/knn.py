#!/usr/bin/env python
# Example of execution: knn.py ../data/train.dat ../data/test.dat
# Author: Sri Sravya Tirupachur Comerica
# ID: 11259523
# Date: 2/28/2019
# Course Number: CSCE 5380 Data Mining
# Instructor: Eduardo Blanco Villar
# Implementation: Calculates the accuracy for k=1, k=3 and k=5 nearest neighbors

import logging
import math
import operator
import os
import random
import sys
from optparse import OptionParser


## Reads corpus and creates the appropiate data structures:
def read_corpus(file_name):
    f = open(file_name, 'r')

    ## first line contains the list of attributes
    attr = {}
    ind = 0
    for att in f.readline().strip().split("\t"):
        attr[att] = {'ind': int(ind)}
        ind += 1

    ## the rest of the file contains the instances
    instances = []
    ind = 0
    for inst in f.readlines():
        inst = inst.strip()

        elems = inst.split("\t")
        if len(elems) < 3: continue

        instances.append({'values': map(int, elems[0:-1]),
                          'class': int(elems[-1]),
                          'index': int(ind),
                          })
        ind += 1

    return attr, instances


# calculates accuracy for k=1 nearest neighbor
def get_closest_instance(test, instances, instances_list):
    prev_distance = float("inf")
    euclidean_list = {}
    closest_instance = 0
    i = 0
    for instance in instances_list:
        Sum_of_squares = 0
        index = 0
        # calculates euclidean distance between train and test
        for ins in instance:
            Sum_of_squares += (math.pow(ins - test[index], 2))
            index += 1
        euclidean = math.sqrt(Sum_of_squares)
        euclidean_list[i] = euclidean
        # updates the previous distance if a lesser distance is found
        if euclidean < prev_distance:
            prev_distance = euclidean
            closest_instance = i
        i += 1

    # gets predictions for k=3
    predictions_k3 = calculate_k3(euclidean_list, instances)
    # gets predictions for k=5
    predictions_k5 = calculate_k5(euclidean_list, instances)
    return instances[closest_instance], predictions_k3, predictions_k5


# calculates accuracy for k=3 nearest neighbor
def calculate_k3(euclidean_list, instances):
    class_zero = 0
    class_one = 0
    # sorts the hashmap
    sorted_by_value = sorted(euclidean_list.items(), key=lambda kv: kv[1])
    # gets first 3 closest instances index
    val1 = sorted_by_value[0]
    val2 = sorted_by_value[1]
    val3 = sorted_by_value[2]
    # compares their classes and increases the count of the found class by 1
    if instances[val1[0]]['class'] == 0:
        class_zero += 1
    else:
        class_one += 1
    if instances[val2[0]]['class'] == 0:
        class_zero += 1
    else:
        class_one += 1
    if instances[val3[0]]['class'] == 0:
        class_zero += 1
    else:
        class_one += 1
    # returns the class with highest count
    if class_zero > class_one:
        return 0
    else:
        return 1


# calculates accuracy for k=5 nearest neighbor
def calculate_k5(euclidean_list, instances):
    class_zero = 0
    class_one = 0
    # sorts the hashmap
    sorted_by_value = sorted(euclidean_list.items(), key=lambda kv: kv[1])
    # gets first 5 closest instances index
    val1 = sorted_by_value[0]
    val2 = sorted_by_value[1]
    val3 = sorted_by_value[2]
    val4 = sorted_by_value[3]
    val5 = sorted_by_value[4]
    # compares their classes and increases the count of the found class by 1
    if instances[val1[0]]['class'] == 0:
        class_zero += 1
    else:
        class_one += 1
    if instances[val2[0]]['class'] == 0:
        class_zero += 1
    else:
        class_one += 1
    if instances[val3[0]]['class'] == 0:
        class_zero += 1
    else:
        class_one += 1
    if instances[val4[0]]['class'] == 0:
        class_zero += 1
    else:
        class_one += 1
    if instances[val5[0]]['class'] == 0:
        class_zero += 1
    else:
        class_one += 1
    # returns the class with highest count
    if class_zero > class_one:
        return 0
    else:
        return 1


def calculate_accuracy(instances, predictions):
    predictions_ok = 0
    for i in range(len(instances)):
        if instances[i]["class"] == predictions[i]:
            predictions_ok += 1

    return 100 * predictions_ok / float(len(instances))


if __name__ == '__main__':
    usage = "usage: %prog [options] TRAINING_FILE TEST_FILE"

    parser = OptionParser(usage=usage)
    parser.add_option("-d", "--debug", action='store_true',
                      help="Turn on debug mode")

    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("Incorrect number of arguments")
    if not os.path.isfile(args[0]):
        parser.error("Training file does not exist\n\t%s" % args[0])
    if not os.path.isfile(args[1]):
        parser.error("Training file does not exist\n\t%s" % args[1])

    if options.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.CRITICAL)

    file_tr = args[0]
    file_te = args[1]
    logging.info("Training: " + file_tr)
    logging.info("Testing: " + file_te)

    ## (I)  Training: read instances
    attr_tr, instances_tr = read_corpus(file_tr)

    ## (II) Testing: read instances and
    ##      predict the class of the closest instance in training
    attr_te, instances_te = read_corpus(file_te)
    predictions = []
    predictions_k3 = []
    predictions_k5 = []
    train_instances = []
    test_instances = []
    list1 = list(filter(lambda instance: "values" in instance, instances_tr))
    for instance in list1:
        train_instances.append(list(instance['values']))
    list2 = list(filter(lambda instance: "values" in instance, instances_te))
    for instance in list2:
        test_instances.append(list(instance['values']))
    ## for each test instance
    for i_te in test_instances:
        ## get the closest one and store the prediction
        closest_instance, prediction_k3, prediction_k5 = get_closest_instance(i_te, instances_tr, train_instances)
        predictions.append(closest_instance["class"])
        predictions_k3.append(prediction_k3)
        predictions_k5.append(prediction_k5)

    if options.debug:
        print(predictions)

    accuracy_te = calculate_accuracy(instances_te, predictions)
    accuracy_te_k3 = calculate_accuracy(instances_te, predictions_k3)
    accuracy_te_k5 = calculate_accuracy(instances_te, predictions_k5)
    print("Accuracy on test set (%s  instances): %.2f" % (len(instances_te), accuracy_te))
    print("Accuracy on test set using k3(%s  instances): %.2f" % (len(instances_te), accuracy_te_k3))
    print("Accuracy on test set using k5(%s  instances): %.2f" % (len(instances_te), accuracy_te_k5))
