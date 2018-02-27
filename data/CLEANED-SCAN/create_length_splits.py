"""
Script to make it easier to split the data set on basis of lengths
"""

import os
import numpy as np
import random
from collections import Counter


def show_statistics():
    """
    Opens the file containing all data examples, counts the distribution of these
    examples in terms of length of both input and output, and prints these statistics
    """
    input_length_counter = Counter()
    output_length_counter = Counter()

    with open('tasks.txt', 'r') as all_tasks_file:
        total_number_of_tasks = 0

        for line in all_tasks_file:
            total_number_of_tasks += 1

            line = line.strip()
            input_sequence, output_sequence = line.split('\t')

            input_sequence_length = len(input_sequence.split())
            output_sequence_length = len(output_sequence.split())

            input_length_counter[input_sequence_length] += 1
            output_length_counter[output_sequence_length] += 1

        print("How often input lengths occur:")
        accumulative_input_count = 0
        for input_length in sorted(input_length_counter.keys()):
            count = input_length_counter[input_length]
            accumulative_input_count += count
            print("{} occurs {: <4} times ({:5.02f}%). Accumulative: {:.02f}%".format(input_length, count,
                                                                                      100.0 * count / total_number_of_tasks, 100.0 * accumulative_input_count / total_number_of_tasks))
        print("")

        print("How often input lengths occur (reversed):")
        accumulative_input_count = 0
        for input_length in sorted(input_length_counter.keys(), reverse=True):
            count = input_length_counter[input_length]
            accumulative_input_count += count
            print("{} occurs {: <4} times ({:5.02f}%). Accumulative: {:.02f}%".format(input_length, count,
                                                                                      100.0 * count / total_number_of_tasks, 100.0 * accumulative_input_count / total_number_of_tasks))
        print("")

        print("How often output lengths occur:")
        accumulative_output_count = 0
        for output_length in sorted(output_length_counter.keys()):
            count = output_length_counter[output_length]
            accumulative_output_count += count
            print("{: <2} occurs {: <4} times ({:4.02f}%). Accumulative: {:.02f}%".format(output_length, count,
                                                                                          100.0 * count / total_number_of_tasks, 100.0 * accumulative_output_count / total_number_of_tasks))
        print("")

        print("How often output lengths occur (reversed):")
        accumulative_output_count = 0
        for output_length in sorted(output_length_counter.keys(), reverse=True):
            count = output_length_counter[output_length]
            accumulative_output_count += count
            print("{: <2} occurs {: <4} times ({:4.02f}%). Accumulative: {:.02f}%".format(output_length, count,
                                                                                          100.0 * count / total_number_of_tasks, 100.0 * accumulative_output_count / total_number_of_tasks))
        print("")


def create_split(split_on, included, split_name):
    """
    Creates a train/test split on basis of either input or output length of
    the examples
    Args:
        split_on (str): Split on either 'input' or 'output'
        included (list): A list of all the lengths that should be included in the
        split_name (str): Name of the directory in which the train/test files will be stored
        training set. The rest are in the test set
    """
    input_file_all = open('tasks.txt', 'r')

    if not os.path.exists(os.path.join('length_split', split_name)):
        os.mkdir(os.path.join('length_split', split_name))
    output_file_train = open(os.path.join(
        'length_split', split_name, 'tasks_train.txt'), 'w')
    output_file_test = open(os.path.join(
        'length_split', split_name, 'tasks_test.txt'), 'w')

    n_included = 0
    n_excluded = 0

    actual_included = set()
    actual_excluded = set()

    for line in input_file_all:
        line_stripped = line.strip()
        input_sequence, output_sequence = line_stripped.split('\t')

        input_sequence_length = len(input_sequence.split())
        output_sequence_length = len(output_sequence.split())

        if split_on == 'input':
            sequence_length_to_split_on = input_sequence_length
        elif split_on == 'output':
            sequence_length_to_split_on = output_sequence_length
        else:
            print("Incorrect argument")

        if sequence_length_to_split_on in included:
            output_file_train.write(line)
            n_included += 1
            actual_included.add(sequence_length_to_split_on)
        else:
            output_file_test.write(line)
            n_excluded += 1
            actual_excluded.add(sequence_length_to_split_on)

    total_number_of_tasks = n_included + n_excluded

    print("Included {} lengths in training set ({:.02f}%): {}".format(split_on, 100.0 * n_included / total_number_of_tasks, sorted(actual_included)))
    print("Included {} lengths in test set ({:.02f}%): {}".format(split_on, 100.0 * n_excluded / total_number_of_tasks, sorted(actual_excluded)))
    print("Results are in '{}'".format(os.path.join('length_split', split_name)))
    print()

def sample_equally_from_file(split_on, input_file_name, output_file_name):
    """
    Creates a train/test split on basis of either input or output length of
    the examples
    Args:
        split_on (str): Split on either 'input' or 'output'
        included (list): A list of all the lengths that should be included in the
        split_name (str): Name of the directory in which the train/test files will be stored
        training set. The rest are in the test set
    """

    input_file = open(input_file_name, 'r')
    output_file = open(output_file_name, 'w')
    dev_file = open(output_file_name + 'dev', 'w')

    lines = input_file

    length_dict = {}
    num_lines = 0

    for line in lines:
        num_lines += 1
        line_stripped = line.strip()
        input_sequence, output_sequence = line_stripped.split('\t')

        input_sequence_length = len(input_sequence.split())
        output_sequence_length = len(output_sequence.split())

        if split_on == 'input':
            sequence_length_to_split_on = input_sequence_length
        elif split_on == 'output':
            sequence_length_to_split_on = output_sequence_length
        else:
            print("Incorrect argument")

        if sequence_length_to_split_on not in length_dict:
            length_dict[sequence_length_to_split_on] = [line]
        else:
            length_dict[sequence_length_to_split_on].append(line)


    for _ in range(num_lines):
        sampled_length = random.choice(list(length_dict.keys()))
        sampled_line = random.choice(length_dict[sampled_length])
        if random.random() > 0.1:
            output_file.write(sampled_line)
        else:
            dev_file.write(sampled_line)

def sample_equally_from_file_all_data(split_on, input_file_name, output_file_name):
    """
    Creates a train/test split on basis of either input or output length of
    the examples
    Args:
        split_on (str): Split on either 'input' or 'output'
        included (list): A list of all the lengths that should be included in the
        split_name (str): Name of the directory in which the train/test files will be stored
        training set. The rest are in the test set
    """

    input_file = open(input_file_name, 'r')
    output_file = open(output_file_name, 'w')

    lines = input_file

    length_dict = {}
    num_lines = 0

    for line in lines:
        num_lines += 1
        line_stripped = line.strip()
        input_sequence, output_sequence = line_stripped.split('\t')

        input_sequence_length = len(input_sequence.split())
        output_sequence_length = len(output_sequence.split())

        if split_on == 'input':
            sequence_length_to_split_on = input_sequence_length
        elif split_on == 'output':
            sequence_length_to_split_on = output_sequence_length
        else:
            print("Incorrect argument")

        if sequence_length_to_split_on not in length_dict:
            length_dict[sequence_length_to_split_on] = [line]
        else:
            length_dict[sequence_length_to_split_on].append(line)

    all_lengths = list(length_dict.keys())
    max_length_samples = max([len(length_dict[length]) for length in all_lengths])
    sampled_lines = []
    for length in all_lengths:
        for i in range(max_length_samples):
            sampled_line = length_dict[length][i % len(length_dict[length])]
            sampled_lines.append(sampled_line)

    random.shuffle(sampled_lines)

    for sampled_line in sampled_lines:
        output_file.write(sampled_line)



if __name__ == '__main__':
    show_statistics()

    # create_split(split_on='output', included=range(23), split_name='experiment2a_output_short_to_long')
    # create_split(split_on='output', included=range(7, 49), split_name='experiment2b_output_long_to_short')
    # create_split(split_on='output', included=list(range(1, 9)) + list(range(11,19)) + list(range(21, 29)), split_name='experiment2c_output_interleaved_short_to_long')
    #
    # create_split(split_on='input', included=range(9), split_name='experiment2d_input_short_to_long')
    # create_split(split_on='input', included=range(7, 10), split_name='experiment2e_input_long_to_short')
    # create_split(split_on='input', included=[1, 2, 3, 6, 7, 8], split_name='experiment2f_input_interleaved_short_to_long')

    sample_equally_from_file('output', 'length_split/experiment5/tasks_train.txt', 'length_split/experiment5/output.txt')


    #sample_equally_from_file('output', 'length_split/experiment4b_output_long_to_short_equally_distributed/tasks_train.txt', 'length_split/experiment4b_output_long_to_short_equally_distributed/output_all_shuffled.txt')
    create_split(split_on='output', included=range(22), split_name='22')