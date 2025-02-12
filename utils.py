import json
from multiprocessing import Pool, cpu_count
import yaml
import os
import functools
import datetime
from re import search
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import csv
import shutil
import pandas as pd

datetime_format = '%Y-%m-%d %H:%M:%S'


def split_list(input_list, num_chunks):
        
        avg_chunk_size = len(input_list) // num_chunks
        remainder = len(input_list) % num_chunks

        chunks = []
        start = 0
        for i in range(num_chunks):
            end = start + avg_chunk_size + (1 if i < remainder else 0)
            chunks.append(input_list[start:end])
            start = end

        return chunks


def process_batch(settings):

    data_ = {}

    for setting in settings:

        res_ = single_log_short_analysis(setting)

        if res_[0] in data_:
            if res_[1] in data_[res_[0]]:
                
                original_ = copy_dict_except_member(data_[res_[0]][res_[1]], 'file')
                new_ = copy_dict_except_member(res_[2], 'file')

                if original_ != new_:

                    if new_['end_time'] > original_['end_time']:
                        import pdb; pdb.set_trace()
                
            else:
                data_[res_[0]][res_[1]] = res_[2]

        else:
            data_[res_[0]] = {res_[1]: res_[2]}

    return data_


def single_log_short_analysis(file):

    datetime_format = '%Y%m%d %H:%M:%S'

    info_start = find_string_in_file(file, "starting...").split(" ")
    starting_time = datetime.datetime.strptime(info_start[1] + " " + info_start[2], datetime_format)
    shot_number = info_start[4][1:-1]

    info_nodes = find_string_in_file(file, "SAL_JOBID").split("=")[-1].split(".")[0]
    job_id = info_nodes[1:]

    info_nodes = find_string_in_file(file, "SAL_NUM_NODES").split("=")[-1]
    nodes_number = int(info_nodes[1:-1])

    info_nodes = find_string_in_file(file, "SAL_MAX_CORES_PER_NODE").split("=")[-1]
    cores_per_node = int(info_nodes[1:-1])

    info_nodes = find_string_in_file(file, "SAL_MPITASKS_PER_NODE").split("=")[-1]
    mpi_task_per_node = int(info_nodes[1:-1])

    info_nodes_list = find_string_in_file(file, "SAL_NODES_USAGE").split("SAL_NODES_USAGE=")[1][1:-1].split(" ")
    nodes_list = {}
    for node in info_nodes_list:
        nodes_list[node.split(".")[0]] = {}

    if nodes_number != len(nodes_list):
        import pdb; pdb.set_trace()

    info_run = find_string_in_file(file, "'-p'")
    if info_run == []:
        
        info_run = find_string_in_file(file, "-p ")

        if type(info_run) is not list:
            info_run = [info_run]

        info_run = [''] + info_run[0].split("-p ")[1].split(" -f ")[0].split(" ")
        
        for ii in [1, 3, 5]:
            info_run[ii] = "'%s'" %info_run[ii]
    else:
        info_run = info_run.split('(')[1].split(')')[0].split(", ")
    
    mpitasks_ = info_run[1].split("'")[1].split(":")
    mpitasks = 1
    for ii in range(len(mpitasks_)):
        mpitasks *= int(mpitasks_[ii])
    gpus_ = info_run[3].split("'")[1].split(":")
    gpus = 1
    for ii in range(len(gpus_)):
        gpus *= int(gpus_[ii])
    ompthreads_ = info_run[5].split("'")[1].split(":")
    ompthreads = 1
    for ii in range(len(ompthreads_)):
        ompthreads *= int(ompthreads_[ii])

    if check_string_in_file(file, " completed."):

        info_end = find_string_in_file(file, " completed.").split(" ")
        finish_time = datetime.datetime.strptime(info_end[1] + " " + info_end[2], datetime_format)
        duration = float(info_end[5][1:-2])

        timesteps = int(find_string_in_file(file, "number of time step").split(". ")[1])
        cells = float(find_string_in_file(file, "FORW Global Performances").split("cells: ")[1].split(" cells/sec:")[0])

        if find_string_in_file(file, "Normalize angle gathers field with multiplier") == []:
            gathers = 0
        else:
            gathers = 1

        if int(find_string_in_file(file, "wave front tracking flag ...........").split(" ")[-1].split("\n")[0]) == 1:
            wave_front_tracking = 1
        else:
            wave_front_tracking = 0

        if find_string_in_file(file, "hilbert imaging condition flag......").split(" ")[-1].split("\n")[0] == "SINGLE_HILBERT_FILTER":
            hilbert_filter = 1
        else:
            hilbert_filter = 0

        if find_string_in_file(file, "kernel type ........................").split(" ")[-1].split("\n")[0] == "ISOTROPIC_ACOUSTIC":
            isotropic_kernel = 1
        else:
            isotropic_kernel = 0
            
        return [job_id, shot_number, 
                {"number_of_cells": cells, "number_of_timestep": timesteps, "nodes_number": nodes_number, 
                "gathers": gathers, "wave_front_tracking": wave_front_tracking, "hilbert_filter": hilbert_filter, "isotropic_kernel": isotropic_kernel,
                "max_cores_per_node": cores_per_node, "mpitasks": mpitasks, "gpus": gpus, "ompthreads": ompthreads, "mpi_tasks_per_node": mpi_task_per_node, 
                "start_time": str(starting_time), "end_time": str(finish_time), "duration": duration, 
                "nodes_list": nodes_list, "success": True, "file": file}]
        
    else:

        if check_string_in_file(file, "] failed: "):
            info_fail = find_string_in_file(file, "] failed: ").split(" ")[1:3]
            finish_time = datetime.datetime.strptime(info_fail[0] + " " + info_fail[1], datetime_format)

        else:
            finish_time = find_last_timestep(file)

        return [job_id, shot_number, 
                {"nodes_number": nodes_number, "max_cores_per_node": cores_per_node, "mpitasks": mpitasks, "gpus": gpus, "ompthreads": ompthreads,
                "mpi_tasks_per_node": mpi_task_per_node, "start_time": str(starting_time), "end_time": str(finish_time), "duration": np.nan, "nodes_list": nodes_list, "success": False,
                "file": file}]
    

def find_string_in_file(file_path, search_string):

    lines = []

    with open(file_path, 'r') as file:
        for line in file:
            if search_string in line:
                lines.append(line.rstrip())

    if len(lines) == 1:
        return lines[0]
    else:
        return lines
    

def check_string_in_file(file_path, target_string):
    with open(file_path, 'r') as file:
        for line in file:
            if target_string in line:
                return True
    return False


def find_last_timestep(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    date_pattern = r'\d{4}\d{2}\d{2} \d{2}:\d{2}:\d{2}'
    datetime_format = '%Y%m%d %H:%M:%S'

    for line in reversed(lines):            
        match = search(date_pattern, line)
        if bool(match):

            first_idx = match.span()[0]
            last_idx = match.span()[1]

            return datetime.datetime.strptime(match.string[first_idx:last_idx], datetime_format)
        

def copy_dict_except_member(original_dict, member_to_exclude):
    new_dict = {}
    for key, value in original_dict.items():
        if key != member_to_exclude:
            new_dict[key] = value
    return new_dict