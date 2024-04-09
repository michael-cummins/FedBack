import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from natsort import natsorted
from typing import List

def list_files_in_folder(folder_path):
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        print("Error: Provided path is not a directory.")
        return []
    # Get a list of all files in the directory
    file_list = os.listdir(folder_path)
    # Sort the list of file names alphabetically
    file_list = natsorted(file_list)
    # Return the sorted list of file names
    return file_list

def multiple_acc(dirs: List, rate: int, save_file: str):
    save_dir = 'images/joint_acc/'
    data_dir = 'figure_data/'
    file = f'accs_{rate}'
    vals = []

    for dir in dirs:
        files = list_files_in_folder(data_dir+dir)
        for f in files:
            if f[:len(file)] == file: 
                val = np.load(data_dir+dir+f)
                vals.append(val[0])
                break
    
    for val, dir in zip(vals, dirs):
        plt.plot(range(len(val)),val,label=dir[:-1])
    plt.xlabel('Round')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Rate = {rate}')
    plt.legend() 
    plt.show()
    plt.ylim([0,0.9])
    plt.xlim([0,500])
    plt.savefig(save_dir+save_file)
    plt.cla()
    plt.clf()

def load_acc(dirs: List, save_file: str, max_load: float):
    data_dir = 'figure_data/'    
    save_dir = 'images/load_acc/'    
    # get all the accuracies per delta
    for i, dir in enumerate(dirs):
        loads = []
        accs = []
        files = list_files_in_folder(data_dir+dir)
        for f in files:
            if f[:4] == 'load':
                rate = np.load(data_dir+dir+f)
                if rate[0] <= max_load: loads.append(rate[0])
            if f[:3] == 'acc':
                vals = np.load(data_dir+dir+f)
                accs.append(vals[0][-1])
        plt.plot(loads, accs, '-x', label=dir[:-1])
    plt.xlabel('Communication Load')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Load/Accuracy tradeoff')
    plt.legend() 
    plt.show()
    plt.savefig(save_dir+save_file)
    plt.cla()
    plt.clf()

if __name__ == '__main__':
    sns.set_theme()
    dirs = ['FedADMM/', 'FedBack/', 'FedAVG/', 'FedProx/']
    rates = [6,9,12,20,30,40,50,60]
    # for rate in rates:
    #     multiple_acc(dirs, rate=rate, save_file=f'rate_{rate}')
    
    load_acc(dirs=dirs[:], save_file='load_acc.png', max_load=120)

