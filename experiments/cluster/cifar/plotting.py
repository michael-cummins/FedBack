import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from natsort import natsorted
from typing import List
import tikzplotlib

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

def multiple_acc(dirs: List, rate: int, save_file: str, exp:bool):
    save_dir = f'images/{exp}/joint_acc/'
    data_dir = f'figure_data/'
    file = f'accs_{rate}'
    vals = []

    for dir in dirs:
        files = list_files_in_folder(data_dir+exp+'/'+dir)

        for f in files:
            if f[:len(file)] == file: 
                val = np.load(data_dir+exp+'/'+dir+f)
                vals.append(val[0])
                break
    
    for val, dir in zip(vals, dirs):
        plt.plot(range(len(val)),val,label=dir[:-1])
    plt.xlabel('Round')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Rate = {rate}')
    plt.legend() 
    plt.show()
    plt.ylim([0,1])
    # plt.xlim([0,len(val)])
    plt.savefig(save_dir+save_file)
    tikzplotlib.save(save_dir+save_file+'.tex')
    plt.cla()
    plt.clf()

def load_acc(dirs: List, save_file: str, max_load: float):
    data_dir = 'figure_data/'    
    save_dir = 'images/load_acc/'    
    # get all the accuracies per delta
    for i, dir in enumerate(dirs):
        loads = []
        accs = []
        print(data_dir+exp+'/'+dir)
        files = list_files_in_folder(data_dir+exp+'/'+dir)
        for f in files:
            if f[:4] == 'load':
                rate = np.load(data_dir+exp+'/'+dir+f)
                if rate[0] <= max_load: loads.append(rate[0])
            if f[:3] == 'acc':
                vals = np.load(data_dir+exp+'/'+dir+f)
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

def total_comm(dirs: List, rate: int, thresh: float, exp:bool):
    save_dir = f'images/{exp}/joint_acc/'
    data_dir = f'figure_data/'
    acc_file = f'accs_{rate}'
    rate_file = f'rates_{rate}.npy'
    vals = []

    for dir in dirs:
        files = list_files_in_folder(data_dir+exp+'/'+dir)
        cummulative_comm = 'N/A'
        for f in files:
            if f[:len(acc_file)] == acc_file: 
                val = np.load(data_dir+exp+'/'+dir+f)
                val = val[0]
                try:
                    above_thresh = np.where(val > thresh)[0][0]
                except: break
                if dir == 'FedBack/':
                    rates = np.load(data_dir+exp+'/'+dir+rate_file)
                    cummulative_comm = np.sum(rates[:above_thresh+1])*100
                else: cummulative_comm = rate*(above_thresh+1)
                break
        if cummulative_comm != 'N/A': cummulative_comm = int(cummulative_comm)
        print(f'Total communication for {exp} {dir[:-1]} to achieve {thresh*100}% accuracy at rate {rate} = {cummulative_comm}')
            

if __name__ == '__main__':
    sns.set_theme()
    exp = 'mnist'
    dirs = ['FedADMM/', 'FedBack/', 'FedAVG/', 'FedProx/']
    rates = [5,10,15,20,30,40,50,60,70,80,90,100]
    for rate in rates:
        multiple_acc(dirs, rate=rate, save_file=f'rate_{rate}', exp=exp)
        total_comm(dirs=dirs, rate=rate, thresh=0.85, exp=exp)
        print('\n')
    
    
    
