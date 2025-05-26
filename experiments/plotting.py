import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from natsort import natsorted
from typing import List


def list_files_in_folder(folder_path):
    """
    List all files in a given folder, sorted alphanumerically.
    - used for getting list of .npy files in subdirectories of experiments/figure_data/
    """
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        print(f"Provided path <{folder_path}>is not a directory.")
        return []
    # Get a list of all files in the directory
    file_list = os.listdir(folder_path)
    # Sort the list of file names alphabetically
    file_list = natsorted(file_list)
    # Return the sorted list of file names
    return file_list

def multiple_acc(dirs: List, rate: int, save_file: str, exp:bool):
    """
    Plots training curves for each algorithm, with one curve per rate.
    Print a warning if no data is found for the given rate.
    """

    save_dir = f'images/{exp}/joint_acc/'
    os.makedirs(save_dir, exist_ok=True)
    data_dir = f'figure_data/'
    file = f'accs_{rate}.npy'
    vals = []

    valid_dirs = [d for d in dirs if os.path.exists(data_dir + exp + '/' + d + file)]

    if valid_dirs:  
        for dir in valid_dirs:
            files = list_files_in_folder(data_dir+exp+'/'+dir)
            if files:
                for f in files:
                    if f[:len(file)] == file: 
                        val = np.load(data_dir+exp+'/'+dir+f)
                        vals.append(val[0])
                        break
        
        for val, dir in zip(vals, valid_dirs):
            plt.plot(range(len(val)),val,label=dir[:-1])
        plt.xlabel('Round')
        plt.ylabel('Validation Accuracy')
        plt.title(f'Rate = {rate}')
        plt.legend() 
        plt.show()
        plt.ylim([0,1])
        plt.savefig(save_dir+save_file)
        plt.cla()
        plt.clf()
    else:
        print(f'No data found for {exp} at rate {rate}.\nPlease run the experiment in main.py.')
        

def plot_rate(rate: int, exp: str):
    """
    Plots the participation rate per round for FedBack algorithm.
    """

    data_dir = f'figure_data/{exp}/FedBack/'
    save_dir = f'images/{exp}/fedback_rates/'
    os.makedirs(save_dir, exist_ok=True)
    rate_file = f'rates_{rate}.npy'

    # Plot the rate graph if it exists
    if os.path.exists(data_dir + rate_file):
        rates = np.load(data_dir+rate_file)
        plt.plot(range(len(rates)), rates, label='FedBack', color='orange')
        plt.plot(range(len(rates)), (rate/100)*np.ones(rates.shape), linestyle='dashed', color='black', label='Reference')
        plt.xlabel('Round')
        plt.ylabel('Participation Rate')
        plt.title('Participation Rate per Round')
        plt.ylim([0,1.1])
        plt.legend()
        plt.show()
        plt.savefig(save_dir+f'rate_curve_{rate}')
        plt.cla()
        plt.clf()
    else: 
        print(f'No rate file found for {data_dir + rate_file}.\nPlease run FedBack experiment in main.py.')

def total_comm(dirs: List, rate: int, thresh: float, exp:bool):
    """
    Function that prints the number of participation events for algorithm to achieve 
    accuracy "thresh" at participation rate "rate" for each algorithm in "dirs".

    - exp: str, the experiment name (e.g., 'cifar' or 'mnist')
    """

    data_dir = f'figure_data/'
    acc_file = f'accs_{rate}.npy'
    rate_file = f'rates_{rate}.npy'
    load_file = f'loads_{rate}.npy'
    
    valid_dirs = [d for d in dirs if os.path.exists(data_dir + exp + '/' + d + acc_file)]

    if valid_dirs:
        for dir in valid_dirs:
            files = list_files_in_folder(data_dir+exp+'/'+dir)
            cummulative_comm = 'N/A'
            for f in files:
                if f[:len(acc_file)] == acc_file: 
                    val = np.load(data_dir+exp+'/'+dir+f)
                    val = val[0]
                    try:
                        above_thresh = np.where(val > thresh)[0][0]
                    except:
                        comm_rate = rate 
                        break
                    if dir == 'FedBack/':
                        rates = np.load(data_dir+exp+'/'+dir+rate_file)
                        cummulative_comm = np.sum(rates[:above_thresh+1])*100
                        comm_rate = np.load(data_dir+exp+'/'+dir+load_file)[0]*100
                    else: 
                        cummulative_comm = rate*(above_thresh+1)
                        comm_rate = rate 
                    break
            if cummulative_comm != 'N/A': cummulative_comm = int(cummulative_comm)
            print(f'Total communication for {exp} {dir[:-1]} to achieve {int(thresh*100)}% accuracy at rate {comm_rate:.2f} = {cummulative_comm}')        
    else: 
        print(f'No data found for {exp} at rate {rate}.\nPlease run the experiment in main.py.')


if __name__ == '__main__':
    sns.set_theme()

    MNIST_tHRESHOLD = 0.90
    CIFAR_tHRESHOLD = 0.78
    dirs = ['FedADMM/', 'FedBack/', 'FedAVG/', 'FedProx/']

    # Visualise resukts for CIFAR-10
    print('CIFAR')
    exp = 'cifar'
    rates: List[int] = [5,10,15,20,30,40,50,60]
    for rate in rates:
        multiple_acc(dirs, rate=rate, save_file=f'rate_{rate}', exp=exp)
        plot_rate(rate=rate, exp=exp)
        total_comm(dirs=dirs, rate=rate, thresh=CIFAR_tHRESHOLD, exp=exp)
        print('\n')

    # Visualuse results for MNIST
    print('MNIST')
    exp = 'mnist'
    rates = [5,10,15,20,30,40,50,60]
    for rate in rates:
        multiple_acc(dirs, rate=rate, save_file=f'rate_{rate}', exp=exp)
        plot_rate(rate=rate, exp=exp)
        total_comm(dirs=dirs, rate=rate, thresh=MNIST_tHRESHOLD, exp=exp)
        print('\n')




