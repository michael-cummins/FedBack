import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from natsort import natsorted
from typing import List

sns.set_theme()

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
    os.makedirs(save_dir, exist_ok=True)
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

def plot_rate(rate: int, exp: str):
    data_dir = f'figure_data/{exp}/FedBack/'
    save_dir = f'images/{exp}/fedback_rates/'
    rate_file = f'rates_{rate}.npy'
    rates = np.load(data_dir+rate_file)
    plt.plot(range(len(rates)), rates, label='FedBack', color='orange')
    plt.plot(range(len(rates)), (rate/100)*np.ones(rates.shape), linestyle='dashed', color='black', label='Reference')
    plt.xlabel('Round')
    plt.ylabel('Communication Load')
    plt.title('Communication Load per Round')
    plt.ylim([0,1.1])
    plt.legend()
    # if rate <= 20: plt.ylim([0,0.3])
    # elif rate <= 60: plt.ylim([0,0.7])
    plt.show()
    plt.savefig(save_dir+f'rate_curve_{rate}')
    plt.cla()
    plt.clf()

def total_comm(dirs: List, rate: int, thresh: float, exp:bool):
    save_dir = f'images/{exp}/joint_acc/'
    data_dir = f'figure_data/'
    acc_file = f'accs_{rate}'
    rate_file = f'rates_{rate}.npy'
    load_file = f'loads_{rate}.npy'
    back_rate = None
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
                except:
                    comm_rate = rate 
                    break
                if dir == 'FedBack/':
                    rates = np.load(data_dir+exp+'/'+dir+rate_file)
                    cummulative_comm = np.sum(rates[:above_thresh+1])*100
                    back_rate = cummulative_comm
                    comm_rate = np.load(data_dir+exp+'/'+dir+load_file)[0]*100
                else: 
                    cummulative_comm = rate*(above_thresh+1)
                    comm_rate = rate 
                break
        if cummulative_comm != 'N/A': cummulative_comm = int(cummulative_comm)
        print(f'Total communication for {exp} {dir[:-1]} to achieve {int(thresh*100)}% accuracy at rate {comm_rate:.2f} = {cummulative_comm}')        
    
    return back_rate

if __name__ == '__main__':
    sns.set_theme()
    
    print('CIFAR')
    exp = 'cifar'
    dirs = ['FedADMM/', 'FedBack/', 'FedAVG/', 'FedProx/']
    rates = [5,10,15,20,30,40,50,60,70,80]
    back_rates= []
    for rate in rates:
        multiple_acc(dirs, rate=rate, save_file=f'rate_{rate}', exp=exp)
        comm_rate = total_comm(dirs=dirs, rate=rate, thresh=0.78, exp=exp)
        if rate is not None: back_rates.append(comm_rate)
        # plot_rate(rate=rate, exp=exp)
        print('\n')
    
    plt.plot(range(len(back_rates)), back_rates)
    plt.xlabel('Communication load (%)')
    plt.ylabel('Number of events')
    plt.title('Target accuracy = 78%')
    plt.show()
    plt.savefig('images/cifar/loads_event.png')
    plt.cla()
    plt.clf()

    # print('MNIST')
    # exp = 'mnist'
    # dirs = ['FedADMM/', 'FedBack/', 'FedAVG/', 'FedProx/']
    # rates = [5,10,15,20,30,40,50,60,70,80,90]
    # for rate in rates:
    #     multiple_acc(dirs, rate=rate, save_file=f'rate_{rate}', exp=exp)
    #     plot_rate(rate=rate, exp=exp)
    #     total_comm(dirs=dirs, rate=rate, thresh=0.90, exp=exp)
    #     print('\n')
        
    
    
    
