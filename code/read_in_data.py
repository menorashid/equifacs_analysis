import os
import numpy as np
import glob
from helpers import util
import pandas as pd

def rename_files():
    dir_data = '../data'
    files = glob.glob(os.path.join(dir_data,'*'))   
    for file_curr in files:
        print file_curr
        if ' ' in file_curr:
            out_file = file_curr.replace(' ','_')
            print out_file
            raw_input()
            os.rename(file_curr,out_file)

def read_anno_file(file_name):
    df = pd.read_csv(file_name)
    keys = [k for k in df.keys()]
    data_dict = {}

    for idx_key, k in enumerate(keys):
        if 'FILM' in k:
            facs_list = [str(val) for val in df[k].tolist()]
            for idx_next in range(idx_key+1,len(keys)):
                k_plus_one = keys[idx_next]
                time_list = [str(val) for val in df[k_plus_one].tolist()]
                if time_list[0].startswith('Time'):
                    break
            
            vid_number = int(k.strip().split()[-1])
            len_time  = facs_list.index('nan')
            time_list = [float(val.replace(',','.')) for val in time_list[1:len_time]]
            facs_list = facs_list[1:len_time]
            data_dict[vid_number] = [facs_list, time_list]
    
    return data_dict

def clean_data(data_dict, remove_lr = True):
    for k in data_dict.keys():
        
        au_list = data_dict[k][0]
        time_list = data_dict[k][1]

        idx_au = []
        au_keep = []
        time_keep = []

        for idx_au, au in enumerate(au_list):
            
            au = au.lower().strip()
            
            if 'lip' in au:
                continue
            
            if remove_lr:
                if au.endswith('l') or au.endswith('r'):
                    au = au[:-1]
                    au = au.strip()

            au_keep.append(au)
            time_keep.append(time_list[idx_au])

        data_dict[k][0] = au_keep
        data_dict[k][1] = time_keep

    return data_dict

def get_all_aus(data_dict):
    all_aus = []
    for k in data_dict.keys():
        all_aus += data_dict[k][0]
    all_aus = list(set(all_aus))
    all_aus.sort()
    return all_aus


def main():
    pass

if __name__=='__main__':
    main()