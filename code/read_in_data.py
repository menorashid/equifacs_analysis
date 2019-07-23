import os
import numpy as np
import glob
from helpers import util
import pandas as pd

def find_next_list(df, keys, idx_key, key_str):
    for idx_next in range(idx_key+1,len(keys)):
        k_plus_one = keys[idx_next]
        time_list = [str(val) for val in df[k_plus_one].tolist()]
        if time_list[0].lower().startswith(key_str):
            return time_list

    return None

def clean_start_end_times(time_list):
    
    for idx_val,val in enumerate(time_list):
        val = val[val.rindex(':')+1:]
        val = float(val)
        time_list[idx_val] = val
        
    return time_list

def read_start_stop_anno_file(file_name):
    df = pd.read_csv(file_name)
    keys = [k for k in df.keys()]
    data_dict = {}

    for idx_key, k in enumerate(keys):
        if 'FILM' in k:
            facs_list = [str(val) for val in df[k].tolist()]
            duration_list = find_next_list(df, keys, idx_key,'duration')
            start_time_list = find_next_list(df, keys, idx_key,'start')
            end_time_list = find_next_list(df, keys, idx_key,'end')
            vid_number = int(k.strip().split()[-1])
            if 'nan' in facs_list:
                len_time  = facs_list.index('nan')
            else:
                len_time = len(facs_list)
            

            facs_list = facs_list[1:len_time]
            duration_list = [float(val.replace(',','.')) for val in duration_list[1:len_time]]
            start_time_list = clean_start_end_times(start_time_list[1:len_time])
            end_time_list = clean_start_end_times(end_time_list[1:len_time])

            data_dict[vid_number] = [facs_list, duration_list, start_time_list, end_time_list]
    
    return data_dict

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
        all_other_lists = data_dict[k][1:]
        num_other_lists = len(all_other_lists)
        # time_list = data_dict[k][1]

        idx_au = []
        au_keep = []
        all_other_list_keep = [[] for idx in range(num_other_lists)]

        for idx_au, au in enumerate(au_list):
            
            au = au.lower().split()[0].strip()
            
            if 'lip' in au or 'uncodable' in au or 'uppe' in au:
                continue
            
            if remove_lr:
                if au.endswith('l') or au.endswith('r'):
                    au = au[:-1]
                    au = au.strip()

            au_keep.append(au)
            for idx_list_curr in range(num_other_lists):

                all_other_list_keep[idx_list_curr].append(all_other_lists[idx_list_curr][idx_au])


        data_dict[k][0] = au_keep
        for idx_list_curr,list_curr in enumerate(all_other_list_keep):
            data_dict[k][idx_list_curr+1] = list_curr

    return data_dict

def get_all_aus(data_dict):
    all_aus = []
    for k in data_dict.keys():
        all_aus += data_dict[k][0]
    all_aus = list(set(all_aus))
    all_aus.sort()
    return all_aus



def main():
    # rename_files()
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    print data_dict[5]
    data_dict = clean_data(data_dict)
    print data_dict[5]



if __name__=='__main__':
    main()