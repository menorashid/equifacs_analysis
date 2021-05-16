import os
import numpy as np
import glob
from helpers import util
import pandas as pd
import datetime
import re


def clean_data(data_dict, remove_lr = True):
    for k in data_dict.keys():
        
        au_list = data_dict[k][0]
        all_other_lists = data_dict[k][1:]
        num_other_lists = len(all_other_lists)
        # time_list = data_dict[k][1]

        idx_au = []
        au_keep = []
        all_other_list_keep = [[] for idx in range(num_other_lists)]

        strs_exclude = ['lip','uncodable','uppe','nan','scratching']

        for idx_au, au in enumerate(au_list):
            # print idx_au, au
            # print 'au',idx_au, k, au, all_other_lists[0][idx_au], all_other_lists[1][idx_au]
            try:
                au = au.lower().split()[0].strip()
            except:
                print 'au',idx_au, k, au, all_other_lists[0][idx_au], all_other_lists[1][idx_au]
                raw_input()
            
            problem = False
            for str_exclude in strs_exclude:
                if str_exclude in au:
                    problem=True
                    break
            if problem:
            # if 'lip' in au or 'uncodable' in au or 'uppe' in au:
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

def get_file_pain_status(file_names):
    no_pain_strs = re.compile('|'.join(['postBL*', 'preBL*', 'preind*']))
    pain_strs = re.compile('|'.join(['postind*']))
    pain_status = []
    for file_name in file_names:
        if pain_strs.search(file_name):
            pain_status.append(1)
        elif no_pain_strs.search(file_name):
            pain_status.append(0)
        else:
            pain_status.append(-1)
    pain_status = np.array(pain_status)
    return pain_status

def read_in_data_eop(file_name, decoder_file):
    df = pd.read_csv(file_name, header = None, encoding = None)
    arr = df.values
    arr[0, 0] = arr[0, 0].strip('\xef\xbb\xbf')
    header_rows = arr[:, 1]
    header_rows = [idx for idx, val in enumerate(header_rows) \
                    if type(val)!=float and val.startswith('#file')]
    file_names = arr[header_rows, 1]
    file_names = np.array(['_'.join(re.split(' |\.|_', os.path.split(file_name)[1])[:2]) \
                    for file_name in file_names])
    
    code_name_mapping = pd.read_csv(decoder_file, header=None).values
    code_name_mapping[0, 0] = code_name_mapping[0, 0].strip('\xef\xbb\xbf')
    code_name_mapping = np.concatenate([code_name_mapping, get_file_pain_status(code_name_mapping[:, 0])[:,np.newaxis]], axis=-1)

    indices = np.where(np.transpose(code_name_mapping[:, 1:2]) == file_names[:, np.newaxis])[1]
    pain_status = code_name_mapping[indices, -1]
    rel_annos = []

    header_rows = header_rows + [len(df)]
    for idx_row, header_row in enumerate(header_rows[: -1]):
        if idx_row < (len(header_rows) - 1):
            end_idx = header_rows[idx_row + 1] - 1
        else:
            end_idx = header_rows[idx_row + 1]
        rel_anno_curr = df[header_row + 1:end_idx]
        rel_anno_curr = rel_anno_curr.replace(np.nan, 'EMPTY', regex=True)
        rel_annos.append(rel_anno_curr)
        
    for idx, is_pain in enumerate(pain_status):
        print file_names[idx], is_pain
    # for col in range(rel_annos[0].shape[1]):
    #     print col, np.dtype(rel_annos[0].iloc[:, col].values)
    # assert (len(rel_annos) == len(file_names))
    # all_aus = np.concatenate([rel_anno.iloc[:, -1].values for rel_anno in rel_annos])

    # print np.unique(all_aus)
    
def main():
    file_name = '../data/eop_dataset/equifacs_lps.csv'
    decoder_file = '../data/eop_dataset/decoded_names.csv'
    read_in_data_eop(file_name, decoder_file)
    # get_file_pain_status(decoder_file)

if __name__=='__main__':
    main()