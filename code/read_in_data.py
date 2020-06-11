import os
import numpy as np
import glob
from helpers import util
import pandas as pd
import datetime
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

def get_all_aus(data_dict):
    all_aus = []
    for k in data_dict.keys():
        all_aus += data_dict[k][0]
    all_aus = list(set(all_aus))
    all_aus.sort()
    return all_aus

def read_clinical_file(file_name):
    header_rows = [1,61,140,220,325,378,557,638,683,736,766,802,879,944,981,1062,1134,1225,1275,1335,1363,1452]
    header_rows = [val-1 for val in header_rows]
    df = pd.read_csv(file_name,header = None)
    rel_annos = []
    for idx_row,header_row in enumerate(header_rows[:-1]):
        rel_anno_curr = df[header_row:header_rows[idx_row+1]-1]
        rel_annos.append(rel_anno_curr.reset_index())
        
    data_dict = {}
    for rel_anno in rel_annos:
        # print rel_anno[0][0]
        film_name = int(rel_anno[0][0].strip().split()[1])
        aus = rel_anno[4:][5]
        aus = [str(val) for val in aus.tolist()]
        time_lists = [rel_anno[4:][2],rel_anno[4:][3],rel_anno[4:][4]]
        new_lists = []
        
        for time_list in time_lists:
            new_list = []
            for val in time_list:
                val = util.convert_str_to_sec(val)
                # if val>60:
                #     print val
                new_list.append(val)
            new_lists.append(new_list)

        [start_time_list, end_time_list, duration] = new_lists
        data_dict[film_name] = [aus, duration, start_time_list, end_time_list]
        

    return data_dict

def read_clinical_pain():
    # file_name):
    file_name = '../data/clinical_cases_compiled_10_23_2019.csv'
    header_rows = [1,61,140,220,325,378,557,638,683,736,766,802,879,944,981,1062,1134,1225,1275,1335,1363,1452]
    header_rows = [val-1 for val in header_rows]
    df = pd.read_csv(file_name,header = 1451)
    
    annotators = ['Johan', 'Camilla', 'Alina ']

    pain_dict = {}
    for media in set(df['Media'].tolist()):
        
        condition = (df['Media']==media) & (~((df['Pain']=='Non-codeable') | (df['Pain']=='Non-codeable ')))
        vid_number = int(media.strip().split()[1])
        pain_list = []

        for annotator in annotators:
            list_curr = df[condition & (df['Annotator ']==annotator)]
            pain_list_curr = list_curr['Pain'].tolist()
            if len(pain_list_curr)==0:
                break
            elif len(pain_list_curr)==1:
                pain_list+=pain_list_curr
            else:
                durations = list_curr['Duration']
                durations = [util.convert_str_to_sec(val) for val in durations]
                max_idx = np.argmax(durations)
                pain_list.append(pain_list_curr[max_idx])
        
        if len(pain_list)==0:
            continue

        pain_dict[vid_number]=pain_list

    for k in pain_dict.keys():
        vals = pain_dict[k]
        numbers = []
        for val in vals:
            val = val.lower().strip()
            if val == 'no pain':
                numbers.append(0)
            elif val== 'mild pain' or val == 'moderate pain':
                numbers.append(1)
            elif val== 'severe pain':
                numbers.append(2)
            else:
                print val
                raw_input()
        # print numbers
        # print k, pain_dict[k],
        pain_dict[k]=numbers
        # print pain_dict[k]
    # pain_labels = np.array(pain_dict.keys())
    # pain_clinical = np.array([pain_dict[k] for k in pain_labels])
    # print pain_labels
    # print pain_clinical.shape
    # print pain_clinical
    # raw_input()

    return pain_dict
    # pain_labels, pain_clinical
    # 
        
                
def read_caps_anno_file():
    # file_name):
    file_name = '../data/caps_pain.csv'
    header_rows = [2,119,178,259,344]
    header_rows = [val-1 for val in header_rows]
    df = pd.read_csv(file_name,header = None)
    
    rel_annos = []
    for idx_row,header_row in enumerate(header_rows[:-1]):
        if idx_row<(len(header_rows)-2):
            end_idx = header_rows[idx_row+1]-2
            # print end_idx, 1
        else:
            end_idx = header_rows[idx_row+1]
            # print end_idx, 2

        rel_anno_curr = df[header_row:end_idx]
        rel_annos.append(rel_anno_curr.reset_index())

    # print len(rel_annos)

    data_dict = {}
    for idx_rel_anno,rel_anno in enumerate(rel_annos):
        film_name = idx_rel_anno
        aus = rel_anno[1:][5]
        aus = [str(val) for val in aus.tolist()]
        time_lists = [rel_anno[1:][2],rel_anno[1:][3],rel_anno[1:][4]]
        new_lists = []
        
        for time_list in time_lists:
            new_list = []
            for val in time_list:
                val = util.convert_str_to_sec(val)
                # if val>60:
                #     print val
                new_list.append(val)
            new_lists.append(new_list)

        [start_time_list, end_time_list, duration] = new_lists
        data_dict[film_name] = [aus, duration, start_time_list, end_time_list]

    # split lists
    key_curr = 13
    new_data_dict = {}
    for k in data_dict.keys():
        arr_curr = data_dict[k]
        arr_curr = [np.array(arr) for arr in arr_curr]

        bin_curr = arr_curr[2]>=30
        
        
        arr_old = [arr_arr_curr[np.logical_not(bin_curr)] for arr_arr_curr in arr_curr]
        arr_new = [arr_arr_curr[bin_curr] for arr_arr_curr in arr_curr]
        arr_new[2] = arr_new[2]-30
        arr_new[3] = arr_new[3]-30

        arr_old = [list(arr_curr) for arr_curr in arr_old]
        arr_new = [list(arr_curr) for arr_curr in arr_new]
        # for arr_curr in [arr_new,arr_old]:
        #     print key_curr
        #     for idx,arr_curr_curr in enumerate(arr_curr):
        #         print arr_curr_curr[:10],len(arr_curr_curr),
        #         if idx>0:
        #             print np.min(arr_curr_curr), np.max(arr_curr_curr)
        #         else:
        #             print '' 
        #     print '___'

        new_data_dict[key_curr] = arr_old
        new_data_dict[key_curr+1] = arr_new
        key_curr +=2

    # print new_data_dict.keys()
    # data_dict = {}
    # for k in [13,15,17,19,20,14]:
    #     data_dict[k] = new_data_dict[k]
    return new_data_dict




    # 
def read_in_data_stress(file_name, stress_type):
    
    data_dict, stress_anno, file_codes = read_in_data_stress_rough(file_name, stress_type)

    if stress_type == 'si':
        codes_to_keep = [0, 7, 18, 15, 8, 12, 19, 6, 4, 17, 3, 9, 14, 2, 16, 10, 13, 5, 11, 1]
    elif stress_type == 'tr':
        codes_to_keep = [20, 21, 26, 59, 42, 65, 0, 54, 18, 60, 8, 62, 19, 55, 4, 56, 3, 63, 14, 64, 16, 57, 13, 61, 11, 58, 30, 23, 22, 35, 25, 29, 27, 24, 37, 31, 53, 28, 52, 49, 32, 34, 40, 39, 36, 33, 41, 46, 48, 47, 45, 43, 44, 38, 50, 51]
    else:
        return data_dict, stress_anno, file_codes
    
    codes_to_keep = np.array(codes_to_keep)

    data_dict_new = {}
    for k in codes_to_keep:
        data_dict_new[k] = data_dict[k]

    bin_keep = np.in1d(file_codes, codes_to_keep)
    stress_anno = list(np.array(stress_anno)[bin_keep])
    file_codes = list(np.array(file_codes)[bin_keep])



    return data_dict, stress_anno, file_codes

def read_in_data_stress_rough(file_name, stress_type):
    # get_matches = False):
    # if get_matches:
    #     return read_in_data_stress_matches(file_name)

    df = pd.read_csv(file_name,header = None)
    arr = df.values
    
    arr = arr[1:,:]

    file_names = np.unique(arr[:,7])
    
    if stress_type == 'si':
        stress_strs = ['Baseline', 'Social isolation']
    elif stress_type == 'tr':
        stress_strs = ['Baseline', 'Transportation']
    else:
        stress_strs = ['Baseline', 'Social isolation','Transportation']
    stressers_needed = np.array(['Baseline'])

    # stress_strs = ['Baseline','Social isolation']
    # stressers_needed = np.array(['Baseline','Social isolation'])

    # stress_strs = ['Baseline','Transportation']
    # stressers_needed = np.array(['Baseline','Transportation'])

    horse_ids = np.unique(arr[:,0])
    horse_ids_keep = []
    for horse_id in horse_ids:
        
        

        stressers = np.unique(arr[arr[:,0]==horse_id,1])
        if np.sum(np.in1d(stressers_needed,stressers))==stressers_needed.size:
            # print horse_id, stressers, fn.shape
            horse_ids_keep.append(horse_id)

        # print horse_id, stressers
    horse_ids_keep = np.array(horse_ids_keep)
    # print horse_ids_keep
    # print horse_ids_keep.size
    # raw_input()

    # horse_names_si = np.array([])
    bin_rows = np.in1d(arr[:,0],horse_ids_keep)
    file_names = np.unique(arr[bin_rows,7])
    

    # horse_names_si = np.array([])
    # bin_rows = np.in1d(arr[:,0],horse_names_si)
    # file_names = np.unique(arr[bin_rows,7])
    # stress_strs = ['Baseline','Transportation']    

    
    
    stress_anno = []
    file_codes = []
    data_dict = {}
    # selected_ints = [0, 0, 0, 0, 0, 2, 1, 2, 1, 1, 2, 1, 1, 0, 0, 1, 1, 1, 2, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 2, 0, 1, 2, 2, 0, 1, 2, 0, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 1, 0, 2, 0, 0, 2, 0, 0, 0, 0, 1, 0, 2, 2, 2, 2, 2, 0, 1]
    # file_names_index = ['KV 1.eaf', 'KV 10.eaf', 'KV 11.eaf', 'KV 12.eaf', 'KV 13.eaf', 'KV 14.eaf', 'KV 15.eaf', 'KV 16.eaf', 'KV 17.eaf', 'KV 18.eaf', 'KV 19.eaf', 'KV 2.eaf', 'KV 20.eaf', 'KV 21.eaf', 'KV 22.eaf', 'KV 23.eaf', 'KV 24.eaf', 'KV 25.eaf', 'KV 26.eaf', 'KV 27.eaf', 'KV 28.eaf', 'KV 29.eaf', 'KV 3.eaf', 'KV 30.eaf', 'KV 4.eaf', 'KV 5.eaf', 'KV 6.eaf', 'KV 7.eaf', 'KV 8.eaf', 'KV 9.eaf', 'Media 1.eaf', 'Media 10.eaf', 'Media 16.eaf', 'Media 17.eaf', 'Media 18.eaf', 'Media 2.eaf', 'Media 20.eaf', 'Media 21.eaf', 'Media 22.eaf', 'Media 24.eaf', 'Media 25.eaf', 'Media 27.eaf', 'Media 28.eaf', 'Media 29.eaf', 'Media 30.eaf', 'Media 31.eaf', 'Media 32.eaf', 'Media 33.eaf', 'Media 34.eaf', 'Media 35.eaf', 'Media 36.eaf', 'Media 37.eaf', 'Media 38.eaf', 'Media 39.eaf', 'Media 4.eaf', 'Media 40.eaf', 'Media 41.eaf', 'Media 42.eaf', 'Media 43.eaf', 'Media 44.eaf', 'Media 47.eaf', 'Media 48.eaf', 'Media 49.eaf', 'Media 5.eaf', 'Media 50.eaf', 'Media 53.eaf', 'Media 54.eaf', 'Media 55.eaf', 'Media 56.eaf', 'Media 57.eaf', 'Media 58.eaf', 'Media 59.eaf', 'Media 6.eaf', 'Media 60.eaf', 'Media 61.eaf', 'Media 62.eaf', 'Media 63.eaf', 'Media 64.eaf', 'Media 8.eaf', 'Media 9.eaf']

    selected_ints = [1, 0, 1, 2, 2, 1, 1, 0, 1, 0, 0, 1, 0, 2, 2, 1, 0, 1, 2, 1, 0, 0, 0, 2, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 0, 0, 2, 1, 0, 1, 0, 1, 2, 0, 2, 1, 2, 1, 0, 0, 2, 1, 0, 2, 2, 1, 2, 1, 1, 0, 0, 2, 1, 2, 0, 2]
    # []


    file_names_index = ['KV 1.eaf', 'KV 10.eaf', 'KV 12.eaf', 'KV 13.eaf', 'KV 15.eaf', 'KV 18.eaf', 'KV 2.eaf', 'KV 20.eaf', 'KV 21.eaf', 'KV 22.eaf', 'KV 23.eaf', 'KV 25.eaf', 'KV 27.eaf', 'KV 3.eaf', 'KV 30.eaf', 'KV 4.eaf', 'KV 5.eaf', 'KV 7.eaf', 'KV 8.eaf', 'KV 9.eaf', 'Media 1.eaf', 'Media 10.eaf', 'Media 13.eaf', 'Media 15.eaf', 'Media 16.eaf', 'Media 18.eaf', 'Media 2.eaf', 'Media 20.eaf', 'Media 21.eaf', 'Media 22.eaf', 'Media 23.eaf', 'Media 25.eaf', 'Media 27.eaf', 'Media 28.eaf', 'Media 29.eaf', 'Media 3.eaf', 'Media 30.eaf', 'Media 33.eaf', 'Media 35.eaf', 'Media 36.eaf', 'Media 37.eaf', 'Media 39.eaf', 'Media 4.eaf', 'Media 41.eaf', 'Media 42.eaf', 'Media 44.eaf', 'Media 47.eaf', 'Media 48.eaf', 'Media 49.eaf', 'Media 50.eaf', 'Media 51.eaf', 'Media 52.eaf', 'Media 53.eaf', 'Media 54.eaf', 'Media 55.eaf', 'Media 56.eaf', 'Media 57.eaf', 'Media 58.eaf', 'Media 59.eaf', 'Media 6.eaf', 'Media 60.eaf', 'Media 61.eaf', 'Media 62.eaf', 'Media 63.eaf', 'Media 64.eaf', 'Media 8.eaf']

    # []
    


    # print list(file_names)
    # raw_input()
    # []
    matches = []
    # annotators_strs
    for file_name in file_names:
        rel_anno = arr[arr[:,7]==file_name,:]
        stress_type = np.unique(rel_anno[:,1])
        horse_id = np.unique(rel_anno[:,0])[0]

        annotators = np.unique(rel_anno[:,8])
        
        # rand_int = np.random.randint(len(annotators))
        # selected_ints.append(rand_int)
        # file_names_index.append(file_name)


        # print (file_name, len(annotators))

        # for annotator in annotators:
        file_code = file_names_index.index(file_name)
        rand_int = selected_ints[file_code]

        bin_keep = rel_anno[:,8]==annotators[rand_int]
        rel_anno = rel_anno[bin_keep,:]
        
        # print len(annotators)

        # print horse_id 
        # raw_input()
        # if horse_id=='19':
        #     print file_name, stress_type

        assert stress_type.size ==1
        if stress_type not in stress_strs:
            # print 'new type', stress_type
        # =='Post-induction':
            continue
        else:
            # print rel_anno[0,0],rel_anno[0,1]
            # file_code = len(stress_anno)
            file_codes.append(file_code)
            stress_anno.append(stress_strs.index(stress_type))
            matches.append([int(horse_id),int(file_code),int(stress_strs.index(stress_type))])
                # ,int(annotators_strs.index(annotator))])


        # film_name = int(rel_anno[0][0].strip().split()[1])
        aus = rel_anno[:,6]
        aus = [str(val) for val in aus.tolist()]

        time_lists = [rel_anno[:,2],rel_anno[:,3],rel_anno[:,4]]
        new_lists = []
        
        for idx_time_list, time_list in enumerate(time_lists):
            # if idx_time_list<2:
            #     continue
            new_list = []
            for val in time_list:
                val = '00:'+val
                val = util.convert_str_to_sec(val)
                # if val>60:
                #     print val
                new_list.append(val)
            new_lists.append(new_list)

        [start_time_list, end_time_list, duration] = new_lists
        # if horse_id=='19':
        #     print np.min(start_time_list), np.max(end_time_list)
        data_dict[file_code] = [aus, duration, start_time_list, end_time_list]
        
    
    # assert len(file_names_index)==len(selected_ints)
    # print '['+', '.join([str(val) for val in selected_ints])+']'
    # print '['+"', '".join(file_names_index)+']'
    # return None, None, None
    data_dict = clean_data(data_dict)

    # print selected_ints
    assert len(stress_anno)==len(file_codes)
    assert np.all(np.array(data_dict.keys())==np.array(file_codes))
    # print len(file_codes)
    # raw_input()
    return data_dict, stress_anno, file_codes





def read_in_data_stress_matches(file_name):
    df = pd.read_csv(file_name,header = None)
    arr = df.values
    arr = arr[1:,:]
    file_names = np.unique(arr[:,7])
    
    stress_strs = ['Baseline','Social isolation','Transportation']
    stressers_needed = np.array(['Baseline'])

    # stress_strs = ['Baseline','Social isolation']
    # stressers_needed = np.array(['Baseline','Social isolation'])

    # stress_strs = ['Baseline','Transportation']
    # stressers_needed = np.array(['Baseline','Transportation'])

    horse_ids = np.unique(arr[:,0])
    horse_ids_keep = []
    for horse_id in horse_ids:
        
        

        stressers = np.unique(arr[arr[:,0]==horse_id,1])
        if np.sum(np.in1d(stressers_needed,stressers))==stressers_needed.size:
            # print horse_id, stressers, fn.shape
            horse_ids_keep.append(horse_id)

        # print horse_id, stressers
    horse_ids_keep = np.array(horse_ids_keep)
    # print horse_ids_keep
    # print horse_ids_keep.size
    # raw_input()

    # horse_names_si = np.array([])
    bin_rows = np.in1d(arr[:,0],horse_ids_keep)
    file_names = np.unique(arr[bin_rows,7])
    

    # horse_names_si = np.array([])
    # bin_rows = np.in1d(arr[:,0],horse_names_si)
    # file_names = np.unique(arr[bin_rows,7])
    # stress_strs = ['Baseline','Transportation']    

    
    
    stress_anno = []
    file_codes = []
    data_dict = {}
    # selected_ints = [0, 0, 0, 0, 0, 2, 1, 2, 1, 1, 2, 1, 1, 0, 0, 1, 1, 1, 2, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 2, 0, 1, 2, 2, 0, 1, 2, 0, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 1, 0, 2, 0, 0, 2, 0, 0, 0, 0, 1, 0, 2, 2, 2, 2, 2, 0, 1]
    # file_names_index = ['KV 1.eaf', 'KV 10.eaf', 'KV 11.eaf', 'KV 12.eaf', 'KV 13.eaf', 'KV 14.eaf', 'KV 15.eaf', 'KV 16.eaf', 'KV 17.eaf', 'KV 18.eaf', 'KV 19.eaf', 'KV 2.eaf', 'KV 20.eaf', 'KV 21.eaf', 'KV 22.eaf', 'KV 23.eaf', 'KV 24.eaf', 'KV 25.eaf', 'KV 26.eaf', 'KV 27.eaf', 'KV 28.eaf', 'KV 29.eaf', 'KV 3.eaf', 'KV 30.eaf', 'KV 4.eaf', 'KV 5.eaf', 'KV 6.eaf', 'KV 7.eaf', 'KV 8.eaf', 'KV 9.eaf', 'Media 1.eaf', 'Media 10.eaf', 'Media 16.eaf', 'Media 17.eaf', 'Media 18.eaf', 'Media 2.eaf', 'Media 20.eaf', 'Media 21.eaf', 'Media 22.eaf', 'Media 24.eaf', 'Media 25.eaf', 'Media 27.eaf', 'Media 28.eaf', 'Media 29.eaf', 'Media 30.eaf', 'Media 31.eaf', 'Media 32.eaf', 'Media 33.eaf', 'Media 34.eaf', 'Media 35.eaf', 'Media 36.eaf', 'Media 37.eaf', 'Media 38.eaf', 'Media 39.eaf', 'Media 4.eaf', 'Media 40.eaf', 'Media 41.eaf', 'Media 42.eaf', 'Media 43.eaf', 'Media 44.eaf', 'Media 47.eaf', 'Media 48.eaf', 'Media 49.eaf', 'Media 5.eaf', 'Media 50.eaf', 'Media 53.eaf', 'Media 54.eaf', 'Media 55.eaf', 'Media 56.eaf', 'Media 57.eaf', 'Media 58.eaf', 'Media 59.eaf', 'Media 6.eaf', 'Media 60.eaf', 'Media 61.eaf', 'Media 62.eaf', 'Media 63.eaf', 'Media 64.eaf', 'Media 8.eaf', 'Media 9.eaf']

    selected_ints = [1, 0, 1, 2, 2, 1, 1, 0, 1, 0, 0, 1, 0, 2, 2, 1, 0, 1, 2, 1, 0, 0, 0, 2, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 0, 0, 2, 1, 0, 1, 0, 1, 2, 0, 2, 1, 2, 1, 0, 0, 2, 1, 0, 2, 2, 1, 2, 1, 1, 0, 0, 2, 1, 2, 0, 2]
    # []


    file_names_index = ['KV 1.eaf', 'KV 10.eaf', 'KV 12.eaf', 'KV 13.eaf', 'KV 15.eaf', 'KV 18.eaf', 'KV 2.eaf', 'KV 20.eaf', 'KV 21.eaf', 'KV 22.eaf', 'KV 23.eaf', 'KV 25.eaf', 'KV 27.eaf', 'KV 3.eaf', 'KV 30.eaf', 'KV 4.eaf', 'KV 5.eaf', 'KV 7.eaf', 'KV 8.eaf', 'KV 9.eaf', 'Media 1.eaf', 'Media 10.eaf', 'Media 13.eaf', 'Media 15.eaf', 'Media 16.eaf', 'Media 18.eaf', 'Media 2.eaf', 'Media 20.eaf', 'Media 21.eaf', 'Media 22.eaf', 'Media 23.eaf', 'Media 25.eaf', 'Media 27.eaf', 'Media 28.eaf', 'Media 29.eaf', 'Media 3.eaf', 'Media 30.eaf', 'Media 33.eaf', 'Media 35.eaf', 'Media 36.eaf', 'Media 37.eaf', 'Media 39.eaf', 'Media 4.eaf', 'Media 41.eaf', 'Media 42.eaf', 'Media 44.eaf', 'Media 47.eaf', 'Media 48.eaf', 'Media 49.eaf', 'Media 50.eaf', 'Media 51.eaf', 'Media 52.eaf', 'Media 53.eaf', 'Media 54.eaf', 'Media 55.eaf', 'Media 56.eaf', 'Media 57.eaf', 'Media 58.eaf', 'Media 59.eaf', 'Media 6.eaf', 'Media 60.eaf', 'Media 61.eaf', 'Media 62.eaf', 'Media 63.eaf', 'Media 64.eaf', 'Media 8.eaf']

    # print list(file_names)
    # raw_input()
    # []
    matches = []
    annotators_strs = ['alina', 'johan', 'camilla']
    for file_name in file_names:
        rel_anno = arr[arr[:,7]==file_name,:]
        stress_type = np.unique(rel_anno[:,1])
        horse_id = np.unique(rel_anno[:,0])[0]

        annotators = np.unique(rel_anno[:,8])
        # print (annotators)
        # rand_int = np.random.randint(len(annotators))
        # selected_ints.append(rand_int)
        for annotator in annotators:
            # idx_file_name = file_names_index.index(file_name)
            # rand_int = selected_ints[idx_file_name]
            # print annotator

            bin_keep = rel_anno[:,8]==annotator
            annotator = annotator.strip().lower()
            # s[rand_int]
            rel_anno = rel_anno[bin_keep,:]
            # print rel_anno.shape
            # print len(annotators)

            # print horse_id 
            # raw_input()
            # if horse_id=='19':
            #     print file_name, stress_type

            assert stress_type.size ==1
            if stress_type not in stress_strs:
                # print stress_type
            # =='Post-induction':
                continue
            else:
                # print rel_anno[0,0],rel_anno[0,1]
                file_code = len(stress_anno)
                file_codes.append(file_code)
                stress_anno.append(stress_strs.index(stress_type))
                annotator_idx = annotators_strs.index(annotator)
                horse_id_curr = horse_id
                # +str(annotator_idx)

                matches.append([int(horse_id_curr),int(file_code),int(stress_strs.index(stress_type)), annotator_idx])
                # ,int(annotators_strs.index(annotator))])
                # print matches[-1]
                # raw_input()


            # film_name = int(rel_anno[0][0].strip().split()[1])
            aus = rel_anno[:,6]
            # if file_code==
            for au in aus.tolist():
                au = str(au)
                
                au_check = au.lower().split()[0].strip()
                # print 'rin',au_check
                # if len(au.lower().split())==0:
                #     print 'PROBLEM', au, horse_id, annotator, file_name
                #     raw_input()

            # print aus
            # raw_input()
            # print rel_anno
            aus = [str(val) for val in aus.tolist()]
            
            time_lists = [rel_anno[:,2],rel_anno[:,3],rel_anno[:,4]]
            new_lists = []
            
            for idx_time_list, time_list in enumerate(time_lists):
                # if idx_time_list<2:
                #     continue
                new_list = []
                for val in time_list:
                    val = '00:'+val
                    val = util.convert_str_to_sec(val)
                    # if val>60:
                    #     print val
                    new_list.append(val)
                new_lists.append(new_list)

            [start_time_list, end_time_list, duration] = new_lists
            # if horse_id=='19':
            #     print np.min(start_time_list), np.max(end_time_list)
            data_dict[file_code] = [aus, duration, start_time_list, end_time_list]
        
    data_dict = clean_data(data_dict)

    # print selected_ints
    assert len(stress_anno)==len(file_codes)
    assert np.all(np.array(data_dict.keys())==np.array(file_codes))
    # print len(file_codes)
    # raw_input()
    return data_dict, [stress_anno,matches], file_codes



def main():

    # rename_files()
    # file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    # data_dict = read_start_stop_anno_file(file_name)
    # print data_dict.keys()
    # print len(data_dict[1][0])
    # print type(data_dict[1][1][0])

    # data_dict = clean_data(data_dict)
    # print data_dict[5]

    # file_name = '../data/clinical_cases_compiled_10_23_2019.csv'
    # read_clinical_file(file_name)
    # read_clinical_pain()
    # file_name)
    # data_dict_caps = read_caps_anno_file()



    # lines = util.readLinesFromFile('../data/johan_stress_4_26.csv')
    # lines = [line.replace(';',',').replace('Media 54 .eaf','Media 54.eaf').replace('L VC75', 'VC75 L')  for line in lines]
    # # lines = [line.replace(';',',') for line in lines]
    # util.writeFile('../data/johan_stress_4_26_comma.csv', lines)

    # return
    data_dict, stress_anno, file_codes = read_in_data_stress('../data/johan_stress_4_26_comma.csv')
    print len(data_dict.keys())
    [pain, matches] = stress_anno
    matches = np.array(matches)

    # bl_match = []
    tr_match =[]
    si_match = []
    for horse in np.unique(matches[:,0]):
        rel_rows = matches[matches[:,0]==horse,:]
        bl = rel_rows[rel_rows[:,2]==0,1][0]

        si = rel_rows[rel_rows[:,2]==1,1]
        tr = rel_rows[rel_rows[:,2]==2,1]
        if len(si)>0:
            si_match.append((bl,si[0]))
        if len(tr)>0:
            tr_match.append((bl, tr[0]))

    print (si_match)
    print (tr_match)


    # print np.array(matches).shape
    # print (matches)
    # [int(horse_id),int(file_code),int(stress_strs.index(stress_type))]
    # print pain
    # print np.array(pain).shape
    # print stress_anno
    # print file_codes



if __name__=='__main__':
    main()