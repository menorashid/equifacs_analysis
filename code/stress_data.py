# from helpers import util, visualize
# import numpy as np
# import os
# import pandas as pd
# import read_in_data as rid
# from start_stop_stuff import get_start_stop_feat
# from script_cumulative_separation import get_feats
from doing_ml import *
# def read_clinical_file(file_name):
#     header_rows = [1,61,140,220,325,378,557,638,683,736,766,802,879,944,981,1062,1134,1225,1275,1335,1363,1452]
#     header_rows = [val-1 for val in header_rows]
#     df = pd.read_csv(file_name,header = None)
#     rel_annos = []
#     for idx_row,header_row in enumerate(header_rows[:-1]):
#         rel_anno_curr = df[header_row:header_rows[idx_row+1]-1]
#         rel_annos.append(rel_anno_curr.reset_index())
        
#     data_dict = {}
#     for rel_anno in rel_annos:
#         # print rel_anno[0][0]
#         film_name = int(rel_anno[0][0].strip().split()[1])
#         aus = rel_anno[4:][5]
#         aus = [str(val) for val in aus.tolist()]
#         time_lists = [rel_anno[4:][2],rel_anno[4:][3],rel_anno[4:][4]]
#         new_lists = []
        
#         for time_list in time_lists:
#             new_list = []
#             for val in time_list:
#                 val = util.convert_str_to_sec(val)
#                 # if val>60:
#                 #     print val
#                 new_list.append(val)
#             new_lists.append(new_list)

#         [start_time_list, end_time_list, duration] = new_lists
#         data_dict[film_name] = [aus, duration, start_time_list, end_time_list]
        

#     return data_dict

# def read_in_data(file_name):
#     df = pd.read_csv(file_name,header = None)
#     arr = df.values
#     arr = arr[1:,:]
#     file_names = np.unique(arr[:,7])
#     # print file_names
#     # pain_stats = []
#     # for file_name in file_names:
#     #   pain_stat = arr[arr[:,7]==file_name,1]
#     #   pain_stat = np.unique(pain_stat)
#     #   assert pain_stat.size==1
#     #   pain_stats.append(pain_stat[0])
#     # pain_stats = np.array(pain_stats)
#     # print file_names.shape
#     # print pain_stats.shape
#     # # print file_names
#     # # print pain_stats
#     # print np.unique(pain_stats)



#     # for pain_stat in np.unique(pain_stats):
#     #   print pain_stat, file_names[pain_stats==pain_stat].size

#     stress_strs = ['Baseline','Social isolation','Transportation']
#     stress_anno = []
#     file_codes = []
#     data_dict = {}
#     for file_name in file_names:
#         rel_anno = arr[arr[:,7]==file_name,:]
#         stress_type = np.unique(rel_anno[:,1])
#         assert stress_type.size ==1
#         if stress_type=='Post-induction':
#             continue
#         else:
#             file_code = len(stress_anno)
#             file_codes.append(file_code)
#             stress_anno.append(stress_strs.index(stress_type))


#         # film_name = int(rel_anno[0][0].strip().split()[1])
#         aus = rel_anno[:,6]
#         aus = [str(val) for val in aus.tolist()]

#         time_lists = [rel_anno[:,2],rel_anno[:,3],rel_anno[:,4]]
#         new_lists = []
        
#         for idx_time_list, time_list in enumerate(time_lists):
#             # if idx_time_list<2:
#             #     continue
#             new_list = []
#             for val in time_list:
#                 val = '00:'+val
#                 val = util.convert_str_to_sec(val)
#                 # if val>60:
#                 #     print val
#                 new_list.append(val)
#             new_lists.append(new_list)

#         [start_time_list, end_time_list, duration] = new_lists
#         # print np.max(end_time_list)
#         # aus = np.array(aus)
        

#         # if np.sum(aus=='nan')>0:
#         #     print idx_time_list
#         #     print rel_anno[aus=='nan',:]
#         #     raw_input()
#         # print file_code
#         data_dict[file_code] = [aus, duration, start_time_list, end_time_list]
        
#     # print data_dict.keys()
#     # print len(data_dict.keys())
#     # print stress_anno
#     # print len(stress_anno)

#     # all_aus_org = rid.get_all_aus(data_dict)
#     data_dict = rid.clean_data(data_dict)

    
#     assert len(stress_anno)==len(file_codes)
#     assert np.all(np.array(data_dict.keys())==np.array(file_codes))

#     return data_dict, stress_anno, file_codes



#     # print arr.shape
#     # print df[7][0]
#     # print df.keys()
#     # file_names = df[

def checks():
    # file_name = '../data/stress_all_anno.csv'
    # data_dict, stress_anno, key_arr = read_in_data(file_name)

    data_type = 'frequency'
    step_size = 30
    inc = 30
    type_dataset = 'stress'
    features, labels, all_aus, pain = get_feats(inc, step_size, data_type = data_type, type_dataset = type_dataset)

    # feat_keep = ['au','ad','ead']
    # all_aus_org = rid.get_all_aus(data_dict)

    # data_type = 'frequency'
    # step_size = 30
    # inc = 30
    # stress_anno =np.array(stress_anno)
    # key_arr = np.array(range(len(stress_anno)))
    # print key_arr
    # print data_dict.keys()

    # list(np.where(stress_anno>0)[0])
    # print key_arr

    # feat_keep = None
    # file_str =[feat_keep]+[data_type, inc, step_size]
    # features, labels, all_aus = get_start_stop_feat(data_dict, all_aus_org, key_arr, inc, data_type, feat_keep = feat_keep, step_size = step_size)

    print features.shape
    print labels.shape
    print all_aus


def main():
    
    # results_fs = []
    # for fs, fs_name in zip(fs_list, fs_list_names):
    out_dir_meta = '../experiments/stress_exp'
    util.mkdir(out_dir_meta)
    type_dataset = 'stress'        
    feature_selection = 'kunz'
    ows = [[30,30]]
    selection_params = dict(type_dataset=type_dataset)
    feature_types = [['frequency'],['duration'],['frequency','duration']]
    eval_methods = ['majority']

    # ml_model = {}
    # ml_model['norm'] = 'l2_mean_std'
    # ml_model['model_type'] = 'svm'
    # ml_model['model_params'] = {'C':1.,'kernel' : 'linear','class_weight':'balanced'}
    # ml_model['bootstrap'] = False
    ml_model = {}
    ml_model['norm'] = 'l2_mean_std'
    ml_model['model_type'] = 'lda'
    ml_model['model_params'] = None
    ml_model['bootstrap'] = False

    norm = ml_model['norm']
    model_type = ml_model['model_type']
    model_params = ml_model['model_params']
    bootstrap = ml_model['bootstrap']

    results = script_loo(out_dir_meta, ows, feature_types, feature_selection,selection_params, eval_methods = eval_methods, norm = norm, model_type = model_type, model_params = model_params, bootstrap = bootstrap)  
            # result_list.append(results[np.newaxis,:,:])

    print results

if __name__=='__main__':
    main()

