import sklearn
from sklearn import preprocessing,decomposition,discriminant_analysis,pipeline,neighbors, metrics
from helpers import util, visualize
from read_in_data import read_start_stop_anno_file, clean_data, get_all_aus, read_clinical_file, read_caps_anno_file, read_in_data_stress, read_clinical_pain, read_in_data_stress_matches
import itertools
import loo_classifying as lc
import scipy
from start_stop_stuff import get_start_stop_feat, count_cooc, get_time_series_feat
from cooc import find_best_clusters_custom
import numpy as np
import os
import math


def plot_cumulative(out_file, features, labels, pain, file_str, separator):

    print file_str
    bin_pain = np.in1d(labels, pain)
    class_pain = np.zeros(bin_pain.shape)
    class_pain[bin_pain]=1
    class_pain = class_pain.astype(int)
    num_aus = np.sum(features,axis = 1)
    num_aus_p = num_aus[class_pain>0]
    num_aus_np = num_aus[class_pain<=0]
    
    num_bins = range(int(np.max(num_aus))+1)
    legend_entries = ['Pain','No Pain']
    vals = [num_aus_p, num_aus_np]

    k = scipy.stats.scoreatpercentile(num_aus_np, separator)
    k = int(math.floor(k))
    
    file_str.extend(['k',k])
    file_str = [str(val) for val in file_str]

    title = ' '.join(file_str)
    
    xlabel = 'Number of AUs'
    ylabel = 'Frequncy'
    cumulative = True
    xtick_labels = [str(val) for val in num_bins]
    visualize.plotMultiHist(out_file,vals = vals, num_bins = [num_bins, num_bins], legend_entries = legend_entries, title = title, xlabel = xlabel, ylabel = ylabel, xticks = xtick_labels, cumulative = cumulative, density = True, align = 'mid')
    return k


def get_feats(inc, step_size, flicker = 0,blink=0, data_type = 'frequency',clinical = False, type_dataset = 'isch', split_pain = False, get_matches = False):
    # print type_dataset
    pain_isch = [1,2,4,5,11,12]
    no_pain_isch = [3,6,7,8,9,10]
    # pain_caps = [13,14,15,16,17,18]

    if clinical:
        file_name = '../data/clinical_cases_compiled_10_23_2019.csv'
        data_dict = read_clinical_file(file_name)
        key_arr = list(data_dict.keys())
        pain = None
    else:    
        # raise ValueError('not a valid feat type '+str(feat_type))
        if type_dataset=='isch':
            file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
            data_dict = read_start_stop_anno_file(file_name)
            key_arr = list(data_dict.keys())
            pain = np.array(pain_isch)
        elif type_dataset =='clinical':
            file_name = '../data/clinical_cases_compiled_10_23_2019.csv'
            data_dict = read_clinical_file(file_name)
            key_arr = list(data_dict.keys())
            pain_dict = read_clinical_pain()
            vid_labels = []
            pain_list = []
            for key,val in pain_dict.items():
                vid_labels.append(key)
                pain_list.append(val)
            pain_list = np.array(pain_list)
            

            # print pain_list
            # pain_list[pain_list>1]=1
            # npain = np.sum(pain_list==0,axis = 1)
            # p = np.sum(pain_list==1,axis = 1)
            # for idx in range(len(p)):
            #     print '%d\t%d'%(npain[idx],p[idx])
            
            # raw_input()
            vid_labels = np.array(vid_labels)
            # print vid_labels
            # print pain_list[vid_labels==25]
            # print pain_list[vid_labels==7]            
            # pain_list = pain_list/2.
            # pain_list = np.mean(pain_list, axis = 1)
            # pain_list[pain_list>=0.5] = 1
            # pain_list[pain_list<0.5] = 0

            # print pain_list
            pain_list[pain_list>1]=1
            pain_list = np.sum(pain_list, axis = 1)

            pain_list[pain_list<2] = 0
            pain_list[pain_list>=2] = 1
            pain = vid_labels[pain_list>0]
            
            # remove_list = pain_list==1
            # pain_list[pain_list<2] = 0
            # pain_list[pain_list>=2] = 1
            # remove_list = vid_labels[remove_list]
            # pain = vid_labels[pain_list>0]
            # data_dict_new = {}
            # for k in data_dict.keys():
            #     if k not in remove_list:
            #         data_dict_new[k]=data_dict[k]
            # data_dict = data_dict_new
            # key_arr = list(data_dict.keys())

            # print vid_labels
            # print list(data_dict.keys())
            # print list(data_dict_new.keys())
            # print remove_list
            # print pain

            # print pain
            # raw_input()
            

        elif type_dataset=='caps':
            file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
            data_dict = read_start_stop_anno_file(file_name)
            caps_dict = read_caps_anno_file()
            dict_big = {}
            no_pain = np.array(no_pain_isch)
            for k in no_pain:
                dict_big[k] = data_dict[k]
            
            for k in caps_dict.keys():
                dict_big[k] = caps_dict[k]

            data_dict = dict_big
            key_arr = list(data_dict.keys())
            pain = np.array(caps_dict.keys())
        elif type_dataset=='both':
            file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
            
            data_dict = read_start_stop_anno_file(file_name)
            caps_dict = read_caps_anno_file()
            # print caps_dict.keys()
            dict_big = {}
            for dict_curr in [data_dict,caps_dict]:
                for k in dict_curr:
                    dict_big[k] = dict_curr[k]
            data_dict = dict_big
            
            key_arr = list(data_dict.keys())
            pain_caps = list(caps_dict.keys())
            # print pain_caps
            pain = np.array(pain_isch+pain_caps)
            # print pain
        elif type_dataset =='stress':
            # file_name = '../data/stress_all_anno.csv'
            file_name = '../data/johan_stress_4_26_comma.csv'
            if get_matches == True:
                data_dict, pain, key_arr = read_in_data_stress_matches(file_name)
            else:
                data_dict, pain, key_arr = read_in_data_stress(file_name, type_dataset)
            if get_matches:
                [pain, matches] = pain
            pain = np.array(pain)
            key_arr = np.array(key_arr)
            if not split_pain:
                pain = list(key_arr[pain>0])
                # print pain
            else:
                pain = [key_arr[pain==0],key_arr[pain==1],key_arr[pain==2]]
            # print pain
            key_arr = list(key_arr)
            # print pain
            if get_matches:
                pain = [pain, matches]

            # print len(key_arr)
            # print len(pain)
            # raw_input()
        elif type_dataset.startswith('stress'):
            # file_name = '../data/stress_all_anno.csv'
            file_name = '../data/johan_stress_4_26_comma.csv'
            data_dict, pain, key_arr = read_in_data_stress(file_name, stress_type = type_dataset.split('_')[1])
            pain = np.array(pain)
            key_arr = np.array(key_arr)
            pain = list(key_arr[pain>0])
            key_arr = list(key_arr)
        else:    
            raise ValueError('not a valid feat type '+str(type_dataset))
        # caps_dict = read_caps_anno_file()
        # dict_big = {}
        # for dict_curr in [data_dict,caps_dict]:
        #     for k in dict_curr:
        #         dict_big[k] = dict_curr[k]
        # data_dict = dict_big
        # key_arr = dict_big.keys()
        
    
    data_dict = clean_data(data_dict)
    if flicker>0:
        data_dict = create_flicker_category(data_dict, threshold = flicker)
    if blink>0:
        assert flicker>0
        data_dict = create_flicker_category(data_dict, threshold = blink,au_names = ['ead101_ead104','au101'])

    all_aus_org = get_all_aus(data_dict)
    
    # pain = np.array([1,2,4,5,11,12])

    feat_keep = ['au','ad','ead']
    # data_type = 'frequency'
    # feat_keep = None
    file_str =[feat_keep]+[data_type, inc, step_size]
    features, labels, all_aus = get_start_stop_feat(data_dict, all_aus_org, key_arr, inc, data_type, feat_keep = feat_keep, step_size = step_size)
    # print all_aus
    # raw_input()

    # print pain, key_arr
    return features, labels, all_aus, pain 
    # , matches

def plot_pruned_num_cooc(features, labels, all_aus, core_aus, pain, k, out_file):

    bin_keep = np.sum(features, axis= 1)>=k
    labels = labels[bin_keep]
    features = features[bin_keep, :]

    bin_pain = np.in1d(labels, pain)

    cols_keep = np.in1d(all_aus,core_aus)
    features = features[:,cols_keep]
    num_aus = np.sum(features>0,axis = 1)
    num_aus_p = num_aus[bin_pain]
    num_aus_np = num_aus[~bin_pain]
    print bin_pain.shape, np.sum(bin_pain)

    num_bins = range(int(np.max(num_aus)+2))
    vals = [num_aus_p, num_aus_np]
    legend_entries = ['Pain', 'No Pain']
    file_str = ['pruned','k',str(k)]
    title = ' '.join(file_str)
    xlabel = 'Number of AUs'
    ylabel = 'Frequency'
    
    xtick_labels = [str(val) for val in num_bins]
    visualize.plotMultiHist(out_file,vals = vals, num_bins = [num_bins, num_bins], legend_entries = legend_entries, title = title, xlabel = xlabel, ylabel = ylabel, xticks = xtick_labels,align = 'mid', density = True)

def plot_prob_classification(features, labels, all_aus, core_aus, out_dir, remove_aus= None, keep_aus = None):
    pain_labels = [1,2,4,5,11,12]
    no_pain_labels = [6,9,8,10,7,3]
    num_core_range = range(len(core_aus))
    num_aus = np.sum(features>0,axis = 1)
    
    if remove_aus is not None:
        keep_bin = filter_by_aus(features, all_aus, labels, remove_aus)
        keep_bin = np.logical_not(keep_bin)
        # cols = np.in1d(all_aus, remove_aus[0])

    if keep_aus is not None:
        keep_bin = filter_by_aus(features, all_aus, labels, keep_aus)
        # keep_bin = np.logical_not(keep_bin)
        # cols = np.in1d(all_aus, remove_aus[0])
        


    # cols_keep = np.in1d(all_aus,core_aus)
    # features = features[:,cols_keep]
    

    vid_types = ['pain','no pain']

    for idx_label, labels_rel in enumerate([pain_labels, no_pain_labels]):
        vid_type = vid_types[idx_label]
        xAndYs = []
        legend_entries = []
    
        for label_curr in labels_rel:
            
            bin_rel = labels==label_curr
            if remove_aus is not None or keep_aus is not None:
                bin_rel_rem = np.logical_and(bin_rel, keep_bin)
                # print bin_rel[:10]
                # print keep_bin[:10]
                # print bin_rel_rem[:10]
                # print features[bin_rel_rem,:][:,cols]
                rel_num_aus = num_aus[bin_rel_rem]
            else:
                rel_num_aus = num_aus[bin_rel]
            y_val = []
            
            for num_core_k in num_core_range:
                pain_times = np.sum(rel_num_aus>=num_core_k)/float(np.sum(bin_rel))
                y_val.append(pain_times)
            xAndYs.append((num_core_range, y_val))
            legend_entries.append(str(label_curr))

        
        out_file = os.path.join(out_dir, 'prob_classification_'+vid_type.replace(' ','_')+'.jpg')
        title = vid_type.title()+' Videos'
        ylabel = 'Pain Classification Probability'
        xlabel = 'Number of Core AUs Required'
        visualize.plotSimple(out_file = out_file, xAndYs = xAndYs, title = title, xlabel =xlabel, ylabel = ylabel, legend_entries = legend_entries)

    # title = 'No Pain Videos'
    # xlabel = 'Probability'
    # ylabel = 'Number of Core AUs Required'
    # visualize.plotSimple(out_file = out_file_np, xAndYs = xAndYs_np, title = title, xlabel =xlabel, ylabel = ylabel, legend_entries = legend_entries_np)

def convert_feat_binary_to_str(rel_occ_feats, aus_keep):
    feat_strs = []
    for rel_curr in rel_occ_feats:
        # print rel_curr
        # print aus_keep
        rel_aus = ','.join(list(aus_keep[rel_curr>0]))
        feat_strs.append(rel_aus)
    return feat_strs

def plot_cooc_clusters(features, labels, all_aus, core_aus, out_dir):
    pain_labels = [1]
    # ,2,4,5,11,12]
    no_pain_labels = [6]
    # ,9,8,10,7,3]
    num_core_range = range(1,len(core_aus)+1)
    cols_keep = np.in1d(all_aus,core_aus)
    aus_keep = np.array(all_aus)[cols_keep]
    features = features[:,cols_keep]
    print labels.shape
    print features.shape

    num_aus = np.sum(features>0,axis = 1)
    
    vid_types = ['pain','no pain']

    
    for num_core_k in num_core_range:
        au_clusters = []
        unique_clusters = []
        for idx_label, labels_rel in enumerate([pain_labels, no_pain_labels]):
            vid_type = vid_types[idx_label]
            feat_strs_all = []
            for label_curr in labels_rel:
                bin_rel = labels==label_curr
                rel_features = features[bin_rel,:]
                rel_num_aus = num_aus[bin_rel]
                rel_occs = rel_num_aus==num_core_k

                rel_occ_feats = rel_features[rel_occs]
                feat_strs = convert_feat_binary_to_str(rel_occ_feats, aus_keep)
                feat_strs_all += feat_strs
            
            unique_clusters += list(set(feat_strs_all))
            au_clusters.append(feat_strs_all)

        unique_clusters = list(set(unique_clusters))
        xtick_labels = []
        pain_counts = []
        no_pain_counts = []

        for cluster_curr in unique_clusters:
            xtick_labels.append(cluster_curr)
            pain_counts.append(au_clusters[0].count(cluster_curr))
            no_pain_counts.append(au_clusters[1].count(cluster_curr))
        dict_vals = {'Pain':pain_counts, 'No Pain': no_pain_counts}
        legend_vals = ['Pain','No Pain']

        file_str = ['cluster','frequency',num_core_k]
        title = ' '.join([str(val).title() for val in file_str])
        out_file = os.path.join(out_dir,'_'.join([str(val) for val in file_str])+'.jpg')
        colors = ['b','r']

        print out_file
        visualize.plotGroupBar(out_file,dict_vals,xtick_labels,legend_vals,colors,xlabel='AUs',ylabel='Frequency',title=title,width=0.4)
        raw_input()


def plot_print_cluster_stats(features, labels, all_aus, core_aus, out_dir, cluster_size, threshold = 0.1):
    pain_labels = [1,2,4,5,11,12]
    no_pain_labels = [6,9,8,10,7,3]
    num_core_range = range(1,len(core_aus)+1)
    cols_keep = np.in1d(all_aus,core_aus)
    aus_keep = np.array(all_aus)[cols_keep]
    features = features[:,cols_keep]
    
    num_aus = np.sum(features>0,axis = 1)
    
    vid_types = ['pain','no pain']

    xtick_labels =[]
    dict_vals = {'Pain':[], 'No Pain': []}
    legend_vals = ['Pain','No Pain']

    for aus_curr in itertools.combinations(aus_keep, cluster_size):
        cols_keep = np.in1d(aus_keep, aus_curr)
        features_curr = features[:,cols_keep]
        rel_num_aus = 0
        tot_counts = 0
        percents = []
        for idx_label, labels_rel in enumerate([pain_labels, no_pain_labels]):
            vid_type = vid_types[idx_label]
            
            for label_curr in labels_rel:
                bin_rel = labels==label_curr
                rel_features = features_curr[bin_rel,:]

                tot_counts += np.sum(num_aus>=1)
                rel_num_aus += np.sum(np.sum(rel_features>0, axis=1)==cluster_size)
            
            percent = rel_num_aus/float(tot_counts)
            # print percent
            percents.append(percent)
            print percents

        if (np.sum(np.array(percents)>=threshold))==2:
            xtick_labels.append(','.join(aus_curr))
            dict_vals['Pain'].append(percents[0])
            dict_vals['No Pain'].append(percents[1])






    #             rel_num_aus = num_aus[bin_rel]
    #             rel_occs = rel_num_aus==num_core_k

    #             rel_occ_feats = rel_features[rel_occs]
    #             feat_strs = convert_feat_binary_to_str(rel_occ_feats, aus_keep)
    #             feat_strs_all += feat_strs
            
    #         unique_clusters += list(set(feat_strs_all))
    #         au_clusters.append(feat_strs_all)

    # unique_clusters = list(set(unique_clusters))
    # xtick_labels = []
    # pain_counts = []
    # no_pain_counts = []

    # for cluster_curr in unique_clusters:
    #     xtick_labels.append(cluster_curr)
    #     pain_counts.append(au_clusters[0].count(cluster_curr))
    #     no_pain_counts.append(au_clusters[1].count(cluster_curr))
    
    if len(dict_vals['Pain'])==0:
        return

    file_str = ['cluster','frequency',str(cluster_size), str(threshold)]
    title = ' '.join([str(val).title() for val in file_str])
    out_file = os.path.join(out_dir,'_'.join([str(val) for val in file_str])+'.jpg')
    colors = ['b','r']

    print out_file
    visualize.plotGroupBar(out_file,dict_vals,xtick_labels,legend_vals,colors,xlabel='AUs',ylabel='Frequency',title=title,width=0.4)


def filter_by_aus(features, all_aus, labels, filter_aus, inverse= False):
    feature_bin = features>0
    row_bin = np.zeros((feature_bin.shape[0],))

    for filter_curr in filter_aus:
        cols_keep = np.in1d(all_aus, filter_curr)
        print cols_keep[:10]
        print np.sum(feature_bin[:,cols_keep], axis = 1)[:10]
        print np.sum(feature_bin[:,cols_keep], axis = 1)[:10]==len(filter_curr)
        print '----'
        # print np.sum(feature_bin[:,cols_keep], axis = 1).shape

        row_bin = np.logical_or(row_bin, np.sum(feature_bin[:,cols_keep], axis = 1) == len(filter_curr))

    # if inverse:
    #     row_bin = 1-row_bin

    return row_bin
    # features_keep = features[row_bin,:]
    # labels_keep = labels[row_bin]

    # return features_keep, labels_keep


def create_flicker_category(data_dict, threshold = 1.0, au_names = ['ead104','ead101']):
    keys = data_dict.keys()
    au_names.sort()
    for k in keys:
        
        data_curr = data_dict[k]
        data_curr = [np.array(val) for val in data_curr]
        times = np.array([data_curr[2],data_curr[3]]).T
        bin_104 = data_curr[0]==au_names[0]
        # 'ead104'
        bin_101 = data_curr[0]==au_names[1]
        # 'ead101'

        pairs = []

        for occ in np.where(bin_101)[0]:
            times_104 = times[bin_104,:]
            diff_start = np.abs(times_104 - times[occ,0])
            diff_end = np.abs(times_104 - times[occ,1])
            paired = np.logical_or(diff_start<threshold, diff_end<threshold)
            paired = np.where(np.sum(paired, axis = 1)>0)[0]
            paired = np.where(bin_104)[0][paired]
            
            if paired.size>0:
                pairs.append(list(paired)+[occ])

        bin_remove = np.zeros((times.shape[0],))
        starts = []
        ends = []
        durations = []
        aus = []
        for pair in pairs:
            # print pair
            starts.append(np.min(times[pair,0]))
            ends.append(np.max(times[pair,1]))
            durations.append(ends[-1] - starts[-1])
            aus.append('+'.join(au_names))
            bin_remove[pair]=1
        
        bin_keep = np.logical_not(bin_remove)
        data_curr = [list(val[bin_keep]) for val in data_curr]
        data_curr = [data_curr[val_idx]+val for val_idx, val in enumerate([aus, durations, starts, ends])]
        data_dict[k]= data_curr

    return data_dict



def finding_flicker_length():
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict)
    all_aus_org = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])
    feat_keep = ['au','ad','ead']
    data_type = 'duration'

    core_aus =['ead101', 'ead104']
    diffs =[]
    for key_curr in key_arr:
        data_curr = data_dict[key_curr]
        au_arr = np.array(data_curr[0])
        start_arr = np.array(data_curr[2])
        end_arr = np.array(data_curr[3])
        start_times = []
        for au_curr in core_aus:
            bin_rel = au_arr==au_curr
            start_rel = start_arr[bin_rel]
            start_times.append(start_rel)

        rep_row = start_times[0][np.newaxis,:]
        rep_col = start_times[1][:,np.newaxis]
        
        if rep_row.size ==0 or rep_col.size==0:
            continue
        rep_row = np.repeat(rep_row, rep_col.shape[0],0)
        rep_col = np.repeat(rep_col, rep_row.shape[1],1)
        axis_for_diff = 0 if rep_row.shape[1]<rep_col.shape[0] else 1
        
        diffs += list(np.min(np.abs(rep_row - rep_col),axis = axis_for_diff))
        
    out_dir = '../experiments/diff_between_front_back_ear_flicker'
    util.mkdir(out_dir)
    out_file = os.path.join(out_dir, 'start_times.jpg')
    visualize.hist(diffs,out_file, normed = False, title = 'Start time diff between ead104 and ead101')


def main():

    get_feats(30,30, flicker = 0,blink=0, data_type = 'frequency',clinical = False, type_dataset = 'clinical')


    return
    # file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    # data_dict = read_start_stop_anno_file(file_name)
    # data_dict = clean_data(data_dict)

    # create_flicker_category(data_dict, threshold = 1.0)
    # # finding_flicker_length()
    # return

    inc = 5
    step_size = 2.5
    separator = 50
    flicker = 1.
    pain = np.array([1,2,4,5,11,12])

    out_dir = '../experiments/plot_prob_classification_'+str(step_size)+'_'+str(inc)+'_wflicker_'+str(flicker)
    
    # map subtraction method threshold 0.29. rem 24 for 0.3
    core_aus = ['ad38',
            'au47',
            'au17',
            'au101',
            'ead101',
            'ad54',
            'au145',
            'ead104']
            # ,
            # 'au24']

    # sec 2, inc 1, thresh 0.25
    # core_aus = ['au101',
    #         'au47',
    #         'ad38',
    #         'ead104',
    #         'ead101',
    #         'au17',
    #         'ad1']
            
    

    # kunz_core
    out_dir = '../experiments/plot_prob_classification_kunz'
    core_aus = ['ad38',
                'au47',
                'au17',
                'au101',
                'ead104']

    util.mkdir(out_dir)
    util.writeFile(os.path.join(out_dir,'aus_considered.txt'), core_aus)
    

    features, labels, all_aus, file_str =get_feats(inc, step_size,flicker = flicker)


    # k = plot_cumulative(out_file, features, labels, pain, file_str, separator)

    # print features.shape
    # raw_input()
    # k = 0
    # out_file = os.path.join(out_dir,'pain_aus_cooc_hist_'+str(k)+'.jpg')

    # plot_pruned_num_cooc(features, labels, all_aus, core_aus, pain, k, out_file)

    plot_prob_classification(features, labels, all_aus, core_aus, out_dir)
    # remove_aus = [['ead101','ead104']]
    # remove_aus = [['ead1014']]
    # out_dir = os.path.join(out_dir,'flicker_removed')
    # util.mkdir(out_dir)
    # plot_prob_classification(features, labels, all_aus, core_aus, out_dir, remove_aus = remove_aus)

    # keep_aus = [['ead104'],['ad38']]
    # out_dir = os.path.join(out_dir,'keeping_ead104_38')
    # util.mkdir(out_dir)
    # plot_prob_classification(features, labels, all_aus, core_aus, out_dir, keep_aus = keep_aus)


    # plot_cooc_clusters(features, labels, all_aus, core_aus, out_dir)
    # for cluster_size in range(1,len(core_aus)+1):
    # cluster_size = 6
        # plot_print_cluster_stats(features, labels, all_aus, core_aus, out_dir, cluster_size, threshold = 0.007)
        # visualize.writeHTMLForFolder(out_dir)















if __name__=='__main__':
    main()