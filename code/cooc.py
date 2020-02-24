import sklearn
from sklearn import preprocessing,decomposition,discriminant_analysis,pipeline,neighbors, metrics
from helpers import util, visualize
from read_in_data import read_start_stop_anno_file, clean_data, get_all_aus
import itertools
import loo_classifying as lc
import scipy
from start_stop_stuff import get_start_stop_feat, count_cooc, get_time_series_feat
import numpy as np
import os
import networkx as nx

def get_cooc_mat(features, data_keep, feat_keep, bin_pain, all_aus):
    if data_keep=='pain':
        features_curr = features[bin_pain,:]
    elif data_keep=='no_pain':
        features_curr = features[~bin_pain,:]
    else:
        features_curr = features

    cooc_bin, sums, classes = count_cooc(features_curr, all_aus, feat_keep)
    
    sums_div = np.array(sums)
    sums_div[sums_div==0]=1
    sums_div = sums_div.T

    cooc_norm = cooc_bin/sums_div
    return cooc_bin, cooc_norm, classes, sums.T

def script_plot_cooc():
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict,remove_lr=True)
    all_aus = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])
    # bin_pain = np.in1d(np.array(key_arr),pain)
    # print bin_pain

    data_keeps = ['pain','no_pain','all']
    feat_keeps = [['au','ead']]
    out_dir_meta = '../experiments/visualizing_cooc_12'
    util.mkdir(out_dir_meta)
    inc_range = [(5,2.5)]
    max_diff = False
    # 5,10,15,30]

    for inc,step_size in inc_range:
        features, labels,_ = get_start_stop_feat(data_dict, all_aus, key_arr, inc, 'binary', step_size = step_size)
        print inc, features.shape
        bin_pain = np.in1d(labels,pain)
        if max_diff:
            features, bin_pain, val_max = prune_max_diff(features, bin_pain )
            
        for data_keep, feat_keep in itertools.product(data_keeps,feat_keeps):
            
            cooc_bin, cooc_norm, classes = get_cooc_mat(features, data_keep, feat_keep, bin_pain, all_aus)
            
            out_dir = os.path.join(out_dir_meta,'_'.join(feat_keep+[str(inc)]))
            if max_diff:
                out_dir = out_dir+'_maxdiff'
            util.mkdir(out_dir)

            figsize = (0.5*cooc_bin.shape[0]+0.5,0.5*cooc_bin.shape[1]-0.5)
            # cooc_bin = cooc_bin[:-1,:]

            file_str = [data_keep,str(int(inc)),'seconds']+feat_keep
            if max_diff:
                file_str.extend(['k',str(val_max)])
            title = ' '.join([val.title() for val in file_str])
            out_file = os.path.join(out_dir,'_'.join(file_str)+'.jpg')
            visualize.plot_confusion_matrix(cooc_bin.astype(int), classes, out_file,normalize=False,title=title,ylabel = '', xlabel = '', figsize = figsize)

            file_str += ['normalized']
            title = ' '.join([val.title() for val in file_str])
            out_file = os.path.join(out_dir,'_'.join(file_str)+'.jpg')
            visualize.plot_confusion_matrix(cooc_norm, classes, out_file,normalize=False,title=title,ylabel = '', xlabel = '',fmt = '.2f', figsize = figsize)

            visualize.writeHTMLForFolder(out_dir,height = figsize[1]*100,width = figsize[0]*100)
            # raw_input()


def plot_cooc(cooc_diff, file_str, out_dir, classes):
    # file_str = [data_keep,str(int(inc)),'%.1f'%step_size]+feat_keep
    # file_str += ['normalized']+['horse',str(k)]
    file_str = [str(val) for val in file_str]
    title = ' '.join([val.title() for val in file_str])
    out_file = os.path.join(out_dir,'_'.join(file_str)+'.jpg')
    # out_file = '../scratch/check_diff_nn.jpg'
    # out_file = os.path.join(out_dir,
    figsize = (0.5*cooc_diff.shape[0]+0.5,0.5*cooc_diff.shape[1]-0.5)
    visualize.plot_confusion_matrix(cooc_diff, classes, out_file,fmt = '.2f',figsize = figsize)
    return out_file

def find_best_clusters():
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict,remove_lr=True)
    all_aus = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])
    
    data_keeps = ['pain','no_pain']
    feat_keep = ['au','ead','ad']
    out_dir = '../experiments/visualizing_cooc_12_diff'
    util.mkdir(out_dir)
    inc = 5
    step_size = 2.5
    max_diff = False

    # for k in key_arr:
    features, labels,all_aus_t = get_start_stop_feat(data_dict, all_aus, key_arr, inc, 'binary', step_size = step_size)
    # print features.shape, labels.shape
    # print features[:10]
    # print labels[:10]
    # print all_aus_t
    bin_pain = np.in1d(labels,pain)
    if max_diff:
        features, bin_pain, val_max = prune_max_diff(features, bin_pain )
    

    cooc_norm_all = []
    sums_all = []
    for data_keep in data_keeps:    
        cooc_bin, cooc_norm, classes, sums = get_cooc_mat(features, data_keep, feat_keep, bin_pain, all_aus)
        cooc_norm_all.append(cooc_norm)
        sums_all.append(sums)
        file_str = [data_keep,inc, step_size]+feat_keep+['normalized']
        out_file = plot_cooc(cooc_norm, file_str, out_dir, classes)

    file_str = ['diff',inc, step_size]+feat_keep+['normalized']
    cooc_diff = np.abs(cooc_norm_all[0]-cooc_norm_all[1])
    out_file = plot_cooc(cooc_diff, file_str, out_dir, classes)
    

    # print cooc_diff.shape
    cooc_diff_sum = np.sum(cooc_diff,axis = 0)

    # print cooc_norm_all[0]>0
    # print cooc_norm_all[1]>0
    # raw_input()
    # print (cooc_norm_all[0]>0+cooc_norm_all[1]>0)>0
    num_non_zero = np.sum((cooc_norm_all[0]>0+cooc_norm_all[1]>0)>0,axis = 0)

    # np.sum(cooc_diff>0, axis = 0)
    # print sums_all[1].shape
    # print coof_diff_sum.shape
    average_diff = cooc_diff_sum/num_non_zero
    # /sums_all[1].squeeze()
    arg_sort = np.argsort(average_diff)[::-1]
    to_print = []
    for idx in arg_sort:
        str_curr = ' '.join([str(val) for val in [classes[idx], average_diff[idx]]])
        # print str_curr
        to_print.append(str_curr)

    util.writeFile(out_file.replace('.jpg','.txt'),to_print)

def find_best_clusters_custom(features, labels, all_aus, pain,feat_keep =None,  out_dir = None, inc = None, step_size = None ,plot_it = False):
    # file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    # data_dict = read_start_stop_anno_file(file_name)
    # data_dict = clean_data(data_dict,remove_lr=True)
    # all_aus = get_all_aus(data_dict)
    # key_arr = range(1,13)
    # pain = np.array([1,2,4,5,11,12])
    # bin_pain = np.in1d(np.array(key_arr),pain)
    # # print bin_pain

    data_keeps = ['pain','no_pain']
    # feat_keep = ['au','ead','ad']
    # out_dir = '../experiments/visualizing_cooc_12_diff'
    # util.mkdir(out_dir)
    # inc = 2
    # step_size = 1
    # max_diff = False

    # # for k in key_arr:
    # features, labels,_ = get_start_stop_feat(data_dict, all_aus, key_arr, inc, 'binary', step_size = step_size)
    
    bin_pain = np.in1d(labels,pain)
    # if max_diff:
    #     features, bin_pain, val_max = prune_max_diff(features, bin_pain )
    

    cooc_norm_all = []
    sums_all = []
    for data_keep in data_keeps:    
        cooc_bin, cooc_norm, classes, sums = get_cooc_mat(features, data_keep, None, bin_pain, all_aus)
        cooc_norm_all.append(cooc_norm)
        sums_all.append(sums)
        if plot_it:
            file_str = [data_keep,inc, step_size]+feat_keep+['normalized']
            out_file = plot_cooc(cooc_norm, file_str, out_dir, classes)

    cooc_diff_org = cooc_norm_all[0]-cooc_norm_all[1]
    if plot_it:
        file_str = ['diff',inc, step_size]+feat_keep+['normalized']
        out_file = plot_cooc(cooc_diff_org, file_str, out_dir, classes)

    average_diffs = []

    # np.save( '../experiments/cooc_simple/cooc_mat.npy',cooc_diff_org)
    # np.save( '../experiments/cooc_simple/classes.npy',np.array(classes))
    # print 'done saving'
    # np.save( '../experiments/cooc_simple/cooc_mat.npy',cooc_diff_org)
    # np.save( '../experiments/cooc_simple/classes.npy',np.array(classes))
    # print 'done saving'

    cooc_diff = np.abs(cooc_diff_org) 
    cooc_diff_sum = np.sum(cooc_diff,axis = 0)
    num_non_zero = np.sum(np.logical_or(cooc_norm_all[0]>0 , cooc_norm_all[1]>0).astype(int),axis = 0)
    num_non_zero[num_non_zero==0]=1
    average_diff = cooc_diff_sum
    # /num_non_zero
    # print average_diff

    average_diffs.append(average_diff)

    cooc_diff_pos = np.array(cooc_diff_org)
    cooc_diff_pos[cooc_diff_pos<0]=0

    cooc_diff_neg = np.array(cooc_diff_org)
    cooc_diff_neg[cooc_diff_neg>0]=0
    cooc_diff_neg = np.abs(cooc_diff_neg)

    cooc_percents = []
    for cooc_curr in [cooc_diff_pos, cooc_diff_neg]:
        cooc_diff_sum_curr = np.sum(cooc_curr,axis = 0)
        cooc_percent = cooc_diff_sum_curr/cooc_diff_sum
        cooc_percents.append(cooc_percent)
        # num_non_zero = np.sum(cooc_curr>0, axis = 0)
        # num_non_zero[num_non_zero==0]=1
        # average_diffs.append(cooc_diff_sum/num_non_zero)

    # print len(average_diffs)
    arg_sorts = []
    to_print = []
    # for average_diff in average_diffs:
    arg_sort = np.argsort(average_diff)[::-1]
    # arg_sorts.append(arg_sort)
    
    for idx in arg_sort:
        str_curr = ' '.join([str(val) for val in [classes[idx], average_diff[idx], cooc_percents[0][idx], cooc_percents[1][idx]]])
        to_print.append(str_curr)
    to_print.append('____')

    cooc_percents = [cooc_percent[arg_sort] for cooc_percent in cooc_percents]
    if plot_it:
        util.writeFile(out_file.replace('.jpg','.txt'),to_print)
    return np.array(classes)[arg_sort], average_diff[arg_sort], cooc_percents


def find_best_clusters_clique(features, labels, all_aus, pain,feat_keep =None,  out_dir = None, inc = None, step_size = None ,plot_it = False):
    
    data_keeps = ['pain','no_pain']
    
    bin_pain = np.in1d(labels,pain)
    

    cooc_norm_all = []
    sums_all = []
    for data_keep in data_keeps:    
        cooc_bin, cooc_norm, classes, sums = get_cooc_mat(features, data_keep, None, bin_pain, all_aus)
        cooc_norm_all.append(cooc_norm)
        sums_all.append(sums)
        if plot_it:
            file_str = [data_keep,inc, step_size]+feat_keep+['normalized']
            out_file = plot_cooc(cooc_norm, file_str, out_dir, classes)

    cooc_diff_org = cooc_norm_all[0]-cooc_norm_all[1]
    if plot_it:
        file_str = ['diff',inc, step_size]+feat_keep+['normalized']
        out_file = plot_cooc(cooc_diff_org, file_str, out_dir, classes)

    average_diffs = []

    cooc_bin = (cooc_diff_org>0).astype(int)
    cooc_up = np.triu(cooc_bin)
    cooc_down = np.tril(cooc_bin)
    cooc_bin = (cooc_up+cooc_down.T)>1

    G = nx.from_numpy_matrix(cooc_bin)
    cliques =  list(nx.find_cliques(G))
    clique_idx = []
    clique_sum = []
    for idx_k, k in enumerate(cliques):
        
        k.sort()        
        edges = cooc_diff_org[k,:]
        edges = edges[:,k]
        edges[edges<0]=0
        clique_idx.append(k)
        clique_sum.append(np.sum(edges))
        

    max_idx = np.argmax(clique_sum)
    max_clique = clique_idx[max_idx]
    return classes, max_clique




def find_best_clusters_simple(features, labels, all_aus, pain,feat_keep =None,  out_dir = None, inc = None, step_size = None ,plot_it = False):
    

    data_keeps = ['pain','no_pain']
    bin_pain = np.in1d(labels,pain)
    cooc_norm_all = []
    sums_all = []
    for data_keep in data_keeps:    
        cooc_bin, cooc_norm, classes, sums = get_cooc_mat(features, data_keep, None, bin_pain, all_aus)

        # print sums.shape
        cooc_norm[cooc_norm==0]=1./21.
        cooc_norm_all.append(cooc_norm)
            # cooc_bin+1)
        # sums_all.append(sums)
        if plot_it:
            file_str = [data_keep,inc, step_size]
            # +feat_keep+['normalized']
            out_file = plot_cooc(cooc_norm_all[-1], file_str, out_dir, classes)

    # total_cooc = (cooc_norm_all[1]+cooc_norm_all[0])
    # total_cooc[total_cooc==0]=1
    # cooc_norm_all[cooc_norm_all==0]=1./cooc_norm_all.shape[0]
    cooc_diff = (cooc_norm_all[0]-cooc_norm_all[1])
    # /total_cooc

    if plot_it:
        file_str = ['diff',inc, step_size]
        # +feat_keep+['normalized']
        out_file = plot_cooc(cooc_diff, file_str, out_dir, classes)
    
    # cooc_diff[cooc_diff<0]=0
    cooc_diff_sum = np.sum(np.abs(cooc_diff),axis = 1)
    num_non_zero = np.sum(np.abs(cooc_diff)>0, axis = 1)

    num_non_zero[num_non_zero==0]=1
    average_diff = cooc_diff_sum/num_non_zero

    arg_sort = np.argsort(average_diff)[::-1]
    to_print = []
    for idx in arg_sort:
        str_curr = ' '.join([str(val) for val in [classes[idx], average_diff[idx]]])
        to_print.append(str_curr)

    if plot_it:
        util.writeFile(out_file.replace('.jpg','.txt'),to_print)
    # else:
    return np.array(classes)[arg_sort], average_diff[arg_sort]



def looking_at_time_series():
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict,remove_lr=True)
    all_aus = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])
    
    out_dir = '../experiments/visualizing_ear_movements'
    util.mkdir(out_dir)
    data_keeps = ['pain','no_pain']
    # feat_keep = ['au','ead','ad38']
    # feat_keep = ['ead101','ead101l','ead101r','ead104','ead104l','ead104r',]
    feat_keep = ['ead101','ead104', 'ead103','ead102']
    # feat_keep = ['au101','ad38']
    mat_time, feat_keep = get_time_series_feat(data_dict, feat_keep, key_arr)

    x_axis = np.arange(0,30,0.01)
    for k in mat_time.keys():
        
        mat_curr = mat_time[k][0]
        
        xAndYs = []
        for r in range(mat_curr.shape[0]):
            mat_curr[r,:] = mat_curr[r,:]*(r+1)
            xAndYs.append((list(x_axis), list(mat_curr[r,:])))
        
        file_str = 'pain' if k in pain else 'no pain'
        file_str = [file_str]+[k]+feat_keep
        out_file = '_'.join([str(val) for val in file_str])+'.jpg'
        out_file = os.path.join(out_dir, out_file)
        title = ' '.join([str(val).title() for val in file_str])
        xlabel = 'Time'
        ylabel = 'Presence'
        # print out_file
        visualize.plotSimple( xAndYs,out_file, title , xlabel, ylabel, legend_entries = feat_keep)
        # raw_input()
    print out_dir
    visualize.writeHTMLForFolder(out_dir)
    

def get_freq_dur_dicts(data_dict, key_arr, core_aus):
    au_freq = {}
    au_dur = {}
    for au_curr in core_aus:
        au_freq[au_curr] = []
        au_dur[au_curr] = []

    for key in key_arr:
        # print type(data_dict[key][0])
        # print type(data_dict[key][1])
        au_arr = np.array(data_dict[key][0])
        dur_arr = np.array(data_dict[key][1])

        for au_curr in core_aus:
            bin_rel = au_arr == au_curr
            freq = np.sum(bin_rel)
            if freq ==0:
                continue

            dur = [[key, val] for val in dur_arr[bin_rel]]
            # dur = [[key, np.max(dur_arr[bin_rel])]]
            au_freq[au_curr].append([key,freq])
            au_dur[au_curr].extend(dur)
    return au_freq, au_dur

def comparing_frequency_duration():
    core_aus = ['ad38',
                'au47',
                'au17',
                'au101',
                # 'ead101',
                # 'au145',
                'ead104',]
                # 'ad54',
                # 'au24',
                # 'au18',
                # 'au10']

    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict,remove_lr=True)
    all_aus = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])    

    # out_dir = '../experiments/comparing_frequency_duration_core'

    # keep = ['au','ad','ead']
    # not_keep = ['vc','auh']
    # core_aus = []
    # for au_curr in all_aus:

    #     # for not_keep_curr in not_keep:    
    #     if au_curr.startswith(not_keep[0]) or au_curr.startswith(not_keep[1]):
    #         continue
    #     core_aus.append(au_curr)

    # out_dir = '../experiments/comparing_frequency_all'
    out_dir = '../experiments/comparing_frequency_duration_kunz_core_newcolor'
    util.mkdir(out_dir)
    # type_feat = 'frequency'

    au_freq, au_dur = get_freq_dur_dicts(data_dict, key_arr, core_aus)

  
    pain_labels = [1,2,4,5,11,12]
    no_pain_labels = [6,9,8,10,7,3]
    dur = False
    if dur:
        au_freq = au_dur

    for au_curr in core_aus:
        au_freq_rn = np.array(au_freq[au_curr])
        print au_freq_rn.size
        xtick_labels = []
        legend_entries = []
        # colors = ['b','r']
        colors = ['#842a4b','#206159']
        pain_vals = []
        no_pain_vals = []
        
        for idx_horse, pain_horse in enumerate(pain_labels):
            str_curr = str(idx_horse+1)
            xtick_labels.append(str_curr)
            
            au_freq_curr = au_freq_rn[au_freq_rn[:,0]==pain_horse,1]
            if au_freq_curr.size<1:
                au_freq_curr = 0
            else:
                # print au_freq_curr
                if dur:
                    au_freq_curr = np.max(au_freq_curr)
                else:
                    au_freq_curr = au_freq_curr[0]

            pain_vals.append(au_freq_curr)
            au_freq_curr = au_freq_rn[au_freq_rn[:,0]==no_pain_labels[idx_horse],1]
            if au_freq_curr.size<1:
                au_freq_curr = 0
            else:
                # print au_freq_curr
                if dur:
                    au_freq_curr = np.max(au_freq_curr)
                else:
                    au_freq_curr = au_freq_curr[0]

            no_pain_vals.append(au_freq_curr)
        
        print au_curr
        print 'pain_vals, no_pain_vals',pain_vals, no_pain_vals, len(pain_vals), len(no_pain_vals)
        diff = np.array(pain_vals)-np.array(no_pain_vals)
        print np.mean(diff), np.std(diff)

        raw_input()
        legend_vals = ['Pain','No Pain']
        dict_vals = {'Pain':pain_vals,'No Pain':no_pain_vals}
        if dur:
            ylabel = 'Duration in Seconds'
        else:
            ylabel = 'Frequency'
        # title = 'count'.title()+' '+au_curr.upper()
        title = au_curr.upper()
        # +' Maximum Duration'
        if dur:
            out_file = os.path.join(out_dir, 'dur_'+au_curr+'_per_horse_hist.jpg')
        else:
            out_file = os.path.join(out_dir, 'freq_'+au_curr+'_per_horse_hist.jpg')
        # xtick_labels = all_aus
        visualize.plotGroupBar(out_file ,dict_vals,xtick_labels,legend_vals,colors,xlabel='Horse ID',ylabel = ylabel,title=title,width=0.4,ylim=None,loc=-1)
        # raw_input()



    return

    for type_feat,feat_dict in zip(['count','duration'],[au_freq,au_dur]):
        out_dir_curr = os.path.join(out_dir, type_feat)
        util.mkdir(out_dir_curr)
        sums = [[],[]]
        for au_curr in feat_dict.keys():
            arr_curr = np.array(feat_dict[au_curr])
            bin_rel = np.in1d(arr_curr[:,0],pain)
            pain_feats = arr_curr[bin_rel, 1]
            no_pain_feats = arr_curr[~bin_rel, 1]
            
            sums[0].append(np.sum(pain_feats))
            sums[1].append(np.sum(no_pain_feats))

            maxes = [np.max(feat_curr)  if feat_curr.size>0 else 0 for feat_curr in [pain_feats, no_pain_feats]]
            max_val = max(maxes[0], maxes[1])
            
            if type_feat == 'count':
                num_bins = range(0,max_val+2)
            else:
                inc = max(0.1,max_val/10.)
                num_bins = list(np.arange(0,max_val+inc,inc))
            

            file_str = [type_feat, au_curr]
            out_file = os.path.join(out_dir_curr, '_'.join(file_str)+'.jpg')
            title = ' '.join([type_feat.title(),au_curr.upper()])
            xlabel = type_feat.title()
            ylabel = 'Frequency'
            legend_entries = ['Pain', 'No Pain']
            vals = [pain_feats, no_pain_feats]
            if type_feat == 'duration':
                xtick_labels = ['%.2f'%val for val in num_bins]
                rotate = True
            else:
                xtick_labels = [str(val) for val in num_bins]
                rotate = False

            visualize.plotMultiHist(out_file,vals = vals, num_bins = [num_bins, num_bins], legend_entries = legend_entries, title = title, xlabel = xlabel, ylabel = ylabel, xticks = xtick_labels, rotate = rotate)


        pain_sum = np.array(sums[0])
        no_pain_sum = np.array(sums[1])
        totals = pain_sum
        # +no_pain_sum
        diffs = pain_sum-no_pain_sum
        percents = totals/float(np.sum(totals))
        idx_sort = np.argsort(percents)
        au_types = list(feat_dict.keys())
        for idx_curr in idx_sort:
            print au_types[idx_curr],percents[idx_curr], diffs[idx_curr]

        visualize.writeHTMLForFolder(out_dir_curr)

def time_series_analysis():
    core_aus = [
                'ad38',
                # 'au47',
                # 'au17',
                # 'au101',
                'ead101',
                # 'au145',
                'ead104', 
                # 'ad54',
                # 'au24',
                # 'au18',
                # 'au10'
                ]
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict,remove_lr=True)
    all_aus = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])    

    out_dir = '../experiments/time_series_core_hists_inc_15_1'
    util.mkdir(out_dir)
    # type_feat = 'frequency'
    mat_time, feat_keep = get_time_series_feat(data_dict, core_aus, key_arr, decimal_place = 0)
    print feat_keep
    pain_mat = []
    no_pain_mat = []
    for k in mat_time:
        if k in pain:
            pain_mat.append(mat_time[k][0][:,:,np.newaxis])
        else:
            no_pain_mat.append(mat_time[k][0][:,:,np.newaxis])
    pain_mat = np.concatenate(pain_mat, axis = 2)
    no_pain_mat = np.concatenate(no_pain_mat, axis = 2)

    lim = pain_mat.shape[1]+1
    # inc_range = range(1,31)
    inc_range = [6]
    # range(1,15,1)
    # [1]
    # range(1,50,1)
    # (0.5,15,0.5)
    for inc in inc_range:
        num_count = 0
        start = 0
        counts = [[],[]]
        while start<lim:

            end = start+inc
            if end>=lim:
                break

            print start,end

            for idx_mat_curr,mat_curr in enumerate([pain_mat,no_pain_mat]):
                # print mat_curr[:,start:end,:]
                # print np.sum(mat_curr[:,start:end,:], axis = 1).shape
                # print np.sum(np.sum(mat_curr[:,start:end,:], axis = 1),axis = 0)>0
                # print np.sum(np.sum(mat_curr[:,start:end,:], axis = 1)>0,axis = 0)
                
                freq_curr = np.sum(np.sum(mat_curr[:,start:end,:],axis = 1)>0, axis = 0)
                # print freq_curr
                counts[idx_mat_curr].append(freq_curr)
                # raw_input()
            num_count+=1
            start+=1

        vals = [np.concatenate(val) for val in counts]
        # print vals[0].shape, vals[1].shape
        # print np.unique(vals[0]), np.unique(vals[1])
        # print np.sum(vals[0]==0),np.sum(vals[1]==0)

        # raw_input()
        # print inc, num_count, np.concatenate(counts[0]).shape

        file_str = ['inc',str(inc),str(num_count)]
        title = ' '.join(file_str)
        out_file  = '_'.join(file_str)+'.jpg'
        out_file = os.path.join(out_dir, out_file)

        legend_entries = ['Pain', 'No Pain']
        num_bins = range(int(np.max(vals[0]))+2)
        xtick_labels = [str(val) for val in num_bins]
        rotate = False
        xlabel = 'Number of AUs'
        ylabel = 'Frequency'
        visualize.plotMultiHist(out_file,vals = vals, num_bins = [num_bins, num_bins], legend_entries = legend_entries, title = title, xlabel = xlabel, ylabel = ylabel, xticks = xtick_labels, rotate = rotate, align = 'mid')

    visualize.writeHTMLForFolder(out_dir)

def finding_temporal_structure():
    core_aus = ['ad38',
                # 'au47',
                'au17',
                'au101',
                'ead101',
                'au145',
                'ead104',
                'ad54',
                'au24']
                # 'au18',
                # 'au10']
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict,remove_lr=True)
    all_aus = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])    

    core_aus = all_aus
    inc = 5
    out_dir = '../experiments/finding_temporal_structure_all_inc_'+str(inc)
    util.mkdir(out_dir)
    
    # step_size = 2.5
    core_aus = np.array(core_aus)

    keys_all = []
    aus_all = []
    ranks_all = []

    for key in key_arr:
        au_arr = np.array(data_dict[key][0])
        start_arr = np.array(data_dict[key][2])
        end_arr = np.array(data_dict[key][3])

        bin_rel = np.in1d(au_arr, core_aus)
        if np.sum(bin_rel)==0:
            continue
        au_arr = au_arr[bin_rel]
        start_arr = start_arr[bin_rel]
        end_arr = end_arr[bin_rel]

        start = 0
        
        while True:
            print start, np.max(end_arr)
            if start>np.max(end_arr):
                break

            end = start+inc
            
            rel_occurrences = np.logical_or(np.logical_and(start_arr<end,start_arr>=start), np.logical_and(end_arr<end,end_arr>start))
            rel_occurrences = np.logical_or(np.logical_and(start_arr<start,end_arr>end), rel_occurrences)
            
            if np.sum(rel_occurrences)==0:
                start = end
                continue

            start_rel = start_arr[rel_occurrences]
            end_rel = end_arr[rel_occurrences]
            ranks = np.argsort(start_rel - start)

            start = end
            # start_rel[np.argmax(end_rel - start)]
            # print start_rel
            # print end_rel
            # print start
            aus_all += list(au_arr[rel_occurrences])
            ranks_all += list(ranks)
            keys_all+=[key]*np.sum(rel_occurrences)
            # raw_input()

    keys_all = np.array(keys_all)
    aus_all = np.array(aus_all)
    ranks_all = np.array(ranks_all)
    bin_pain = np.in1d(keys_all, pain)
    for au_curr in core_aus:
        num_bins = np.max(ranks_all[aus_all==au_curr])
        pain_rank = ranks_all[np.logical_and(bin_pain, aus_all==au_curr)]
        no_pain_rank = ranks_all[np.logical_and(~bin_pain, aus_all==au_curr)]

        file_str = [au_curr]
        title = ' '.join(file_str)
        out_file = os.path.join(out_dir, '_'.join(file_str)+'.jpg')
        # if no_pain_rank.size==0:
        #     num_bins = range(np.max(pain_rank))
        # elif pain_rank.size==0:
        #     num_bins = range(np.max(no_pain_rank))
        # else:
        num_bins = range(num_bins+2)

        vals = [pain_rank, no_pain_rank]
        legend_entries = ['Pain', 'No Pain']
        xtick_labels = [str(val) for val in num_bins]
        rotate = False
        xlabel = 'Rank'
        ylabel = 'Frequency'
        visualize.plotMultiHist(out_file,vals = vals, num_bins = [num_bins, num_bins], legend_entries = legend_entries, title = title, xlabel = xlabel, ylabel = ylabel, xticks = xtick_labels, rotate = rotate, align = 'mid')

    visualize.writeHTMLForFolder(out_dir)

def main():
    print 'hello cooc'
    # finding_temporal_structure()
    # time_series_analysis()
    comparing_frequency_duration()
    # print 'hello'
    # script_plot_cooc()
    # find_best_clusters()
    # looking_at_time_series()

if __name__=='__main__':
    main()



