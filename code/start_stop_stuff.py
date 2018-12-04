import sklearn
from sklearn import preprocessing,decomposition,discriminant_analysis,pipeline
from helpers import util, visualize
from read_in_data import *
import itertools
import loo_classifying as lc

def get_start_stop_feat(data_dict, all_aus, key_arr, inc, data_type, feat_keep = None, vid_length = 30.):

    features = []
    all_aus = np.array(all_aus)
    labels = []
    # np.zeros((len(key_arr),len(all_aus)))
    for idx_k, k in enumerate(key_arr):
        au_anno = np.array(data_dict[k][0])
        start = np.array(data_dict[k][2])
        end = np.array(data_dict [k] [3])
        
        end[end>vid_length]=vid_length
        assert start.size == end.size
        # num_steps = int(vid_length/inc)
        inc_pts = list(np.arange(0,vid_length,inc))
        inc_pts = inc_pts+[vid_length]
        
        mat_curr = np.zeros((len(inc_pts)-1, len(all_aus)))
        for idx_inc, inc_start in enumerate(inc_pts[:-1]):
            inc_end = inc_pts[idx_inc+1]
            bin_ex = end<inc_start
            bin_ex = np.logical_or(bin_ex,start>=inc_end)
            bin_ex = bin_ex<1


            anno_curr = au_anno[bin_ex]

            start_curr = start[bin_ex]
            end_curr = end[bin_ex]
                
            if data_type=='binary':
                idx_aus = np.in1d(all_aus,anno_curr)
                mat_curr[idx_inc,idx_aus] = 1
            elif data_type=='frequency':
                for au_curr in anno_curr:
                    bin_rel = all_aus==au_curr
                    assert np.sum(bin_rel)==1
                    mat_curr[idx_inc,bin_rel]+=1
            elif 'duration' in data_type:
                
                start_curr[start_curr<inc_start]=inc_start
                end_curr[end_curr>=inc_end]=inc_end

                durations = end_curr - start_curr

                for au_curr in np.unique(anno_curr):
                    bin_rel = all_aus==au_curr
                    assert np.sum(bin_rel)==1
                    if data_type =='duration':
                        mat_curr[idx_inc, bin_rel]=np.sum(durations[anno_curr==au_curr])
                    elif data_type=='max_duration':
                        mat_curr[idx_inc, bin_rel]=np.max(durations[anno_curr==au_curr])
                    elif data_type=='min_duration':
                        mat_curr[idx_inc, bin_rel]=np.min(durations[anno_curr==au_curr])
                    elif data_type=='mean_duration':
                        mat_curr[idx_inc, bin_rel]=np.mean(durations[anno_curr==au_curr])
                    else:
                        raise ValueError('not a valid data type '+str(data_type))
            else:
                raise ValueError('not a valid data type '+str(data_type))

            
        
        features.append(mat_curr)
        labels = labels+[k]*mat_curr.shape[0]        

    features = np.concatenate(features, axis = 0)
    labels = np.array(labels)

    if feat_keep is not None:
        features, all_aus = prune_features(features, all_aus, feat_keep)
    
    return features, labels, all_aus

def prune_features(features, all_aus, feat_keep):
    if feat_keep is not None:
        bin_au = np.zeros((len(feat_keep), len(all_aus)))
        for idx_val,val in enumerate(all_aus): 
            for idx_feat, feat in enumerate(feat_keep):
                if feat in val:
                    bin_au[idx_feat, idx_val]=1
        bin_au = np.sum(bin_au,axis = 0)>0
        all_aus = [val for idx_val,val in enumerate(all_aus) if bin_au[idx_val]]
        assert len(all_aus)==np.sum(bin_au)    
        features = features[:,bin_au]
    return features, all_aus


def count_cooc(features, all_aus, feat_keep):
    if feat_keep is not None:
        bin_au = np.zeros((len(feat_keep), len(all_aus)))
        for idx_val,val in enumerate(all_aus): 
            for idx_feat, feat in enumerate(feat_keep):
                if feat in val:
                    bin_au[idx_feat, idx_val]=1
        bin_au = np.sum(bin_au,axis = 0)>0
        all_aus = [val for idx_val,val in enumerate(all_aus) if bin_au[idx_val]]
        assert len(all_aus)==np.sum(bin_au)    
        features = features[:,bin_au]

    cooc_bin = np.zeros((len(all_aus),len(all_aus)))
    for feat_curr in features:
        idx_au = np.sort(np.where(feat_curr>0)[0])
        for idx_idx, idx_curr in enumerate(idx_au[:-1]):
            for idx_match in idx_au[idx_idx+1:]:
                cooc_bin[idx_curr,idx_match]+=1
                cooc_bin[idx_match,idx_curr]+=1
    
    sums = np.sum(features,axis = 0, keepdims = True)
    return cooc_bin, sums, all_aus


def script_plot_cooc():
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict)
    all_aus = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])
    # bin_pain = np.in1d(np.array(key_arr),pain)
    # print bin_pain

    data_keeps = ['pain','no_pain','all']
    feat_keeps = [['au','ad']]
    out_dir_meta = '../experiments/visualizing_cooc_12'
    util.mkdir(out_dir_meta)
    inc_range = [5]
    # 5,10,15,30]

    for inc in inc_range:
        features, labels,_ = get_start_stop_feat(data_dict, all_aus, key_arr, inc, 'binary')
        print inc, features.shape
        for data_keep, feat_keep in itertools.product(data_keeps,feat_keeps):

            out_dir = os.path.join(out_dir_meta,'_'.join(feat_keep+[str(inc)]))
            util.mkdir(out_dir)
            
            bin_pain = np.in1d(labels,pain)

            if data_keep=='pain':
                features_curr = features[bin_pain,:]
            elif data_keep=='no_pain':
                features_curr = features[~bin_pain,:]
            else:
                features_curr = features

            cooc_bin, sums, classes = count_cooc(features_curr, all_aus, feat_keep)
            sums[sums==0]=1
            sums = sums.T

            cooc_norm = cooc_bin/sums

            figsize = (0.5*cooc_bin.shape[0]+0.5,0.5*cooc_bin.shape[1]-0.5)
            # cooc_bin = cooc_bin[:-1,:]

            file_str = [data_keep,str(int(inc)),'seconds']+feat_keep
            title = ' '.join([val.title() for val in file_str])
            out_file = os.path.join(out_dir,'_'.join(file_str)+'.jpg')
            visualize.plot_confusion_matrix(cooc_bin.astype(int), classes, out_file,normalize=False,title=title,ylabel = '', xlabel = '', figsize = figsize)

            file_str += ['normalized']
            title = ' '.join([val.title() for val in file_str])
            out_file = os.path.join(out_dir,'_'.join(file_str)+'.jpg')
            visualize.plot_confusion_matrix(cooc_norm, classes, out_file,normalize=False,title=title,ylabel = '', xlabel = '',fmt = '.2f', figsize = figsize)

            visualize.writeHTMLForFolder(out_dir,height = figsize[1]*100,width = figsize[0]*100)
            # raw_input()

def transform_labels(labels, train_labels, train_pred, test_scheme):
    org_labels = []
    org_pain = []
    pred = []

    for vid_num in np.unique(labels):
        bin_labels = labels == vid_num

        org_labels.append(vid_num)
        org_bin = np.unique(train_labels[bin_labels])
        assert org_bin.size==1
        org_pain.append(org_bin[0])

        pred_bin = train_pred[bin_labels]
        if test_scheme=='majority':
            if np.sum(pred_bin==1)>=np.sum(pred_bin==0):
                pred.append(1)
            else:
                pred.append(0)    
            # pred.append(np.argmax(np.array([np.sum(pred_bin==val) for val in [0,1]])))
        elif test_scheme=='atleast_one':
            val = np.sum(pred_bin)
            val = 1 if val>=1 else val 
            pred.append(val)
        else:
            raise ValueError('Bad test scheme '+str(test_scheme))

    org_labels = np.array(org_labels)
    org_pain = np.array(org_pain)
    pred = np.array(pred)
    
    # print test_scheme
    # print labels
    # print org_labels
    # print '___'
    # print train_labels
    # print org_pain
    # print '___'
    # print train_pred
    # print pred
    # print '___'

    # raw_input()
    return org_labels, org_pain, pred



def loo_log_reg():
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict)
    all_aus_org = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])
    inc = 30
    
    test_scheme = None
    train_scheme = None
    out_dir = '../experiments/loo_log_reg_'+str(inc)+'_test_'+str(test_scheme)+'_train_'+str(train_scheme)
    util.mkdir(out_dir)
    
    feat_keeps = [None,['au','ad']]    
    data_types = ['binary','frequency','duration', 'max_duration', 'min_duration','mean_duration']
    norms = ['l2_mean_std']
    log_reg_params = {'penalty':'l2','dual': False, 'fit_intercept': False}
    
    log_reg_str = []
    for k in log_reg_params:
        log_reg_str.append(k)
        log_reg_str.append(log_reg_params[k])
    log_reg_str = '_'.join([str(val) for val in log_reg_str])
    out_file = os.path.join(out_dir, 'results_'+log_reg_str+'.txt')
    

    to_print = []
    for (data_type, norm, feat_keep) in itertools.product(data_types, norms, feat_keeps):

        if feat_keep is None:
            str_curr = ' '.join([data_type, norm, 'all'])
        else:    
            str_curr = ' '.join([data_type, norm]+feat_keep)

        to_print.append(str_curr)
        print str_curr 
        
        features, labels, all_aus = get_start_stop_feat(data_dict, all_aus_org, key_arr, inc, data_type, feat_keep = feat_keep)
        # print inc, features.shape

        bin_pain = np.in1d(labels, pain)
    
        class_pain = np.zeros(bin_pain.shape)
        class_pain[bin_pain]=1
        class_pain = class_pain.astype(int)

        train_scores = []
        test_scores = []
        test_preds = []
        
        for vid_test in key_arr:
            bin_test = labels == vid_test
            
            bin_train = ~bin_test
            train_data = features[bin_train,:]
            train_labels = class_pain[bin_train]

            if train_scheme is not None:
                # print train_data.shape
                count = np.sum(train_data>0,axis = 1)
                # print train_labels
                # print count
                count = count>=train_scheme
                # print count
                train_labels[~count]=0
                # print train_labels
                # raw_input()

            # print np.sum(train_labels), train_labels.shape
            test_data = features[bin_test,:]
            test_labels = class_pain[bin_test]
            
            # print np.sum(test_labels), test_labels.shape

            model = lc.make_pipeline(log_reg_params, norm)
            model.fit(train_data,train_labels)

            train_pred = model.predict(train_data)
            test_pred = model.predict(test_data)

            if test_scheme is not None:
                # train_vids, train_labels, train_pred = transform_labels(labels[bin_train], train_labels, train_pred, test_scheme)
                test_vids, test_labels, test_pred = transform_labels(labels[bin_test], test_labels, test_pred, test_scheme)
            
            
            
            accu_test = np.sum(test_pred==test_labels)/float(test_labels.size)
            accu_train = np.sum(train_pred==train_labels)/float(train_labels.size)
            
            # print 'test labels pred'
            # print list(test_labels)
            # print list(test_pred)
            # print accu_test
            # print 'train labels pred'
            # print list(train_labels)
            # print list(train_pred)
            # print accu_train

            # raw_input()
            
            test_preds.append(test_pred) 
            train_scores.append(accu_train)
            test_scores.append(accu_test)

        
        str_curr = 'Train mean %.2f std %.2f' % (np.mean(train_scores), np.std(train_scores))
        to_print.append(str_curr)
        print str_curr 

        str_curr = 'Test mean %.2f std %.2f' % (np.mean(test_scores), np.std(test_scores))
        to_print.append(str_curr)
        print str_curr 

    util.writeFile(out_file, to_print)



def main():
    
    loo_log_reg()

    return


    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict)
    all_aus = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])
    # bin_pain = np.in1d(np.array(key_arr),pain)
    # print bin_pain

    data_keeps = ['pain','no_pain','all']
    feat_keeps = [['au','ad']]
    out_dir_meta = '../experiments/visualizing_cooc_12'
    util.mkdir(out_dir_meta)
    inc = 5
    data_types = ['binary','frequency','duration']
    # 5,10,15,30]

    for data_type in data_types:
        features, labels = get_start_stop_feat(data_dict, all_aus, key_arr, inc, data_type)
        print inc, features.shape

    

if __name__=='__main__':
    main()