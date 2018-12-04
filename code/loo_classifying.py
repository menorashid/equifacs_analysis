import os
import numpy as np
import sklearn
from sklearn import preprocessing,decomposition,discriminant_analysis,linear_model,pipeline
from helpers import util, visualize
import clustering as cl
import read_in_data as reader
import numpy as np
import itertools

def loo_lda( data_types, norms, feat_keeps):
    dir_data = '../data'

    key_arr = range(1,13)
    no_pain = np.array([3,6,7,8,9,10])
    pain = np.array([1,2,4,5,11,12])

    out_dir = '../experiments/loo_lda_12'
    util.mkdir(out_dir)
    out_file = os.path.join(out_dir, 'temp.txt')
    
    to_print = []
    
    for (data_type, norm, feat_keep) in itertools.product(data_types, norms, feat_keeps):

        if feat_keep is None:
            str_curr = ' '.join([data_type, norm, 'all'])
        else:    
            str_curr = ' '.join([data_type, norm]+feat_keep)

        to_print.append(str_curr)
        print str_curr 

        data_dict, all_aus  = cl.get_all_data(dir_data)
        
        if feat_keep is not None:
            bin_au = np.zeros((len(feat_keep), len(all_aus)))
            for idx_val,val in enumerate(all_aus): 
                for idx_feat, feat in enumerate(feat_keep):
                    if feat in val:
                        bin_au[idx_feat, idx_val]=1
            bin_au = np.sum(bin_au,axis = 0)>0
            all_aus = [val for idx_val,val in enumerate(all_aus) if bin_au[idx_val]]
            assert len(all_aus)==np.sum(bin_au)

        bin_pain, bin_no_pain, class_pain, bin_au = get_all_bins(all_aus, key_arr, pain, no_pain, feat_keep)
        data_dict, all_aus  = cl.get_all_data(dir_data)
            
        all_counts = cl.get_data_by_type(data_dict, all_aus, key_arr, data_type)
        # print all_counts.shape
        
        train_scores = []
        test_scores = []
        
        for idx_out in range(all_counts.shape[0]):
            train_data, train_labels, test_data, test_labels = get_loo_data(all_counts, class_pain, idx_out)
            
            lda, scaler, train_data = cl.fit_lda(train_data, train_labels, norm)
            test_data = scaler.transform(test_data)
            train_pred = lda.predict_proba(train_data) 
            test_pred = lda.predict_proba(test_data)
            
            # print test_pred, test_labels

            train_scores.append(lda.score(train_data, train_labels))
            test_scores.append(lda.score(test_data, test_labels))
            
        # print lda.intercept_[0]
            
        str_curr = 'Train mean %.2f std %.2f' % (np.mean(train_scores), np.std(train_scores))
        to_print.append(str_curr)
        print str_curr 

        str_curr = 'Test mean %.2f std %.2f' % (np.mean(test_scores), np.std(test_scores))
        to_print.append(str_curr)
        print str_curr 

    util.writeFile(out_file, to_print)
    
def make_pipeline(log_reg_params, norm):
    pipeline_arr = []
    if 'l2' in norm:
        pipeline_arr.append(('l2',sklearn.preprocessing.Normalizer()))
    if 'mean_std' in norm:
        pipeline_arr.append(('mean_std',sklearn.preprocessing.StandardScaler()))
    elif 'mean' in norm:
        pipeline_arr.append(('mean',sklearn.preprocessing.StandardScaler(with_std = False)))
    log_reg = sklearn.linear_model.LogisticRegression(**log_reg_params)
    # print log_reg
    pipeline_arr.append(('log_reg',log_reg))
    pipe = sklearn.pipeline.Pipeline(pipeline_arr)    
    return pipe


def get_all_bins( all_aus, key_arr, pain, no_pain,feat_keep):
    bin_au = None
    if feat_keep is not None:
        bin_au = np.zeros((len(feat_keep), len(all_aus)))
        for idx_val,val in enumerate(all_aus): 
            for idx_feat, feat in enumerate(feat_keep):
                if feat in val:
                    bin_au[idx_feat, idx_val]=1
        bin_au = np.sum(bin_au,axis = 0)>0
        all_aus = [val for idx_val,val in enumerate(all_aus) if bin_au[idx_val]]
        assert len(all_aus)==np.sum(bin_au)

    bin_pain = np.in1d(key_arr, pain)
    bin_no_pain = np.in1d(key_arr, no_pain)
    
    class_pain = np.zeros(bin_pain.shape)
    class_pain[bin_pain] = 1
    class_pain = class_pain.astype(int)
    return bin_pain, bin_no_pain, class_pain, bin_au

def get_loo_data(all_counts, class_pain, idx_out):
    idx_keep = range(all_counts.shape[0])
    idx_keep.remove(idx_out)
    idx_keep = np.array(idx_keep)
    
    train_data = all_counts[idx_keep,:]
    train_labels = class_pain[idx_keep]
    
    test_data = all_counts[[idx_out],:]
    test_labels = class_pain[[idx_out]]
    return train_data, train_labels, test_data, test_labels
    

def loo_log_reg(log_reg_params, data_types, norms, feat_keeps):
    dir_data = '../data'

    key_arr = range(1,13)
    no_pain = np.array([3,6,7,8,9,10])
    pain = np.array([1,2,4,5,11,12])

    out_dir = '../experiments/loo_log_reg_12'
    util.mkdir(out_dir)
    print log_reg_params
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
        
        data_dict, all_aus  = cl.get_all_data(dir_data)
        bin_pain, bin_no_pain, class_pain, bin_au = get_all_bins(all_aus, key_arr, pain, no_pain, feat_keep)
        

        all_counts = cl.get_data_by_type(data_dict, all_aus, key_arr, data_type)
        if bin_au is not None:
            if 'both' in data_type:
                bin_au = np.concatenate((bin_au,bin_au))
            all_counts = all_counts[:,bin_au]
        

        train_scores = []
        test_scores = []
        test_preds = []
        
        for idx_out in range(all_counts.shape[0]):
            train_data, train_labels, test_data, test_labels = get_loo_data(all_counts, class_pain, idx_out)

            model = make_pipeline(log_reg_params, norm)
            model.fit(train_data,train_labels)
            train_pred = model.predict(train_data)
            test_pred = model.predict(test_data)
            
            accu_test = np.sum(test_pred==test_labels)/float(test_labels.size)
            accu_train = np.sum(train_pred==train_labels)/float(train_labels.size)
            
            test_preds.append(test_pred[0]) 
            train_scores.append(accu_train)
            test_scores.append(accu_test)

            # print 'idx_out'
            # print idx_out
            # print 'class_pain'
            # print class_pain
            # print 'train_labels'
            # print train_labels
            # print 'train_pred'
            # print train_pred
            
            # print 'class_pain[idx_out]'
            # print class_pain[idx_out]
            # print 'test_labels'
            # print test_labels
            # print 'test_pred'
            # print test_pred
            
            # print 'accu_test'
            # print accu_test
            # print 'accu_train'
            # print accu_train

            
            # raw_input()

        
        # print test_preds
        # print class_pain
        # print test_scores

        str_curr = 'Train mean %.2f std %.2f' % (np.mean(train_scores), np.std(train_scores))
        to_print.append(str_curr)
        print str_curr 

        str_curr = 'Test mean %.2f std %.2f' % (np.mean(test_scores), np.std(test_scores))
        to_print.append(str_curr)
        print str_curr 

    util.writeFile(out_file, to_print)
        # penalty='l2', dual=True)

def main():
    # data_type = 'duration_normalized'
    # feat_keep = ['au']
    # norm = 'mean_std'
    
    data_types=['frequency', 'duration', 'both', 'both_normalized','duration_normalized']
    norms = ['l2_mean_std']
    feat_keeps = [None,['au'],['au','ad']]
    log_reg_params = {'penalty':'l2','dual': False, 'fit_intercept': False, 'class_weight':'balanced'}

    # loo_lda(data_types, norms, feat_keeps)
    loo_log_reg(log_reg_params, data_types, norms, feat_keeps)

    
    
        # print data_type, norm, feat_keep


if __name__=='__main__':
    main()
