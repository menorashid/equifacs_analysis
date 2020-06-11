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
    
def make_pipeline(log_reg_params, norm, model_type = 'log_reg'):
    pipeline_arr = []
    if 'l2' in norm:
        pipeline_arr.append(('l2',sklearn.preprocessing.Normalizer()))
    if 'mean_std' in norm:
        pipeline_arr.append(('mean_std',sklearn.preprocessing.StandardScaler()))
    elif 'mean' in norm:
        pipeline_arr.append(('mean',sklearn.preprocessing.StandardScaler(with_std = False)))
    
    if model_type =='log_reg':
        log_reg = sklearn.linear_model.LogisticRegression(**log_reg_params)
    elif model_type =='knn':
        log_reg = sklearn.neighbors.NearestNeighbors(**log_reg_params)
    elif model_type is None:
        log_reg = None
    else:
        raise ValueError('not a valid model type '+str(model_type))
    # print log_reg
    if log_reg is not None:
        pipeline_arr.append(('log_reg',log_reg))

    pipe = sklearn.pipeline.Pipeline(pipeline_arr)    
    return pipe

def prune_features(features, all_aus, feat_keep):
    if feat_keep is None:
        return features, all_aus, None

    if feat_keep[0]=='pain':
        aus_keep = ['au101','au17','au24','ead104','ad38']
        bin_au = [1 if au_curr in aus_keep else 0 for au_curr in all_aus]
        bin_au = np.array(bin_au)>0
        # all_aus = [val for idx_val,val in enumerate(all_aus) if bin_au[idx_val]]

    elif feat_keep[0]=='exact':
        feat_keep = feat_keep[1:]
        all_aus = list(all_aus)
        bin_au = [all_aus.index(val) for val in feat_keep]
        bin_au = np.array(bin_au)
        # print feat_keep
        # all_aus =  list(np.array(all_aus)[bin_au])
        # print all_aus
        # raw_input()
        # bin_au = [1 if au_curr in feat_keep[1:] else 0 for au_curr in all_aus]

        # bin_au = np.array(bin_au)>0
    else:
        bin_au = np.zeros((len(feat_keep), len(all_aus)))
        for idx_val,val in enumerate(all_aus): 
            for idx_feat, feat in enumerate(feat_keep):
                if feat in val:
                    bin_au[idx_feat, idx_val]=1
        bin_au = np.sum(bin_au,axis = 0)>0
        # all_aus = [val for idx_val,val in enumerate(all_aus) if bin_au[idx_val]]
    
    # print all_aus
    
    # print 'we are pruning'
    aus_no = ['ad51', 'ad52', 'ad53', 'ad54','ad55','ad56', 'ad57', 'ad58', 'ad84','ad85']
    # raw_input()
    bin_no_keep = np.in1d(all_aus, aus_no)
    bin_au = np.logical_and(bin_au, np.logical_not(bin_no_keep))
    all_aus = list(np.array(all_aus)[bin_au])
    # assert len(all_aus)==np.sum(bin_au)    
    # print all_aus

    # raw_input()
    if features is not None:
        features = features[:,bin_au]
    
    # print all_aus
    # print feat_keep
    # print features.shape

    return features, all_aus, bin_au


def get_all_bins( all_aus, key_arr, pain, no_pain,feat_keep):
    _, _, bin_au = prune_features(None, all_aus, feat_keep) 

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
    out_dir_plots = out_file[:out_file.rindex('.')]
    util.mkdir(out_dir_plots)

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

        if 'both' in data_type:
            all_aus = [au_curr+' f' for au_curr in all_aus]+[au_curr+' d' for au_curr in all_aus]

        if bin_au is not None:
            if 'both' in data_type:
                bin_au = np.concatenate((bin_au,bin_au))
                # all_aus = [au_curr+' f' for au_curr in all_aus]+[au_curr+' d' for au_curr in all_aus]

            all_counts = all_counts[:,bin_au]
            all_aus = np.array(all_aus)[bin_au]

        train_scores = []
        test_scores = []
        test_preds = []
        models = []
        train_p_features = []
        train_np_features = []
        for idx_out in range(all_counts.shape[0]):
            train_data, train_labels, test_data, test_labels = get_loo_data(all_counts, class_pain, idx_out)

            model = make_pipeline(log_reg_params, norm)
            model.fit(train_data,train_labels)
            train_pred = model.predict(train_data)
            test_pred = model.predict(test_data)
            
            train_features = train_data
            for step in model.steps[:-1]:
                train_features = step[1].transform(train_features)
            train_p_features.append(train_features[train_labels>0,:])
            train_np_features.append(train_features[train_labels==0,:])

            accu_test = np.sum(test_pred==test_labels)/float(test_labels.size)
            accu_train = np.sum(train_pred==train_labels)/float(train_labels.size)
            
            test_preds.append(test_pred[0]) 
            train_scores.append(accu_train)
            test_scores.append(accu_test)
            models.append(model)


        str_curr = 'Train mean %.2f std %.2f' % (np.mean(train_scores), np.std(train_scores))
        to_print.append(str_curr)
        print str_curr 

        str_curr = 'Test mean %.2f std %.2f' % (np.mean(test_scores), np.std(test_scores))
        to_print.append(str_curr)
        print str_curr 

        if feat_keep is None:
            out_dir_curr = '_'.join([data_type, norm, 'all'])
        else:    
            out_dir_curr = '_'.join([data_type, norm]+feat_keep)

        out_dir_curr = os.path.join(out_dir_plots,out_dir_curr)
        util.mkdir(out_dir_curr)
                
        weights = [model.named_steps['log_reg'].coef_[0] for model in models]
        average_p_features = [np.mean(features_curr,axis = 0) for features_curr in train_p_features]
        average_np_features = [np.mean(features_curr,axis = 0) for features_curr in train_np_features]

        weights.append(np.mean(np.array(weights),axis = 0))
        average_p_features.append(np.mean(np.array(average_p_features),axis = 0))
        test_scores.append(np.mean(test_scores))

        legend_entries = ['Fold %d %.2f'%(fold_num, test_scores[fold_num]) for fold_num in range(len(test_scores)-1)]+['Average %.2f'%(test_scores[-1])]
        
        titles = ['Weights', 'Weights Pain RW','Weights No Pain RW']
        weights_pain = [w*average_p_features[w_idx] for w_idx,w in enumerate(weights[:-1])]
        weights_no_pain = [w*average_np_features[w_idx] for w_idx,w in enumerate(weights[:-1])]
        weights_all = [weights, weights_pain, weights_no_pain]
        for title,weights_curr in zip(titles,weights_all):
        # title = 'Weights'
        # weights_curr = weights
            file_str = title.lower().replace(' ','_')
            out_file_curr = os.path.join(out_dir_curr,file_str+'.jpg')
            
            x_vals = range(len(all_aus))
            xAndYs = [(x_vals,weight_curr) for weight_curr in weights_curr]
            xlabel = 'AUs'
            ylabel = 'Weights'

            # print all_aus
            visualize.plotSimple(xAndYs,out_file=out_file_curr,title=title,xlabel=xlabel,ylabel=ylabel,legend_entries=legend_entries[:len(xAndYs)],loc=0,outside=True,logscale=False,xticks=all_aus)
            # print out_file_curr

        visualize.writeHTMLForFolder(out_dir_curr,height = 746, width = 485)
        # visualize.plotSimple(
        # print test_scores

        # dict_vals = {}

        # for weight,features_p,features_np,test_score in zip(weights, average_p_features, average_np_features,test_scores):
        #     if idx_legend<len(weights)-1:
        #         legend_curr = 'Fold '+str(idx_legend+1)
        #     weight = weight[0]
        #     print weight.shape
        #     print features_p.shape
        #     print features_np.shape
        #     print all_aus
        #     print test_score
        #     raw_input()




        # print models[0].named_steps['log_reg'].intercept_
        # raw_input()

    util.writeFile(out_file, to_print)
        # penalty='l2', dual=True)

def main():
    # data_type = 'duration_normalized'
    # feat_keep = ['au']
    # norm = 'mean_std'
    
    data_types=['frequency', 'duration','duration_normalized','both','both_normalized']
    norms = ['l2_mean_std']
    feat_keeps = [['pain']]
    log_reg_params = {'penalty':'l1','dual': False, 'fit_intercept': False, 'class_weight':'balanced'}

    # loo_lda(data_types, norms, feat_keeps)
    loo_log_reg(log_reg_params, data_types, norms, feat_keeps)

    
    
        # print data_type, norm, feat_keep


if __name__=='__main__':
    main()
