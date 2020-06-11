from helpers import util, visualize
from script_cumulative_separation import get_feats
import numpy as np
import os
import math
import clustering as cl
# from clustering import fit_model, bootstrap
import cooc
import itertools
import sklearn.metrics
import feature_selectors
import copy
import print_tables as pt
from doing_ml_overflow import *

def select_features( selection_params,selection_type = 'cooc', clinical = False):    
    # print selection_params
    # print selection_type
    # raw_input()
    if selection_type is None:
        selector = feature_selectors.Base_Selector(**selection_params)
            # inc, step_size, feature_type, pain)
    elif selection_type.startswith('cooc'):
        # thresh = float(selection_type.split('_')[1])
        selector = feature_selectors.Cooc_Selector(**selection_params)
            # inc, step_size, feature_type, pain, thresh)
    elif selection_type=='kunz':
        # thresh = 0.05
        selector = feature_selectors.Kunz_Selector(**selection_params)
            # inc, step_size, feature_type, pain)
    elif selection_type=='collapse':
        selector = feature_selectors.Collapsed_Selector(**selection_params)
            # inc, step_size, feature_type, pain)
    # elif selection_type.startswith('collapse'):
    #     sthresh,flicker,blink = [float(val) for val in selection_type

    # elif cluster:
    if clinical:
        return selector.select_exp_clinical()
    else:

        return selector.select_and_split()

def get_labels_for_eval(gt, pred, vid_labels, eval_method = 'raw'):
    if eval_method=='raw':
        pass
    elif eval_method == 'majority':
        new_labels = [[],[]]
        for vid_label in np.unique(vid_labels):
            bin_rel = vid_labels==vid_label
            for idx_pain_label,pain_label in enumerate([gt,pred]):
                pain_rel = pain_label[bin_rel]
                if idx_pain_label==0:
                    assert (np.all(pain_rel==1) or np.all(pain_rel==0))

                # print pain_rel, 
                pain_rel = np.sum(pain_rel)/float(pain_rel.size)
                # print pain_rel, 
                new_labels[idx_pain_label].append(int(pain_rel>=0.5))
                # print new_labels[idx_pain_label][-1]
        new_labels = [np.array(arr) for arr in new_labels]
        [gt, pred] = new_labels
    elif eval_method == 'atleast_1':
        new_labels = [[],[]]
        for vid_label in np.unique(vid_labels):
            bin_rel = vid_labels==vid_label
            for idx_pain_label,pain_label in enumerate([gt,pred]):
                pain_rel = pain_label[bin_rel]
                if idx_pain_label==0:
                    assert (np.all(pain_rel==1) or np.all(pain_rel==0))
                pain_rel = int(np.sum(pain_rel)>0)
                # /float(pain_rel.size)
                # print pain_rel, 
                new_labels[idx_pain_label].append(pain_rel)
                # print new_labels[idx_pain_label][-1]
        new_labels = [np.array(arr) for arr in new_labels]
        [gt, pred] = new_labels
    else:
        raise ValueError('eval_method '+str(eval_method)+' is not valid')

    return gt, pred      

# def script_loo(out_dir_meta, ows, feature_types,  feature_selection = None, selection_params={}, eval_methods = ['raw','majority','atleast_1'], norm = 'mean_std', model_type = None, model_params = None, bootstrap = False):

#     # pain = np.array([1,2,4,5,11,12,13,14,15,16,17,18,19,20])
#     selection_params = copy.deepcopy(selection_params)
#     # selection_params['pain']=pain

#     iterator = itertools.product(ows,feature_types)

#     table_curr = np.zeros((len(ows), len(feature_types)))
#     row_labels = ows
#     col_labels = feature_types

#     for ows_curr, feature_type in iterator:
#         inc,step_size = ows_curr

#         # dir_str = '_'.join([str(val) for val in ['inc',inc,'step',step_size]])
#         # feature_type_str = '_'.join(feature_type)
#         # out_dir = os.path.join(out_dir_meta,dir_str,feature_type_str)
#         # print out_dir
#         # util.mkdir(out_dir)
        
#         selection_params['inc']=inc
#         selection_params['step_size']=step_size
#         selection_params['feature_type']=feature_type

#         features, labels, all_aus, class_pain, idx_test_all, bin_keep_aus = select_features(selection_type = feature_selection, selection_params = selection_params)
#         bin_something = np.sum(features, axis = 1)>0

#         # class_pain = np.in1d(labels, pain).astype(int)
#         # print class_pain
#         # print class_pain_new
#         # raw_input()
#         loo_results_train = []
#         loo_results_test = []

#         for eval_method in eval_methods:
#             preds = [[],[]]
#             gts = [[],[]]

#             for idx_test, bin_aus_keep in zip(idx_test_all, bin_keep_aus):
#                 idx_train = np.logical_not(idx_test)
#                 features_train  = features[idx_train,:]
#                 features_train = features_train[:,bin_aus_keep]
#                 labels_train = class_pain[idx_train]
#                 if bootstrap:
#                     k =  max(1,features_train.shape[0]//8)
#                     bin_keep = cl.bootstrap(features_train, labels_train, norm, k)
#                     features_train = features_train[bin_keep,:]
#                     labels_train = labels_train[bin_keep]

#                 if model_type.startswith('knn'):
#                     k = max(1,features_train.shape[0]//int(model_type.split('_')[-1]))
#                     model_params['n_neighbors'] = k

#                 lda, scaler, data_lda = cl.fit_model(features_train, labels_train, norm = norm, model_type = model_type, model_params = model_params)

#                 bins = [idx_train, idx_test]
                

#                 for idx_loo,bin_rel in enumerate(bins):
#                     features_curr = features[bin_rel,:]
#                     features_curr = features_curr[:,bin_aus_keep]
#                     gts_curr = class_pain[bin_rel]
#                     vid_labels_curr = labels[bin_rel]
#                     preds_curr = lda.predict(scaler.transform(features_curr))
                    
#                     gts_curr, preds_curr = get_labels_for_eval(gts_curr, preds_curr, vid_labels_curr, eval_method)
                    
#                     gts[idx_loo].append(gts_curr) 
#                     preds[idx_loo].append(preds_curr)

#             gts = [np.concatenate(gt_curr, axis = 0) for gt_curr in gts]
#             preds = [np.concatenate(gt_curr, axis = 0) for gt_curr in preds]
#             to_print = ['train','test']
            
#             for idx,(gt,pred) in enumerate(zip(gts, preds)):
#                 if idx==0:
#                     continue
#                 precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(gt, pred, average = 'binary')
#                 accuracy = np.sum(gt ==pred)/float(gt.size)
#                 # print to_print[idx],eval_method, precision, recall, f1
#                 row_idx = row_labels.index(ows_curr)
#                 # print row_idx, row_labels, ows_curr
#                 col_idx = col_labels.index(feature_type)
#                 # print col_idx, col_labels, feature_type
#                 table_curr[row_idx, col_idx] = f1
#                 # raw_input()
#     # print table_curr
#     table_curr = table_curr*100
#     str_row = ['OWS']+[' & '.join([str_curr.title() for str_curr in features_curr]) for features_curr in col_labels]
#     print ','.join(str_row)    
#     for row_idx, row in enumerate(range(table_curr.shape[0])):
#         str_row = []
#         str_row.append(str(row_labels[row_idx][0]))
#         # print row_labels[row_idx][0] ',',
#         for col_idx, col in enumerate(range(table_curr.shape[1])):
#             str_curr = '%.2f'%table_curr[row_idx,col_idx]
#             str_row.append(str_curr)

#         print ','.join(str_row)

#     return table_curr

#                 # , accuracy
#                 # , accuracy
#                 # print to_print[idx],eval_method, f1        
#         # aps = [sklearn.metrics.average_precision_score(gts[idx],preds[idx][:,1]) for idx in range(len(gts))]

#         # aps = [sklearn.metrics.precision_recall_fscore_support(gts[idx],preds[idx], average= 'binary') for idx in range(len(gts))]
#         # print aps

def evaluate(probs, test_labels, test_videos, eval_method):
    prob_vals = np.unique(probs)[::-1]
    prob_vals = np.arange(0.,1.1,0.1)
    prec =[]
    recall = []
    printed =0
    for prob in prob_vals:
        preds = np.zeros(probs.shape)
        preds[probs>=prob] = 1
        gt_combo, pred_combo = get_labels_for_eval(test_labels, preds, test_videos, eval_method)
        
        if np.sum(pred_combo)==0:
            continue
        
        prec_curr, recall_curr, _, _= sklearn.metrics.precision_recall_fscore_support(gt_combo, pred_combo, average = 'binary')
        prec.append(prec_curr)
        recall.append(recall_curr)

    auc = sklearn.metrics.auc(recall, prec)
    # print prec
    # print recall
    # print auc
    # print prob_vals
    # raw_input()
    # gts_curr, preds_curr = get_labels_for_eval(gts_curr, preds_curr, vid_labels_curr, eval_method)
    # gts[idx_loo] = gts_curr 
    # preds[idx_loo] = preds_curr

    # to_print = ['train','test']
    # for idx,(gt,pred) in enumerate(zip(gts, preds)):
    # if idx==0:
    # continue
    # precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(gt, pred, average = 'binary')
    #     print to_print[idx],eval_method, precision, recall, f1



def test_clinical(out_dir_meta, ows, feature_types,  feature_selection = None, selection_params={}, eval_methods = ['raw','majority','atleast_1'], norm = 'mean_std', model_type = 'lda', model_params = None, bootstrap = False):

    # pain = np.array([1,2,4,5,11,12])
    # pain = np.array([1,2,4,5,11,12,13,14,15,16,17,18,19,20])
    
    selection_params = copy.deepcopy(selection_params)
    # selection_params['pain']=pain

    iterator = itertools.product(ows,feature_types)
    table_curr = np.zeros((len(ows), len(feature_types)))
    row_labels = ows
    col_labels = feature_types

    bin_keep_aus_all = []
    all_aus_model = None
    coefs = []
    for ows_curr, feature_type in iterator:
        inc,step_size = ows_curr

        dir_str = '_'.join([str(val) for val in ['inc',inc,'step',step_size]])
        feature_type_str = '_'.join(feature_type)
        out_dir = os.path.join(out_dir_meta,dir_str,feature_type_str)
        # print out_dir
        # util.mkdir(out_dir)
        
        selection_params['inc']=inc
        selection_params['step_size']=step_size
        selection_params['feature_type']=feature_type

        train_package, test_package = select_features(selection_type = feature_selection, selection_params = selection_params, clinical = True)

        train_labels = np.in1d(train_package['labels'], train_package['pain']).astype(int)
        train_features = train_package['features'][:,train_package['bin_keep_aus']]
        test_features = test_package['features'][:,test_package['bin_keep_aus']]
        test_labels = np.mean(test_package['pain'], axis = 1)/2.
        # print test_labels
        test_labels_raw = np.array(test_labels)
        test_labels[test_labels>=0.5] = 1
        test_labels[test_labels<0.5] = 0
        # print test_labels.shape
        # raw_input()
        # print train_package['all_aus'][train_package['bin_keep_aus']]
        # if all_aus_model is None:
        #     all_aus_model = train_package['all_aus']
        # # print all_aus_model, train_package['all_aus']
        # assert np.all(all_aus_model==train_package['all_aus'])
        bin_keep_aus_all.append(train_package['bin_keep_aus'])

        if bootstrap:
            k = max(1,train_features.shape[0]//8)
            # k = 1
            bin_keep = cl.bootstrap(train_features, train_labels, norm, k)
            train_features = train_features[bin_keep,:]
            train_labels = train_labels[bin_keep]
            train_package['labels'] = train_package['labels'][bin_keep]


            # train_labels = cl.bootstrap(train_features, train_labels, norm, max(1,train_features.shape[0]//8))


        if model_type.startswith('knn'):
            k = max(1,train_features.shape[0]//int(model_type.split('_')[-1]))
            model_params['n_neighbors'] = k
            # k = 1

        lda, scaler, data_lda = cl.fit_model(train_features, train_labels, norm = norm, model_type = model_type, model_params = model_params)
        
        if not model_type.startswith('knn'):
            coef_curr = lda.coef_[0]
            coef_block = np.zeros(train_package['all_aus'].shape)
            coef_block[train_package['bin_keep_aus']] = coef_curr
            coefs.append(coef_block)

        features = [train_features, test_features]
        labels = [np.in1d(train_package['labels'], train_package['pain']).astype(int), test_labels]
        vid_labels = [train_package['labels'], test_package['labels']]


        for eval_method in eval_methods:
            preds = [[],[]]
            probs = [[],[]]
            gts = [[],[]]
            for idx_loo,features_curr in enumerate(features):
                gts_curr = labels[idx_loo]
                vid_labels_curr = vid_labels[idx_loo]
                preds_curr = lda.predict(scaler.transform(features_curr))
                # probs_curr = lda.predict_proba(scaler.transform(features_curr))
                # probs_curr = probs_curr[:,1]
                
                # (lda.predict(scaler.transform(features_curr))>0.5).astype(int)
                # print gts_curr, preds_curr
                gts_curr, preds_curr = get_labels_for_eval(gts_curr, preds_curr, vid_labels_curr, eval_method)
                gts[idx_loo] = gts_curr 
                preds[idx_loo] = preds_curr
                # if idx_loo==1:
                #     evaluate(probs_curr, labels[idx_loo], vid_labels_curr, eval_method)
            


            to_print = ['train','test']
            for idx,(gt,pred) in enumerate(zip(gts, preds)):
                if idx==0:
                    continue
                # print gt
                # print pred
                precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(gt, pred, average = 'binary')
                # print to_print[idx],eval_method, precision, recall, f1
                row_idx = row_labels.index(ows_curr)
                # print row_idx, row_labels, ows_curr
                col_idx = col_labels.index(feature_type)
                # print col_idx, col_labels, feature_type
                table_curr[row_idx, col_idx] = f1

    table_curr = table_curr*100
    str_row = ['OWS']+[' & '.join([str_curr.title() for str_curr in features_curr]) for features_curr in col_labels]
    print ','.join(str_row)    
    for row_idx, row in enumerate(range(table_curr.shape[0])):
        str_row = []
        str_row.append(str(row_labels[row_idx][0]))
        # print row_labels[row_idx][0] ',',
        for col_idx, col in enumerate(range(table_curr.shape[1])):
            str_curr = '%.2f'%table_curr[row_idx,col_idx]
            str_row.append(str_curr)

        print ','.join(str_row)

    return table_curr
    # bin_keep_aus_all, all_aus_model,coefs


def get_prob_frame_video():
    inc = 30
    step_size = 30
    data_type = 'frequency'
    type_dataset = 'isch'

    ows = [[2,1],[5,2.5],[10,5],[15,7.5],[20,10],[30,30]]
    # [0.04,0.04]]
    # 
    to_rec = [[],[]]
    for [inc, step_size] in ows:
        features, labels, all_aus, pain =get_feats(inc, step_size, data_type = data_type, clinical = False,type_dataset = type_dataset)
        
        # arr = np.load('temp_features.npz')
        # [features, labels, pain, all_aus] = [arr['arr_'+str(num)] for num in range(4)]

        arr = np.load('temp_svm_noboot.npz')
        [bin_keep_aus_all,all_aus_model, coefs] = [arr['arr_'+str(num)] for num in range(3)]

        
        bin_keep_aus_all = np.array(bin_keep_aus_all)
        bin_keep_aus_all = bin_keep_aus_all.T
        coefs = coefs.T
        # print coefs.shape, bin_keep_aus_all.shape
        bin_keep_sum = np.sum(bin_keep_aus_all, axis = 1)>0
        # rel_cols = coefs[:,bin_keep_sum]
        bin_neg = coefs<0
        # print bin_neg.shape
        bin_neg = np.sum(bin_neg,axis = 1)
        bin_keep = np.logical_and(bin_keep_sum, np.logical_not(bin_neg))

        print all_aus_model[bin_neg>0]
        print all_aus_model[bin_keep_sum]
        print all_aus_model[bin_keep]
        raw_input()
        features[features>0] = 1
        rel_features = features[:,bin_keep]
        bin_pain = np.in1d(labels, pain)
        
        p_feat = rel_features[bin_pain,:]
        np_feat = rel_features[np.logical_not(bin_pain),:]

        p_count = np.sum(p_feat,axis = 1)
        np_count = np.sum(np_feat, axis = 1)
        
        
        for idx_feat_curr, feat_curr in enumerate([p_feat, np_feat]):
            p_count = np.sum(feat_curr,axis = 1)
            print p_count.size
            to_rec_curr = []
            for count_curr in range(np.sum(bin_keep)):
                rel_sum = np.sum(p_count>=count_curr)
                prob = rel_sum/float(p_count.size)
                # print count_curr, rel_sum, rel_sum/float(p_count.size)
                to_rec_curr.append(prob)
            to_rec[idx_feat_curr].append(to_rec_curr)


        print to_rec[0]
        print to_rec[1]

    to_rec = [100*np.array(to_rec_curr).T for to_rec_curr in to_rec]
    print to_rec[0].shape

    for to_rec_curr in to_rec:
        str_curr = ['AUs']+[str(val[0]) for val in ows]
        str_curr = ' & '.join(str_curr)+'\\\\'
        print str_curr
        for idx_r, r in enumerate(to_rec_curr):
            str_curr = ['$\geq $'+str(idx_r)]+['%.2f'%val+"\\%"  for val in r]
            str_curr = ' & '.join(str_curr)+'\\\\'
            print str_curr
    print '___'





    # bin_curr = np.sum(bin_keep_aus_all, axis = 1)>0
    # aus_kept = list(all_aus_model[bin_curr])
    # aus_kept.sort()
    
    # rows = []
    # # row = []
    # row_title = [' ']+[str(val) for val in [2,5,10,15,20,30]]
    # rows.append(row_title)
    # for au in aus_kept:
    #     row = []
    #     row.append(printable_au(au))
    #     idx_au = np.where(all_aus_model==au)[0][0]
        
    #     for val in bin_keep_aus_all[idx_au,:]:
    #         print val
    #         if val:
    #             row.append('\checkmark')
    #         else:
    #             row.append(' ')
    #     # print ' & '.join(row)
    #     rows.append(row)
    # return
    # # return features, labels, pain, all_aus



def get_all_results(is_clinical, out_file):

    out_dir_meta = '../experiments/ml_on_aus_clinical'
    util.mkdir(out_dir_meta)
    eval_methods = ['majority']

    # ows = [[30,30]]
    ows = [[2,1],[5,2.5],[10,5],[15,7.5],[20,10],[30,30]]
    # ows = [[10,5]]
    feature_types = [['frequency']]
    # ,['max_duration'],['frequency','max_duration']]
    
    # ows = [[30,30]]
    # feature_types = [['frequency']]

    type_dataset = 'isch'

    
    fs_list = []
    fs = {}
    fs['feature_selection'] = 'cooc'
    fs['selection_params'] = dict(thresh=-0.1,type_dataset=type_dataset)
    fs_list.append(fs)

    fs = {}
    fs['feature_selection'] = 'cooc'
    fs['selection_params'] = dict(thresh=0.5,type_dataset=type_dataset)
    fs_list.append(fs)

    fs = {}
    fs['feature_selection'] = 'kunz'
    fs['selection_params'] = dict(type_dataset= type_dataset)
    fs_list.append(fs)
    
    fs = {}
    keep_rest = False
    fs['feature_selection'] = 'collapse'
    fs['selection_params'] = dict(select_mode=None,flicker=0,brow=0, type_dataset = type_dataset, keep_rest = keep_rest)
    fs_list.append(fs)

    fs = {}
    fs['feature_selection'] = 'collapse'
    fs['selection_params'] = dict(select_mode='cooc',flicker=0,brow=0, thresh = 0.5, type_dataset = type_dataset, keep_rest = keep_rest)
    fs_list.append(fs)

    fs = {}
    fs['feature_selection'] = 'collapse'
    fs['selection_params'] = dict(select_mode='kunz',flicker=0,brow=0, thresh = 0.05, type_dataset = type_dataset, keep_rest = keep_rest)
    fs_list.append(fs)

    fs_list_names = ['basic','cooc','kunz','collapse','collapse_cooc','collapse_kunz']
    assert len(fs_list_names)==len(fs_list)

    ml_model_list = []

    ml_model = {}
    ml_model['norm'] = 'l2_mean_std'
    ml_model['model_type'] = 'lda'
    ml_model['model_params'] = None
    ml_model['bootstrap'] = False
    ml_model_list.append(ml_model)
    
    ml_model = {}
    ml_model['norm'] = 'l2_mean_std'
    ml_model['model_type'] = 'svm'
    ml_model['model_params'] = {'C':1.,'kernel' : 'linear','class_weight':'balanced'}
    ml_model['bootstrap'] = False
    ml_model_list.append(ml_model)
        
    ml_model = {}
    ml_model['norm'] = 'l2_mean_std'
    ml_model['model_type'] = 'knn_8'
    ml_model['model_params'] = {'algorithm':'brute', 'weights':'uniform'}
    ml_model['bootstrap'] = True
    ml_model_list.append(ml_model)
    
    # ml_model = {}
    # ml_model['norm'] = 'l2_mean_std'
    # ml_model['model_type'] = 'knn_8'
    # ml_model['model_params'] = {'algorithm':'brute', 'weights':'uniform'}
    # ml_model['bootstrap'] = False
    # ml_model_list.append(ml_model)
    ml_model_list_names = ['lda','svm','knn_8']
    assert len(ml_model_list)==len(ml_model_list_names)


    # print feature_selection
    # print model_type
    # print 'norm', norm
    # print 'loo'

    results_fs = []
    for fs, fs_name in zip(fs_list, fs_list_names):
        feature_selection = fs['feature_selection']
        selection_params = fs['selection_params']

        result_list = []
        for ml_model,ml_model_name in zip(ml_model_list, ml_model_list_names):
            norm = ml_model['norm']
            model_type = ml_model['model_type']
            model_params = ml_model['model_params']
            bootstrap = ml_model['bootstrap']

            # results = 
            print '___'
            print fs_name, ml_model_name
            print '---'
            # 
            if is_clinical:
                results = test_clinical_loo(out_dir_meta, ows, feature_types, feature_selection,selection_params, eval_methods = eval_methods, norm = norm, model_type = model_type, model_params = model_params, bootstrap = bootstrap)
            else:
                results = script_loo(out_dir_meta, ows, feature_types, feature_selection,selection_params, eval_methods = eval_methods, norm = norm, model_type = model_type, model_params = model_params, bootstrap = bootstrap)  
            result_list.append(results[np.newaxis,:,:])
        result_list = np.concatenate(result_list, axis = 0)
        print result_list.shape
        # raw_input()
        results_fs.append(result_list[np.newaxis,:,:,:])
    results_fs = np.concatenate(results_fs, axis = 0)
    
    print results_fs.shape

    np.save(out_file,results_fs)
    # np.save('all_results_clinical_nohead.npy',results_fs)


def fix_cooc():

    out_dir_meta = '../experiments/ml_on_aus_clinical'
    util.mkdir(out_dir_meta)
    eval_methods = ['majority']

    # ows = [[30,30]]
    ows = [[2,1],[5,2.5],[10,5],[15,7.5],[20,10],[30,30]]
    # ows = [[10,5]]
    feature_types = [['frequency','duration']]
    
    # ows = [[30,30]]
    # feature_types = [['frequency']]

    type_dataset = 'isch'

    # feature_selection = None
    # selection_params = {}

    fs_list = []
    # fs = {}
    # fs['feature_selection'] = 'cooc'
    # fs['selection_params'] = dict(thresh=0.5,type_dataset=type_dataset)
    # fs_list.append(fs)

    # fs = {}
    # fs['feature_selection'] = 'kunz'
    # fs['selection_params'] = dict(type_dataset= type_dataset)
    # fs_list.append(fs)
    
    fs = {}
    keep_rest = False
    fs['feature_selection'] = 'collapse'
    fs['selection_params'] = dict(select_mode=None,flicker=0,brow=0, type_dataset = type_dataset, keep_rest = keep_rest)
    fs_list.append(fs)

    # fs = {}
    # fs['feature_selection'] = 'collapse'
    # fs['selection_params'] = dict(select_mode='cooc',flicker=0,brow=0, thresh = 0.5, type_dataset = type_dataset, keep_rest = keep_rest)
    # fs_list.append(fs)

    # fs = {}
    # fs['feature_selection'] = 'collapse'
    # fs['selection_params'] = dict(select_mode='kunz',flicker=0,brow=0, thresh = 0.05, type_dataset = type_dataset, keep_rest = keep_rest)
    # fs_list.append(fs)

    fs_list_names = ['collapse']
    assert len(fs_list_names)==len(fs_list)

    ml_model_list = []

    ml_model = {}
    ml_model['norm'] = 'l2_mean_std'
    ml_model['model_type'] = 'lda'
    ml_model['model_params'] = None
    ml_model['bootstrap'] = False
    ml_model_list.append(ml_model)
    
    # ml_model = {}
    # ml_model['norm'] = 'l2_mean_std'
    # ml_model['model_type'] = 'svm'
    # ml_model['model_params'] = {'C':1.,'kernel' : 'linear','class_weight':'balanced'}
    # ml_model['bootstrap'] = False
    # ml_model_list.append(ml_model)
        
    # ml_model = {}
    # ml_model['norm'] = 'l2_mean_std'
    # ml_model['model_type'] = 'knn_8'
    # ml_model['model_params'] = {'algorithm':'brute', 'weights':'uniform'}
    # ml_model['bootstrap'] = True
    # ml_model_list.append(ml_model)
    
    # ml_model = {}
    # ml_model['norm'] = 'l2_mean_std'
    # ml_model['model_type'] = 'knn_8'
    # ml_model['model_params'] = {'algorithm':'brute', 'weights':'uniform'}
    # ml_model['bootstrap'] = False
    # ml_model_list.append(ml_model)
    ml_model_list_names = ['lda']
    # ,'svm','knn_8']
    assert len(ml_model_list)==len(ml_model_list_names)


    # print feature_selection
    # print model_type
    # print 'norm', norm
    # print 'loo'

    results_fs = []
    for fs, fs_name in zip(fs_list, fs_list_names):
        feature_selection = fs['feature_selection']
        selection_params = fs['selection_params']

        result_list = []
        for ml_model,ml_model_name in zip(ml_model_list, ml_model_list_names):
            norm = ml_model['norm']
            model_type = ml_model['model_type']
            model_params = ml_model['model_params']
            bootstrap = ml_model['bootstrap']

            # results = 
            print '___'
            print fs_name, ml_model_name
            print '---'
            results = script_loo(out_dir_meta, ows, feature_types, feature_selection,selection_params, eval_methods = eval_methods, norm = norm, model_type = model_type, model_params = model_params, bootstrap = bootstrap)
            result_list.append(results[np.newaxis,:,:])
        result_list = np.concatenate(result_list, axis = 0)
        print result_list.shape

        results_fs.append(result_list[np.newaxis,:,:,:])
    results_fs = np.concatenate(results_fs, axis = 0)
    
    print results_fs.shape

    np.save('all_results.npy',results_fs)


def get_thresh_mean_std(arr, thresh):
    mean = []
    std = []
    for r in range(arr.shape[0]):
        arr_curr = arr[r,:]
        arr_curr = arr_curr[arr_curr>=thresh]
        mean.append(np.mean(arr_curr))
        std.append(np.std(arr_curr))
    return np.array(mean), np.array(std)


def process_all_results(out_file):
    dim_str = ['fs', 'model', 'ows', 'feat_type']
    # all_results = np.load('all_results_clinical_nohead.npy' )
    # all_results = np.load('all_results_nohead.npy' )

    all_results = np.load(out_file)
    # all_results = all_results[1:2,:,:1,:1]
    # print all_results.shape
    #     # 'all_results_clinical.npy' )
    # # all_results = np.load('all_results.npy' )
    # max_args = np.argmax(all_results)
    # max_args = np.unravel_index(max_args, all_results.shape)
    # print all_results[max_args]
    # # print all_results[0,0,0,0]
    # # print all_results[3,0,0,0]
    # # print all_results[2,1,:,:]
    # # print all_results [1,1,:,:]
    # # print all_results [1,0,:,:]
    # print max_args
    # raw_input()
    # dim_to_see = 3
    max_dims = []
    # max_dims_th = []
    for dim_to_see in range(len(dim_str)):
        # print all_results.shape
        r = np.moveaxis(all_results, dim_to_see, 0)
        # print r.shape
        r = np.reshape(r,(r.shape[0], r.size/r.shape[0]))
        r_mean = np.mean(r, axis = 1)
        r_std = np.std(r, axis = 1)
        r_th_mean, r_th_std = get_thresh_mean_std(r, 50.)
        percent_more = np.sum(r>=50., axis = 1)
        print r.shape
        # /float(r.shape[1])
        max_dim_curr = np.argmax(r_mean)
        max_dims.append(max_dim_curr)
        print dim_str[dim_to_see], r_mean, r_std, np.argmax(r_mean)
        print dim_str[dim_to_see], r_th_mean, r_th_std, np.argmax(r_th_mean)
        print percent_more


    # print max_dims
    # print all_results(max_dims)


    
def stress_exp():
    # from scipy import stats
    # data_type = 'frequency'
    # step_size = 30
    # inc = 30
    # type_dataset = 'stress'
    # # stress_t = 2

    # for stress_t in [1,2]:
    #     for data_type in ['frequency','max_duration']:
    #         features, labels, all_aus, [pain, matches] = get_feats(inc, step_size, data_type = data_type, type_dataset = type_dataset, flicker = 2, get_matches = True, split_pain = True)

            
    #         aus_kunz = np.array(['ad1','ad38','au101','au145','au47','au5','ead101','ead104','ad19', 'au25', 'auh13'])
    #         # aus_kunz = np.array(['ead101+ead104'])
    #         cols_keep = np.in1d(all_aus,aus_kunz)
    #         aus_keep = np.array(all_aus)[cols_keep]
    #         features_keep = features[:,cols_keep]

    #         matches = np.array(matches)
    #         horses_stress = matches[matches[:,2]==stress_t,0]
    #         rows_bl = []
    #         rows_stress = []
    #         for horse in horses_stress:
    #             row_bl = np.where(np.logical_and(matches[:,0]==horse, matches[:,2]==0))[0]
    #             row_stress = np.where(np.logical_and(matches[:,0]==horse, matches[:,2]==stress_t))[0]
    #             keep = min(row_bl.size, row_stress.size)
                
    #             row_bl = row_bl[:keep]
    #             row_stress = row_stress[:keep]
    #             rows_bl.extend(row_bl)
    #             rows_stress.extend(row_stress)
                

    #         print stress_t
    #         print data_type
    #         for col_idx in range(features_keep.shape[1]):
             
    #             pvalue = stats.ttest_rel(features_keep[rows_bl,col_idx], features_keep[rows_stress,col_idx]).pvalue
    #             str_p = aus_keep[col_idx]+'\t'
    #             if pvalue<0.001:
    #                 str_p+='<0.001'
    #             else:
    #                 str_p+='%.3f'%pvalue
                
    #             print str_p

    # return
    # print 'hello'
    # data_type = 'frequency'
    # step_size = 30
    # inc = 30
    # type_dataset = 'stress_tr'
    # # aus_kunz = np.array(['ead101_ead104'])
    # # aus_kunz = np.array(['ad1','ad38','au101','au145','au47','au5','ead101','ead104','ad19', 'au25', 'auh13'])
    # aus_kunz = np.array(['ead101+ead104'])
    # # aus_kunz = np.array(['ad1','ad38','au101','au145','au47','au5','ead101','ead104'])
    # # aus_kunz = np.array(['ad19','ad81','au16', 'au25'])

    # bins = []
    # features_list = [[],[],[],[],[],[]]
    # for idx_d, type_dataset in enumerate([ 'stress_tr','stress_si']):
    #     features, labels, all_aus, pain = get_feats(inc, step_size, data_type = data_type, type_dataset = type_dataset, flicker = 1)

    #     cols_keep = np.in1d(all_aus,aus_kunz)
    #     aus_keep = np.array(all_aus)[cols_keep]
    #     # features_keep = features[:,cols_keep]

    #     features_dur, labels_dur, all_aus_dur, _ = get_feats(inc, step_size, data_type = 'max_duration', type_dataset = type_dataset, flicker = 1)
    #     # features_dur_keep = features[:,cols_keep]
    #     aus_keep_dur = np.array(all_aus_dur)[cols_keep]
    #     assert np.all(aus_keep==aus_keep_dur)
    #     assert np.all(labels_dur==labels)

    #     bin_pain = np.in1d(labels, pain)
    #     bin_bl = np.logical_not(bin_pain)
        
    #     for idx_f, feat in enumerate([features, features_dur]):
    #         feat = feat[:,cols_keep]
    #         feat_p = feat[bin_pain,:]
    #         feat_bl = feat[bin_bl,:]
    #         idx_list = idx_d*2+idx_f
    #         idx_bl = len(features_list)-2+idx_f
    #         # print type_dataset, idx_f, idx_list, idx_bl
    #         features_list[idx_list]= feat_p
    #         features_list[idx_bl].append(feat_bl)
    
    # features_list[-2] = np.concatenate(features_list[-2], axis = 0)
    # features_list[-1] = np.concatenate(features_list[-1], axis = 0)

    # str_p = '\t'.join([val.upper() for val in aus_keep])
    # print str_p
    # str_type = ['Transportation Frequency','Transportation Duration','Social Isolation Frequency',
    #             'Social Isolation Duration', 'Baseline Frequency','Baseline Duration']
    # for idx_f in [0,2,4,1,3,5]:
    #     # print f.shape
    #     f = features_list[idx_f]
    #     vals = np.mean(f,axis = 0)
    #     str_p = '\t'.join([str_type[idx_f]]+['%.3f'%val for val in vals])
    #     print str_p

    # return

    # results_fs = []
    # for fs, fs_name in zip(fs_list, fs_list_names):
    out_dir_meta = '../experiments/stress_exp'
    # numpy.random.seed(seed=99)
    util.mkdir(out_dir_meta)
    type_dataset = 'stress'        
    feature_selection = 'kunz'
    ows = [[30,30]]
    selection_params = dict(type_dataset=type_dataset)
    # ,flicker =1)
    feature_types = [['frequency'],['max_duration'],['frequency','max_duration']]
    eval_methods = ['majority']

    # feature_selection = 'cooc'
    # ows = [[30,30]]
    # selection_params = dict(type_dataset=type_dataset, thresh = 0.5)
    # feature_types = [['frequency'],['max_duration'],['frequency','max_duration']]
    # eval_methods = ['majority']

    ml_model = {}
    ml_model['norm'] = 'l2_mean_std'
    ml_model['model_type'] = 'svm'
    ml_model['model_params'] = {'C':[0.01,0.1,1,10,100],'kernel' : ['linear'],'class_weight':['balanced']}
    # ml_model['model_params'] = {'C':1.,'class_weight':'balanced'}
    ml_model['bootstrap'] = False

    # print 'C 0.1'
    # ml_model = {}
    # ml_model['norm'] = 'l2_mean_std'
    # ml_model['model_type'] = 'lda'
    # ml_model['model_params'] = None
    # ml_model['bootstrap'] = False

    norm = ml_model['norm']
    model_type = ml_model['model_type']
    model_params = ml_model['model_params']
    bootstrap = ml_model['bootstrap']

    results = script_loo(out_dir_meta, ows, feature_types, feature_selection,selection_params, eval_methods = eval_methods, norm = norm, model_type = model_type, model_params = model_params, bootstrap = bootstrap)  
            # result_list.append(results[np.newaxis,:,:])

    print results


def main():
    # get_prob_frame_video()
    # percentage_pain_au_segments()
    # frequency_analysis()
    changing_cooc()
    # import warnings
    # warnings.filterwarnings("ignore")
    # , category=DeprecationWarning)
    # stress_exp()



    return
    is_clinical =False
    out_file = ['all_results_testingcooc']+['_clinical_loo' if is_clinical else '']+['.npy']
    out_file = ''.join(out_file)
    print out_file
    # raw_input()

    get_all_results(is_clinical, out_file)
    # process_all_results(out_file)

    # get_prob_frame_video()
    # arr = np.load('temp_svm_noboot.npz')
    # [bin_keep_aus_all,all_aus_model, coefs] = [arr['arr_'+str(num)] for num in range(3)]
    # print coefs.shape
    # print bin_keep_aus_all.shape
    # # return
    # pt.cooc_table(bin_keep_aus_all, all_aus_model, coefs)
    # return 
    # print 'hello'
    ## moved from here to get_all_results

    # print 'test'
    # bin_keep_aus_all, all_aus_model, coefs = test_clinical(out_dir_meta, ows, feature_types, feature_selection,selection_params, eval_methods = eval_methods, norm = norm, model_type = model_type, model_params = model_params, bootstrap = bootstrap)
    # coefs = np.array(coefs)

    # print coefs.shape
    # np.savez('temp_lda_noboot.npz',bin_keep_aus_all,all_aus_model, coefs)
    # arr = np.load('temp.npz')
    # bin_keep_aus_all = arr['arr_0']
    # all_aus_model = arr['arr_1']

    # pt.cooc_table(bin_keep_aus_all, all_aus_model, coefs)



if __name__=='__main__':
    main()
