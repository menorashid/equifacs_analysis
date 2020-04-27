from doing_ml import *

def train_test_single_fold(train_features, train_labels, train_vl, label_to_test, model_specs):
    pass
    # return pred_labels

def train_test_single_fold(features, class_pain, labels, idx_test, bin_aus_keep, bootstrap, norm, model_type, model_params, eval_method):
    idx_train = np.logical_not(idx_test)
    features_train  = features[idx_train,:]
    features_train = features_train[:,bin_aus_keep]
    labels_train = class_pain[idx_train]

    if bootstrap:
        k =  max(1,features_train.shape[0]//8)
        bin_keep = cl.bootstrap(features_train, labels_train, norm, k)
        features_train = features_train[bin_keep,:]
        labels_train = labels_train[bin_keep]

    if model_type.startswith('knn'):
        k = max(1,features_train.shape[0]//int(model_type.split('_')[-1]))
        model_params['n_neighbors'] = k

    lda, scaler, data_lda = cl.fit_model(features_train, labels_train, norm = norm, model_type = model_type, model_params = model_params)

    bins = [idx_train, idx_test]
    
    gts = []
    preds = []
    for idx_loo,bin_rel in enumerate(bins):
        features_curr = features[bin_rel,:]
        features_curr = features_curr[:,bin_aus_keep]
        gts_curr = class_pain[bin_rel]
        vid_labels_curr = labels[bin_rel]
        preds_curr = lda.predict(scaler.transform(features_curr))
        
        gts_curr, preds_curr = get_labels_for_eval(gts_curr, preds_curr, vid_labels_curr, eval_method)
        gts.append(gts_curr), preds.append(preds_curr)
    return gts, preds
    

def script_loo(out_dir_meta, ows, feature_types,  feature_selection = None, selection_params={}, eval_methods = ['raw','majority','atleast_1'], norm = 'mean_std', model_type = None, model_params = None, bootstrap = False):

    # pain = np.array([1,2,4,5,11,12,13,14,15,16,17,18,19,20])
    selection_params = copy.deepcopy(selection_params)
    # selection_params['pain']=pain

    iterator = itertools.product(ows,feature_types)

    table_curr = np.zeros((len(ows), len(feature_types)))
    row_labels = ows
    col_labels = feature_types

    for ows_curr, feature_type in iterator:
        inc,step_size = ows_curr

        # dir_str = '_'.join([str(val) for val in ['inc',inc,'step',step_size]])
        # feature_type_str = '_'.join(feature_type)
        # out_dir = os.path.join(out_dir_meta,dir_str,feature_type_str)
        # print out_dir
        # util.mkdir(out_dir)
        
        selection_params['inc']=inc
        selection_params['step_size']=step_size
        selection_params['feature_type']=feature_type

        features, labels, all_aus, class_pain, idx_test_all, bin_keep_aus = select_features(selection_type = feature_selection, selection_params = selection_params)
        bin_something = np.sum(features, axis = 1)>0

        # class_pain = np.in1d(labels, pain).astype(int)
        # print class_pain
        # print class_pain_new
        # raw_input()
        loo_results_train = []
        loo_results_test = []

        for eval_method in eval_methods:
            preds = [[],[]]
            gts = [[],[]]

            for idx_test, bin_aus_keep in zip(idx_test_all, bin_keep_aus):
                
                gts_curr, preds_curr = train_test_single_fold(features, class_pain, labels, idx_test, bin_aus_keep, bootstrap, norm, model_type, model_params, eval_method)
                assert len(gts_curr)==len(preds_curr)==2
                for idx in range(len(gts_curr)):
                    gts[idx].append(gts_curr[idx])
                    preds[idx].append(preds_curr[idx])
                    # gts[idx_loo].append(gts_curr) 
                    # preds[idx_loo].append(preds_curr)

            gts = [np.concatenate(gt_curr, axis = 0) for gt_curr in gts]
            preds = [np.concatenate(gt_curr, axis = 0) for gt_curr in preds]
            to_print = ['train','test']
            
            for idx,(gt,pred) in enumerate(zip(gts, preds)):
                if idx==0:
                    continue
                precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(gt, pred, average = 'binary')
                accuracy = np.sum(gt ==pred)/float(gt.size)
                # print to_print[idx],eval_method, precision, recall, f1
                # print ' '.join(['%.2f'%val for val in [precision, recall, f1]])
                row_idx = row_labels.index(ows_curr)
                # print row_idx, row_labels, ows_curr
                col_idx = col_labels.index(feature_type)
                # print col_idx, col_labels, feature_type
                table_curr[row_idx, col_idx] = f1
                # raw_input()
    # print table_curr
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


def test_clinical_loo(out_dir_meta, ows, feature_types,  feature_selection = None, selection_params={}, eval_methods = ['raw','majority','atleast_1'], norm = 'mean_std', model_type = 'lda', model_params = None, bootstrap = False):

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
        test_package['labels'] = test_package['labels']*100

        test_labels_raw = np.array(test_labels)
        test_labels[test_labels>=0.5] = 1
        test_labels[test_labels<0.5] = 0

        class_pain = np.concatenate([train_labels,test_labels])
        features = np.concatenate([train_features, test_features], axis = 0)
        vl_labels_all = np.concatenate([train_package['labels'], test_package['labels']])
        bin_aus_keep = np.ones((features.shape[1],))>0
        
        # print class_pain.shape
        # print features.shape
        # print vl_labels_all.shape
        # print bin_aus_keep.shape

        # for eval_method in eval_methods:
        eval_method = eval_methods[0]
        gts = [[],[]]
        preds = [[],[]]
        for label_test in np.unique(vl_labels_all):
            idx_test = vl_labels_all==label_test
            gts_curr, preds_curr = train_test_single_fold(features, class_pain, vl_labels_all, idx_test, bin_aus_keep, bootstrap, norm, model_type, model_params, eval_method)

            assert len(gts_curr)==len(preds_curr)==2
            for idx in range(len(gts_curr)):
                gts[idx].append(gts_curr[idx])
                preds[idx].append(preds_curr[idx])
                    
        gts = [np.concatenate(gt_curr, axis = 0) for gt_curr in gts]
        preds = [np.concatenate(gt_curr, axis = 0) for gt_curr in preds]
        to_print = ['train','test']

        for idx,(gt,pred) in enumerate(zip(gts, preds)):
            if idx==0:
                continue
            precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(gt, pred, average = 'binary')
            row_idx = row_labels.index(ows_curr)
            col_idx = col_labels.index(feature_type)
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

def get_p_values(features, labels, partners, class_pain):
    from scipy import stats
    idx_p = []
    idx_np = []
    for h1,h2 in partners:
        idx_p+=list(np.where(labels==h1)[0])
        idx_np+=list(np.where(labels==h2)[0])

    idx_p= np.array(idx_p)
    idx_np = np.array(idx_np)
    assert idx_p.size == idx_np.size
    
    if class_pain is not None:
        assert np.unique(class_pain[idx_p])==np.array([1])
        assert np.unique(class_pain[idx_np])==np.array([0])

    pvalues= []
    for feature_idx in range(features.shape[1]):
        pvalue = stats.ttest_rel(features[idx_p,feature_idx],features[idx_np,feature_idx]).pvalue
        pvalues.append(pvalue)

    return np.array(pvalues)


def changing_cooc():
    feature_selection = 'cooc'
    selection_params = {}
    selection_params['inc']=2
    selection_params['step_size']=1
    selection_params['feature_type']=['frequency']
    selection_params['thresh'] = 0.5
    selection_params['type_dataset']='isch'
    ows = [[2,1],[5,2.5],[10,5],[15,7.5],[20,10],[30,30]]
    partners = [(1,6),(2,9),(12,3),(4,8),(5,10),(11,7)]

    
    all_aus_all = []
    bin_keep_aus_all = []
    p_values_all = []

    for ows_curr in ows:
        selection_params['inc']=ows_curr[0]
        selection_params['step_size']=ows_curr[1]
        features, labels, all_aus, class_pain, idx_test_all, bin_keep_aus = select_features(selection_type = feature_selection, selection_params = selection_params)
        # features [features>0]
        # print features[:3]
        # print features.shape, np.sum(bin_keep_aus[0])
        # print np.sum(features, axis = 0)

        p_values = get_p_values(features, labels, partners, class_pain)
        p_values_all.append(p_values)
        bin_keep_aus_all.append(bin_keep_aus[0])
        all_aus_all.append(np.array(all_aus))

    for all_au in all_aus_all:
        assert np.all(all_aus_all[0]==all_au)

    bin_keep_aus_all = np.array(bin_keep_aus_all)
    print bin_keep_aus_all.shape
    sum_aus = np.sum(bin_keep_aus_all, axis = 0)
    arg_sort = np.argsort(sum_aus)[::-1]
    arg_sort = arg_sort[:12] 
    print sum_aus[arg_sort]

    p_values_print = np.zeros((bin_keep_aus_all.shape[0],arg_sort.size))
    # for idx_col in arg_sort:
    for idx_row in range(p_values_print.shape[0]):
        for idx_col in range(p_values_print.shape[1]):
            idx_au = arg_sort[idx_col]
            p_value_rel = p_values_all[idx_row][idx_au]
            bin_keep_rel = bin_keep_aus_all[idx_row][idx_au]
            if bin_keep_rel:
                p_values_print[idx_row, idx_col] = p_value_rel

    rows_to_print = []
    row_curr = [' ']+[str_curr.upper() for str_curr in all_aus_all[0][arg_sort]]
    rows_to_print.append(' & '.join(row_curr))
    for idx_row, p_value_row in enumerate(p_values_print):
        row_curr = [' ']
        row_cm = ['\multirow{2}{1}{$'+str(ows[idx_row][0])+'$}']
        for p_value in p_value_row:
            # print p_value
            if p_value ==0:
                row_cm.append(' ')
            else:
                row_cm.append('\checkmark')

            if p_value ==0:
                row_curr.append(' ')
            elif p_value <0.001:
                row_curr.append('(p<0.001)')
            elif p_value<0.01:
                row_curr.append('(p<0.01)')
            elif p_value<0.05:
                row_curr.append('(p<0.05)')
            else:
                # row_curr.append('\\mr{'+'%.2f'%p_value+'}')
                row_curr.append('(p=%.2f'%p_value+')')

        str_row = ' & '.join(row_cm)+'\\\\'
        rows_to_print.append(str_row)
        str_row = ' & '.join(row_curr)+'\\\\\\hline'
        rows_to_print.append(str_row)

    for row_to_print in rows_to_print:
        print row_to_print
        # +'\\\\\\hline'



    # print p_values_print
        

            # print np.unique(class_pain[labels==h1]),np.unique(class_pain[labels==h2])

            # pvalue = stats.ttest_rel(features_keep[rows_bl,col_idx], features_keep[rows_stress,col_idx]).pvalue

      
    return
    out_dir_meta = '../experiments/cooc_simple'
    eval_methods = ['majority']
    # util.mkdir(out_dir)
        
    # features, labels, all_aus, class_pain, idx_test_all, bin_keep_aus = select_features(selection_type = feature_selection, selection_params = selection_params)

    is_clinical = False
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
    fs['selection_params'] = dict(thresh=0.5,type_dataset=type_dataset)
    fs_list.append(fs)

    fs_list_names = ['cooc']
    assert len(fs_list_names)==len(fs_list)

    ml_model_list = []

    # ml_model = {}
    # ml_model['norm'] = 'l2_mean_std'
    # ml_model['model_type'] = 'lda'
    # ml_model['model_params'] = None
    # ml_model['bootstrap'] = False
    # ml_model_list.append(ml_model)
    
    ml_model = {}
    ml_model['norm'] = 'l2_mean_std'
    ml_model['model_type'] = 'svm'
    ml_model['model_params'] = {'C':1.,'kernel' : 'linear','class_weight':'balanced'}
    ml_model['bootstrap'] = False
    ml_model_list.append(ml_model)
        
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
    ml_model_list_names = ['svm']
    # ['lda','svm','knn_8']
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

    # np.save(out_file,results_fs)
    # np.save('all_results_clinical_nohead.npy',results_fs)


def frequency_analysis():
    feature_selection = 'kunz'
    selection_params = {}
    selection_params['inc']=2
    selection_params['step_size']=1
    selection_params['feature_type']=['frequency']
    # selection_params['thresh'] = 0.5
    selection_params['type_dataset']='clinical'
    ows = [[2,1],[5,2.5],[10,5],[15,7.5],[20,10],[30,30]]
    partners = [(1,6),(2,9),(12,3),(4,8),(5,10),(11,7)]

    
    for eval_method in ['raw','majority']:
        
        print_rows = []
        
        
        for idx_ows, ows_curr in enumerate(ows):
            selection_params['inc']=ows_curr[0]
            selection_params['step_size']=ows_curr[1]
            features, labels, all_aus, class_pain, idx_test_all, bin_keep_aus = select_features(selection_type = feature_selection, selection_params = selection_params)    
            
            bin_keep = bin_keep_aus[0]
            aus_curr = np.array(all_aus)[bin_keep]
            features = features[:,bin_keep]
            features[features>0] = 1
            features_both = np.logical_or(features[:,0],features[:,1]).astype(int)
            
            preds = [features[:,0],features[:,1], np.logical_or(features[:,0],features[:,1]).astype(int),np.logical_and(features[:,0],features[:,1]).astype(int)]

            # str_preds_all = list(aus_curr)+['either','both']
            
            if idx_ows==0:
                title_row = ['OWS']+[au_curr.upper() for au_curr in aus_curr]+['Either','Both']
                title_row = title_row+title_row[1:]
                print_rows.append(title_row)

            print_row = [[],[]]

            for preds_in in preds:
                gts_curr, preds_curr = get_labels_for_eval(class_pain, preds_in, labels, eval_method)
                precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(gts_curr, preds_curr, average = 'binary')

                
                ppv = np.sum(np.logical_and(gts_curr==preds_curr,gts_curr==1))/float(np.sum(preds_curr==1))
                npv = np.sum(np.logical_and(gts_curr==preds_curr,gts_curr==0))/float(np.sum(preds_curr==0))
                print_row[0].append('%.2f'%(ppv*100)+'\%')
                print_row[1].append('%.2f'%(npv*100)+'\%')

                # sens = np.sum(np.logical_and(gts_curr==preds_curr,gts_curr==1))/float(np.sum(gts_curr==1))
                # specs = np.sum(np.logical_and(gts_curr==preds_curr,gts_curr==0))/float(np.sum(gts_curr==0))
                # print sens,specs
                # print_row[0].append('%.2f'%(sens*100)+'\%')
                # print_row[1].append('%.2f'%(specs*100)+'\%')
                
            
            print_row = [str(ows_curr[0])]+print_row[0]+print_row[1]
            print_rows.append(print_row)

        print eval_method
        pt.print_table_strs(print_rows)
        print ' '


            
            

def percentage_pain_au_segments():
    feature_selection = 'kunz'
    selection_params = {}
    selection_params['inc']=2
    selection_params['step_size']=1
    selection_params['feature_type']=['frequency']
    # selection_params['thresh'] = 0.5
    selection_params['type_dataset']='isch'
    ows = [[0.04,0.04],[2,1],[5,2.5],[10,5],[15,7.5],[20,10],[30,30]]
    partners = [(1,6),(2,9),(12,3),(4,8),(5,10),(11,7)]

    
    all_aus_all = []
    bin_keep_aus_all = []
    p_values_all = []
    probs = [[],[]]
    for ows_curr in ows:
        selection_params['inc']=ows_curr[0]
        selection_params['step_size']=ows_curr[1]
        features, labels, all_aus, class_pain, idx_test_all, bin_keep_aus = select_features(selection_type = feature_selection, selection_params = selection_params)    

        bin_keep = bin_keep_aus[0]
        num_aus = np.sum(bin_keep)
        aus_curr = np.array(all_aus)[bin_keep]
        # print aus_curr
        # raw_input()
        features = features[:,bin_keep]
        features[features>0] = 1
        fs = np.sum(features, axis = 1)
        # print class_pain.shape, np.sum(class_pain)
        # raw_input
        for p_np in range(2):
            features_rel = features[class_pain==p_np,:]
            features_au17 = features_rel[:,3]
            feature_sum = np.sum(features_rel, axis = 1)
            # print p_np,features_rel.shape, feature_sum.size
            prob_arr = []
            for num_au in range(num_aus+1):
                bin_num_au = feature_sum>=num_au
                prob_arr.append(np.sum(bin_num_au)/float(feature_sum.size))
                prob_arr[-1] = np.sum(features_au17[bin_num_au])/float(feature_sum.size)



            probs[p_np].append(prob_arr)

    # probs = np.array(probs)
    # print probs
    ows = [val[0] for val in ows]
    print 'no pain',selection_params['type_dataset']
    pt.print_prob_table(100*np.array(probs[0]).T,ows)
    print ''
    print ''
    print 'pain',selection_params['type_dataset']
    pt.print_prob_table(100*np.array(probs[1]).T,ows)
        


