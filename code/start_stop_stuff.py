import sklearn
from sklearn import preprocessing,decomposition,discriminant_analysis,pipeline,neighbors, metrics
from helpers import util, visualize
from read_in_data import *
import itertools
import loo_classifying as lc
import scipy

def get_start_stop_feat(data_dict, all_aus, key_arr, inc, data_type, feat_keep = None, vid_length = 30., step_size = None):

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
        if step_size is None:
            step_size = inc
        
        inc_pts = list(np.arange(0,vid_length,step_size))
        inc_pts = inc_pts+[vid_length]
        
        # print inc_pts
        # raw_input()
        


        mat_curr = np.zeros((len(inc_pts)-1, len(all_aus)))
        num_keep = 0
        for idx_inc, inc_start in enumerate(inc_pts[:-1]):
            inc_end = min(vid_length, inc_start+inc)
            if (inc_end - inc_start)<inc:
                break
            num_keep +=1
            # print inc_start, inc_end

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

            
        mat_curr = mat_curr[:num_keep,:]
        features.append(mat_curr)
        labels = labels+[k]*mat_curr.shape[0]        

    features = np.concatenate(features, axis = 0)
    labels = np.array(labels)

    if feat_keep is not None:
        features, all_aus,_ = lc.prune_features(features, all_aus, feat_keep)
    
    return features, labels, all_aus

def get_time_series_feat(data_dict, feat_keep, key_arr, vid_length = 30., decimal_place = 2):
    features = []
    # all_aus = np.array(all_aus)
    labels = []
    # np.zeros((len(key_arr),len(all_aus)))

    mat_dict = {}

    for idx_k, k in enumerate(key_arr):
        
        num_dec = 10**decimal_place
        
        au_anno = np.array(data_dict[k][0])
        start = np.array(data_dict[k][2])
        end = np.array(data_dict [k] [3])
        end[end>vid_length]=vid_length

        assert start.size == end.size
        
        mat_time = np.zeros((len(feat_keep), int(vid_length*num_dec)))
        times = np.arange(0,vid_length*num_dec)
        start = start*num_dec
        end = end*num_dec

        times = []
        for idx_feat_curr, feat_curr in enumerate(feat_keep):
            bin_rel = au_anno==feat_curr
            start_vals = start[bin_rel]
            end_vals = end[bin_rel]
            for start_curr,end_curr in zip(start_vals, end_vals):
                start_curr = int(start_curr)
                end_curr = int(end_curr)
                mat_time[idx_feat_curr, start_curr:end_curr] = 1
            times_curr = np.concatenate([start_vals[:,np.newaxis], end_vals[:,np.newaxis]],axis = 1)
            times.append(times_curr)

        mat_dict[k]=(mat_time, times)

    return mat_dict, feat_keep
    # , all_aus


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


def prune_max_diff(features, bin_pain ):
    # features, labels, all_aus = get_start_stop_feat(data_dict, all_aus_org, key_arr, inc, data_type, feat_keep = feat_keep)

    # bin_pain = np.in1d(labels, pain)

    class_pain = np.zeros(bin_pain.shape)
    class_pain[bin_pain]=1
    class_pain = class_pain.astype(int)

    # feat_keep_str = ['+'.join([str_curr.upper() for str_curr in feat_keep_str])]
    # if max_diff:
    features_bin = np.zeros(features.shape)
    features_bin[features>0]= 1    
    num_aus = np.sum(features_bin,axis = 1)
    num_aus_p = num_aus[class_pain>0]
    num_aus_np = num_aus[class_pain<=0]
    bin_range = range(features.shape[1]+2)

    hist_p,bin_edges = np.histogram(num_aus_p,bin_range)
    hist_np,bin_edges = np.histogram(num_aus_np,bin_range)

    diffs = hist_np-hist_p
    idx_max = np.argmax(diffs)
    val_max = bin_edges[idx_max+1]

    idx_keep = num_aus>idx_max
    features = features[idx_keep,:]
    class_pain = class_pain[idx_keep]
    bin_pain = bin_pain[idx_keep]
    # feat_keep_str.append('k='+str(val_max))
    # else:
    # feat_keep_str.append('k not set')

    return features, bin_pain, val_max

def script_plot_cooc():
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict,remove_lr=False)
    all_aus = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])
    # bin_pain = np.in1d(np.array(key_arr),pain)
    # print bin_pain

    data_keeps = ['pain','no_pain','all']
    feat_keeps = [['au','ead']]
    out_dir_meta = '../experiments/visualizing_cooc_12'
    util.mkdir(out_dir_meta)
    inc_range = [1]
    max_diff = False
    # 5,10,15,30]

    for inc in inc_range:
        features, labels,_ = get_start_stop_feat(data_dict, all_aus, key_arr, inc, 'binary')
        print inc, features.shape
        bin_pain = np.in1d(labels,pain)
        if max_diff:
            features, bin_pain, val_max = prune_max_diff(features, bin_pain )
            
        for data_keep, feat_keep in itertools.product(data_keeps,feat_keeps):
            
            out_dir = os.path.join(out_dir_meta,'_'.join(feat_keep+[str(inc)]))
            if max_diff:
                out_dir = out_dir+'_maxdiff'
            util.mkdir(out_dir)
            
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
    
    
    test_scheme = None
    feat_keeps = [['au']] 
    # train_scheme_inc_pairs =[(2,2),(3,5),(4,10),(6,15),(7,30)] 

    # feat_keeps = [['au','ad']] 
    # train_scheme_inc_pairs =[(3,2),(4,5),(7,10),(9,15),(11,30)] 
    train_scheme_inc_pairs =[(None,2),(None,5),(None,10),(None,15),(None,30)] 

    for train_scheme,inc in train_scheme_inc_pairs:
    

        out_dir = '../experiments/loo_log_reg_'+str(inc)+'_test_'+str(test_scheme)+'_train_'+str(train_scheme)
        util.mkdir(out_dir)
        
           
        data_types = ['binary','frequency','duration']
        # , 'max_duration', 'min_duration','mean_duration']
        norms = ['l2_mean_std']
        log_reg_params = {'penalty':'l2','dual': False, 'fit_intercept': False,'class_weight':'balanced'}
        
        log_reg_str = []
        for k in log_reg_params:
            log_reg_str.append(k)
            log_reg_str.append(log_reg_params[k])
        log_reg_str = '_'.join([str(val) for val in log_reg_str])
        out_file = os.path.join(out_dir, 'results_'+log_reg_str+'.txt')
        

        to_print = []
        for (data_type, norm, feat_keep) in itertools.product(data_types, norms, feat_keeps):
            if feat_keep is None:
                feat_keep_str = ['all']
            else:
                feat_keep_str = feat_keep
            
            str_curr = ' '.join([str(inc),str(train_scheme), str(test_scheme),data_type, norm]+feat_keep_str)

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
                
                test_data = features[bin_test,:]
                test_labels = class_pain[bin_test]
                

                bin_train = ~bin_test
                train_data = features[bin_train,:]
                train_labels = class_pain[bin_train]


                if train_scheme is not None:
                    # print 'in train_scheme'
                    # print train_data.shape
                    # print train_labels.shape

                    num_aus = np.sum(train_data>0,axis = 1)
                    # print train_data[:3,:]
                    # print num_aus[:3]
                    # print np.sum(train_labels==1)
                    # print np.sum(train_labels==0)

                    bin_keep = num_aus>train_scheme
                    bin_keep = np.logical_or(bin_keep, train_labels==0)

                    train_labels = train_labels[bin_keep]
                    train_data = train_data[bin_keep,:]
                    # print np.sum(train_labels==1)
                    # print np.sum(train_labels==0)

                    # print train_data.shape
                    # print train_labels.shape
                    # raw_input()
                    # print count
                    # train_labels[~count]=0
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


def select_knn(data_dict,all_aus_org,key_arr,pain,inc,data_type,feat_keep,norm,log_reg_params,test_scheme,k_range,to_print = [], b_type = False):
    

    features, labels, all_aus = get_start_stop_feat(data_dict, all_aus_org, key_arr, inc, data_type, feat_keep = feat_keep)
    exclude_list = ['ad133', 'ead103']
    bin_keep = [1 if au not in exclude_list else 0 for au in all_aus]
    bin_keep = np.array(bin_keep)>0
    
    all_aus = list(np.array(all_aus)[bin_keep])
    features = features[:,bin_keep]
    # print type(all_aus)
    # print all_aus
    # print features.shape
    # print [0]+range(2,features.shape[1])

    # features = features[:,[0]+range(2,features.shape[1])]
    # all_aus = all_aus[:1]+all_aus[2:]
    # print all_aus
    # print features.shape
    # raw_input()
    # print inc, features.shape

    bin_pain = np.in1d(labels, pain)

    class_pain = np.zeros(bin_pain.shape)
    class_pain[bin_pain]=1
    class_pain = class_pain.astype(int)

    # print features.shape, class_pain.shape
    # print features.shape[0]
    log_reg_params['n_neighbors'] = features.shape[0]-1
    nn = sklearn.neighbors.NearestNeighbors(**log_reg_params)
    nn.fit(features)
    dist, indices_total = nn.kneighbors()
    
    precisions = []
    recalls = []
    f1s = []
    # print indices_total.shape
    # k_range = [1,3,5,7,9,11]
    class_pain_ts = []
    pred_ts = []

    for k in k_range:
        indices = indices_total[:,:k]

        bin_count = np.zeros(indices.shape)
        for idx, indices_curr in enumerate(indices):
            bin_count[idx,:] = class_pain[indices_curr]

        majority_labels = np.sum(bin_count,axis = 1)
        half_less = k//2
        bin_maj = majority_labels>half_less
        majority_labels[bin_maj] = 1
        majority_labels[~bin_maj] = 0
        

        if test_scheme=='All':
            class_pain_t = class_pain
            pred_t = majority_labels
        else:
            _, class_pain_t, pred_t = transform_labels(labels, class_pain, majority_labels, test_scheme)

        precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(class_pain_t, pred_t, beta=1.0, average = 'binary')
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        class_pain_ts.append(class_pain_t)
        pred_ts.append(pred_t)

        str_curr = 'K %d Precision %.2f Recall %.2f F1 %.2f' % (k, precision, recall, f1)

        to_print.append(str_curr)
        print str_curr 
    
    if b_type:
        return nn, f1s, features, labels, class_pain, all_aus, class_pain_ts, pred_ts
    else:
        return precisions, recalls, f1s, to_print

def knn(k_range , inc = 5 ):
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict)
    all_aus_org = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])
    
    out_dir = '../experiments/knn_'+str(inc)
    # +'_k_'+str(k)
    util.mkdir(out_dir)
    
    feat_keeps = [['au','ad']]    
    # ['au']]
    # ,['au','ad']]    
    data_types = ['frequency']
    # ,'frequency','duration']
    norms = ['l2_mean_std']
    log_reg_params = {'metric': 'cosine', 'algorithm': 'brute'}
    
    # test_scheme = 'majority'
    test_scheme = 'atleast_one'

    log_reg_str = []
    for key_curr in log_reg_params:
        log_reg_str.append(key_curr)
        log_reg_str.append(log_reg_params[key_curr])
    
    log_reg_str.append(test_scheme)

    log_reg_str_str = '_'.join([str(val) for val in log_reg_str])
    out_file = os.path.join(out_dir, 'results_'+log_reg_str_str+'.txt')
    

    to_print = []
    for (data_type, norm, feat_keep) in itertools.product(data_types, norms, feat_keeps):

        if feat_keep is None:
            feat_keep_str = ['all']
        else:
            feat_keep_str = feat_keep

        

        #     str_curr = ' '.join([data_type, norm, 'all'])
        # else:    
        str_curr = ' '.join([data_type, str(norm)]+feat_keep_str)
        to_print.append(str_curr)
        
        ####cut here####
        
        precisions, recalls, f1s, to_print,_ = select_knn(data_dict,all_aus_org,key_arr,pain,inc,data_type,feat_keep,norm,log_reg_params,test_scheme,k_range,to_print = to_print)
        
        # out_file_plot = out_file[:out_file.rindex('.')]+'.jpg'
        out_file_str = log_reg_str+[data_type]+feat_keep_str
        out_file_plot = os.path.join(out_dir,'_'.join(out_file_str)+'.jpg')
        xAndYs = [(k_range, precisions), (k_range, recalls), (k_range, f1s)]
        legend_entries = ['Precision','Recall','F1']
        ylabel = ''
        xlabel = 'Num Neighbors'

        
        feat_keep_str = ['+'.join([str_curr.upper() for str_curr in feat_keep_str])]
        title = ', '.join([str_curr.title() for str_curr in ['t='+str(inc),data_type, test_scheme]]+feat_keep_str)
        print title
        print np.mean(precisions), np.mean(recalls), np.mean(f1s)

        visualize.plotSimple(xAndYs, xlabel = xlabel, ylabel = ylabel, out_file = out_file_plot, legend_entries = legend_entries, title = title)

        print out_file_plot
        print ''

    visualize.writeHTMLForFolder(out_dir)
        

    util.writeFile(out_file, to_print)

def plot_frquency_distribution(inc, step_size):
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict)
    all_aus_org = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])

    out_dir = '../experiments/au_count_freq_inc_'+str(inc)+'_step_'+str(step_size)
    # +'_k_'+str(k)
    util.mkdir(out_dir)
    
    feat_keeps = [['au','ad','ead']]
    # ['au']]
    # ,['au','ad'],None]    
    data_types = ['frequency']
    # norms = ['l2_mean_std']
    # log_reg_params = {'metric': 'cosine', 'algorithm': 'brute'}
    
    # log_reg_str = []
    # for key_curr in log_reg_params:
    #     log_reg_str.append(key_curr)
    #     log_reg_str.append(log_reg_params[key_curr])
    # log_reg_str = '_'.join([str(val) for val in log_reg_str])
    
    

    to_print = []
    for (data_type, feat_keep) in itertools.product(data_types,  feat_keeps):
        if feat_keep is None:
            feat_keep_str = ['all']
        else:
            feat_keep_str = feat_keep
        
        file_str =feat_keep_str+[data_type, inc, step_size]
        
        # str_curr = ' '.join([data_type, str(norm)]+feat_keep_str)

        # to_print.append(str_curr)
        # print str_curr
        # title = str_curr 
        
        features, labels, all_aus = get_start_stop_feat(data_dict, all_aus_org, key_arr, inc, data_type, feat_keep = feat_keep, step_size = step_size)
        # print features[labels==2]
        # print all_aus
        # raw_input()
        # print inc, features.shape

        bin_pain = np.in1d(labels, pain)
    
        class_pain = np.zeros(bin_pain.shape)
        class_pain[bin_pain]=1
        class_pain = class_pain.astype(int)
        print features

        print features.shape
        num_aus = np.sum(features,axis = 1)
        # print num_aus.shape
        num_aus_p = num_aus[class_pain>0]
        num_aus_np = num_aus[class_pain<=0]
        # print 'num_aus_p',num_aus_p.shape, np.min(num_aus_p), np.max(num_aus_p), np.mean(num_aus_p)
        # print 'num_aus_np',num_aus_np.shape, np.min(num_aus_np), np.max(num_aus_np), np.mean(num_aus_np)



        num_bins = range(int(np.max(num_aus))+2)
        # ,range(features.shape[1]+2)]
        legend_entries = ['Pain','No Pain']
        vals = [num_aus_p, num_aus_np]

        hist_p,_ = np.histogram(num_aus_p,num_bins)
        hist_np,_ = np.histogram(num_aus_np,num_bins)
        P = hist_p / np.linalg.norm(hist_p, ord=1)
        Q = hist_np / np.linalg.norm(hist_np, ord=1)
        # P = Q
        # P = P / np.linalg.norm(P, ord=1)

        # kl1 = scipy.stats.entropy(P,Q, base = 2)
        # kl2 = scipy.stats.entropy(Q,P, base = 2)
        diffs = hist_np-hist_p
        idx_max = np.argmax(diffs)
        val_max = num_bins[idx_max+1]


        idx_keep = num_aus>idx_max
        features = features[idx_keep,:]
        class_pain = class_pain[idx_keep]
        bin_pain = bin_pain[idx_keep]
        # title = file_str[:]
        
        file_str.extend(['k',val_max])
        file_str = [str(val) for val in file_str]

        title = ' '.join(file_str)
        out_file = os.path.join(out_dir, '_'.join(file_str)+'.jpg')
        
        M = 0.5 * (P + Q)
        jsd = 0.5 * (scipy.stats.entropy(P, M, base = 2) + scipy.stats.entropy(Q, M, base = 2))

        # print 'kl1',kl1,'kl2',kl2,'sum',kl1+kl2,'jsd',jsd

        # title = 't='+str(inc)+', '+'+'.join([str_curr.upper() for str_curr in feat_keep_str])

        # out_file = os.path.join(out_dir, 'hist_'+log_reg_str+'.jpg')

        print title+' %.3f'% jsd
        xlabel = 'Number of AUs'
        ylabel = 'Frequncy'
        cumulative = False
        xtick_labels = [str(val) for val in num_bins[:-1]]
        visualize.plotMultiHist(out_file,vals = vals, num_bins = [num_bins, num_bins], legend_entries = legend_entries, title = title, xlabel = xlabel, ylabel = ylabel, xticks = xtick_labels, cumulative = cumulative)


def plot_per_au_freq(inc):
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict)
    all_aus_org = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])
    
    out_dir = '../experiments/au_type_freq'+str(inc)
    # +'_k_'+str(k)
    util.mkdir(out_dir)
    
    feat_keeps = [['au'],['au','ad']]    
    data_types = ['frequency']
    norms = ['l2_mean_std']
    max_diff = False
    # log_reg_params = {'metric': 'cosine', 'algorithm': 'brute'}
    
    # log_reg_str = []
    # for key_curr in log_reg_params:
    #     log_reg_str.append(key_curr)
    #     log_reg_str.append(log_reg_params[key_curr])
    # log_reg_str = '_'.join([str(val) for val in log_reg_str])
    
    

    to_print = []
    for (data_type, norm, feat_keep) in itertools.product(data_types, norms, feat_keeps):
        if feat_keep is None:
            feat_keep_str = ['all']
        else:
            feat_keep_str = feat_keep
        
        log_reg_str = '_'.join([str(val) for val in feat_keep_str])
        
        log_reg_str = log_reg_str+'_'+str(max_diff)

        out_file = os.path.join(out_dir, 'hist_'+log_reg_str+'.jpg')

        str_curr = ' '.join([data_type, str(norm)]+feat_keep_str)

        to_print.append(str_curr)
        print str_curr
        title = str_curr 
        
        features, labels, all_aus = get_start_stop_feat(data_dict, all_aus_org, key_arr, inc, data_type, feat_keep = feat_keep)

        bin_pain = np.in1d(labels, pain)
    
        class_pain = np.zeros(bin_pain.shape)
        class_pain[bin_pain]=1
        class_pain = class_pain.astype(int)

        feat_keep_str = ['+'.join([str_curr.upper() for str_curr in feat_keep_str])]
        if max_diff:
            features_bin = np.zeros(features.shape)
            features_bin[features>0]= 1    
            num_aus = np.sum(features_bin,axis = 1)
            num_aus_p = num_aus[class_pain>0]
            num_aus_np = num_aus[class_pain<=0]
            bin_range = range(features.shape[1]+2)
            
            hist_p,bin_edges = np.histogram(num_aus_p,bin_range)
            hist_np,bin_edges = np.histogram(num_aus_np,bin_range)
            
            diffs = hist_np-hist_p
            idx_max = np.argmax(diffs)
            val_max = bin_edges[idx_max+1]

            idx_keep = num_aus>idx_max
            features = features[idx_keep,:]
            class_pain = class_pain[idx_keep]
            feat_keep_str.append('k='+str(val_max))
        else:
            feat_keep_str.append('k not set')


        features_p = features[class_pain>0,:]
        features_np = features[class_pain<=0,:]
        hist_p = np.sum(features_p,axis = 0)
        hist_np = np.sum(features_np, axis = 0)
        hist_p = hist_p/np.linalg.norm(hist_p, ord = 1)
        hist_np = hist_np/np.linalg.norm(hist_np, ord = 1)


        diffs_p = hist_p - hist_np
        idx_sort = np.argsort(diffs_p)[::-1]
        p_diff = diffs_p[idx_sort]
        p_diff[p_diff<0]=0
        np_diff = -1*diffs_p[idx_sort]
        np_diff[np_diff<0]=0

        dict_vals = {'Pain':p_diff,'No Pain': np_diff}
        legend_vals = ['Pain','No Pain']
        ylabel = 'Positive Difference'
        xtick_labels = np.array(all_aus)[idx_sort]
        xlabel = 'AU Type'
        colors = ['b','g']
        title = 't='+str(inc)+', '+', '.join(feat_keep_str)
        out_file_diff = out_file[:out_file.rindex('.')]+'_diff.jpg'
        visualize.plotGroupBar(out_file_diff,dict_vals = dict_vals,xtick_labels = xtick_labels,legend_vals = legend_vals,colors = colors,xlabel=xlabel,ylabel=ylabel,title=title, width = 0.5)



        dict_vals = {'Pain':hist_p,'No Pain': hist_np}
        legend_vals = ['Pain','No Pain']
        ylabel = 'Frequency'
        xtick_labels = all_aus
        xlabel = 'AU Type'
        colors = ['b','g']
        title = 't='+str(inc)+', '+', '.join(feat_keep_str)

        visualize.plotGroupBar(out_file,dict_vals = dict_vals,xtick_labels = xtick_labels,legend_vals = legend_vals,colors = colors,xlabel=xlabel,ylabel=ylabel,title=title, width = 0.5)

def get_duration_stats():
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict)
    all_aus_org = get_all_aus(data_dict)
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])
    
    out_dir = '../experiments/au_duration_freq'
    util.mkdir(out_dir)
    
    feat_keeps = ['au','ad']
    print all_aus_org
    
    # au_dict = {}
    # for au_curr in all_aus_org:
    #     keep = False
    #     for feat_keep in feat_keeps:
    #         if feat_keep in au_curr:
    #             keep = True
    #             break
        
    #     if not keep:
    #         continue


    # vid_names = []
    # duration_arr = []
    # for 

    # data_dict[vid_number] = [facs_list, duration_list, start_time_list, end_time_list]


    au_duration_dict = {}
    au_list = []
    vid_name = []
    duration_list = []
    for k in data_dict.keys():
        # print k,len(data_dict[k][1])
        num_occ = len(data_dict[k][1])

        vid_name.extend([k]*num_occ)
        duration_list += data_dict[k][1]
        au_list += data_dict[k][0]

    # print len(vid_name), len(duration_list), len(au_list)
    arr = [vid_name, duration_list, au_list]
    for idx_arr_curr, arr_curr in enumerate(arr):
        # print len(arr_curr)
        arr[idx_arr_curr] = np.array(arr_curr)
        # print arr[idx_arr_curr].shape
        # print np.unique(arr[idx_arr_curr])

    [vid_name, duration_list, au_list] = arr

    # get max duration
    max_duration = np.max(duration_list)
    # decide inc for duration
    inc = 1
    max_duration = np.floor(max_duration)
    bin_range = np.arange(0,max_duration+(inc*2),inc)

    # for each au get hist pain and no pain
    bin_pain = np.in1d(vid_name, pain)
    # print np.sum(bin_pain), bin_pain.shape

    for au_curr in np.unique(au_list):
        bin_au = au_list==au_curr
        rel_bin = np.logical_and(bin_pain, bin_au)
        rel_bin_np = np.logical_and(~bin_pain, bin_au)
        dur_p = duration_list[rel_bin]
        dur_np = duration_list[rel_bin_np]
        
        max_duration = np.max(duration_list[bin_au])
        num_occ = np.sum(bin_au)
        num_occ = 2 if num_occ==1 else num_occ
        # print num_occ
        # print max_duration
        bin_range = np.linspace(0,max_duration,num = min(num_occ,10), endpoint = True)
        




        hist_p,bin_edges = np.histogram(dur_p,bin_range)
        hist_np,bin_edges = np.histogram(dur_np,bin_range)
        
        diffs = hist_np-hist_p
        idx_max = np.argmax(diffs)
        val_max = bin_edges[idx_max+1]

        hist_p = hist_p/np.linalg.norm(hist_p, ord = 1)
        hist_np = hist_np/np.linalg.norm(hist_np, ord = 1)

        # print hist_p.shape,dur_p.shape
        # print hist_np.shape,dur_np.shape
        hist_np [ np.isnan(hist_np)] = 0
        hist_p [ np.isnan(hist_p)] = 0

        # raw_input()
        # print hist_p.shape
        # print hist_np.shape


        diffs_p = hist_p - hist_np
        
        dict_vals = {'Pain':hist_p,'No Pain': hist_np}
        legend_vals = ['Pain','No Pain']
        ylabel = 'Frequency'
        # print bin_edges.shape
        # print hist_p.shape
        # print hist_np.shape
        xtick_labels = ['%.2f'%edge_curr for edge_curr in bin_edges[1:]]
        xlabel = 'Duration'
        colors = ['b','g']
        title = au_curr.upper()
        out_file_diff = os.path.join(out_dir, au_curr+'.jpg')

        visualize.plotGroupBar(out_file_diff,dict_vals = dict_vals,xtick_labels = xtick_labels,legend_vals = legend_vals,colors = colors,xlabel=xlabel,ylabel=ylabel,title=title, width = 0.5)
        # raw_input()

    visualize.writeHTMLForFolder(out_dir)


    # plot it



    # for k in data_dict.keys():
    #     print k
    #     print len(data_dict[k])
    #     print data_dict[k][0][3]
    #     print data_dict[k][1][3]
    #     print data_dict[k][2][3]
    #     print data_dict[k][3][3]
    #     break
    

def testing_clinical():
    file_name = '../data/FILM1-12Start_STOP_final_27.11.18.csv'
    data_dict = read_start_stop_anno_file(file_name)
    data_dict = clean_data(data_dict)
    all_aus_org = get_all_aus(data_dict)
    
    key_arr = range(1,13)
    pain = np.array([1,2,4,5,11,12])
    
    inc = 2
    data_type = 'frequency'
    feat_keep = ['au','ad']    
    norm = ['l2_mean_std']
    log_reg_params = {'metric': 'cosine', 'algorithm': 'brute'}
    test_scheme = 'majority'
    k_range = [1, 8, 15, 23, 30, 37, 45]
    nn, f1s, features, labels, class_pain, all_aus, class_pain_ts, pred_ts  = select_knn(data_dict, all_aus_org, key_arr, pain, inc, data_type, feat_keep, norm, log_reg_params, test_scheme, k_range, b_type = True)
    
    print features.shape
    for au in all_aus:
        print au

    # test_scheme = 'atleast_one'



    file_name = '../data/clinical_cases_wb1.csv'
    data_dict_t = read_start_stop_anno_file(file_name)
    data_dict_t = clean_data(data_dict_t)
    all_aus_org_t = get_all_aus(data_dict_t)
    feat_keep_t = ['exact']+all_aus
    # print feat_keep_t
    # print type(all_aus_org_t)
    # print '____'
    # print len(all_aus_org_t)
    # for au in all_aus_org_t:
    #     print au
    
    key_arr_t = []
    for k in data_dict_t.keys():
        key_arr_t.append(k)
    
    features_t, labels_t, all_aus_t = get_start_stop_feat(data_dict_t, all_aus_org_t, key_arr_t, inc, data_type, feat_keep = feat_keep_t)
    class_pain_t = np.ones((len(labels_t),))
    print class_pain_t.shape

    dist, indices_total = nn.kneighbors(features_t)
    
    precisions = []
    recalls = []
    f1s = []
    # print indices_total.shape
    # k_range = [1,3,5,7,9,11]
    class_pain_ts = []
    pred_ts = []
    str_print = []

    for k in k_range:
        indices = indices_total[:,:k]

        bin_count = np.zeros(indices.shape)
        for idx, indices_curr in enumerate(indices):
            bin_count[idx,:] = class_pain[indices_curr]

        majority_labels = np.sum(bin_count,axis = 1)
        half_less = k//2
        bin_maj = majority_labels>half_less
        majority_labels[bin_maj] = 1
        majority_labels[~bin_maj] = 0
        # print majority_labels.shape
        # print labels_t.shape
        str_print.append('K=%d' % k)
        print str_print[-1]

        for label in np.unique(labels_t):
            rel_preds = majority_labels[labels_t==label]
            pain_percent = np.sum(rel_preds)/float(rel_preds.size)
            
            
            str_print.append('Film %d, Pain segments %.2f' % (label, pain_percent*100))
            print str_print[-1] 

        if test_scheme=='All':
            class_pain_v= class_pain
            pred_v = majority_labels
        else:
            labels_v, class_pain_v, pred_v = transform_labels(labels_t, class_pain_t, majority_labels, test_scheme)
            
        # print class_pain_v, pred_v, labels_v
        # raw_input()
    out_file = '../experiments/knn_clinical_cases.txt'
    util.writeFile(out_file, str_print)
    
    # get_duration_stats()
    # loo_log_reg()


def main():
    inc = 5
    step_size = 2.5

    plot_frquency_distribution(inc, step_size)
    # script_plot_cooc()
    # testing_clinical()    
    # get_duration_stats()

    return

    inc_curr = [2,5,10,15,30]
    num_vids = 12
    duration = 30
    num_k = 7

    for inc in inc_curr:
        total_segs = num_vids*duration/inc
        k_range = np.unique(np.linspace(1, total_segs//4, num=num_k, dtype=int))
        print inc, k_range
        knn(k_range, inc)
        raw_input()
        

    # for inc in [2,5,10,15,30]:
    #     plot_per_au_freq(inc)
        # plot_frquency_distribution(inc)

    # return 
    # inc = 5
    # k_range = [1,3,5,7,9,11]
    # knn(k_range, inc)

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