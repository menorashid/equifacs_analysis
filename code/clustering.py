import sklearn
from sklearn import preprocessing,decomposition,discriminant_analysis,pipeline
from helpers import util, visualize
from read_in_data import *

def make_data_mat_frequency(data_dict, all_aus, key_arr ):
    all_counts = np.zeros((len(key_arr),len(all_aus)))
    for idx_k, k in enumerate(key_arr):
        au_anno = data_dict[k][0]
        
        for idx_au, au_curr in enumerate(all_aus):
            all_counts[idx_k,idx_au] = au_anno.count(au_curr)

        # assert sum(all_counts[idx_k,:])==len(au_anno)

    
    return all_counts



def make_data_mat_duration(data_dict, all_aus, key_arr ):
    
    all_counts = np.zeros((len(key_arr),len(all_aus)))
    for idx_k, k in enumerate(key_arr):
        au_anno = data_dict[k][0]
        au_anno = np.array(au_anno)
        time_anno = np.array(data_dict[k][1])
        
        
        for idx_au, au_curr in enumerate(all_aus):

            all_counts[idx_k,idx_au] = np.sum(time_anno[au_anno==au_curr])
            # au_anno.count(au_curr)

        # assert np.abs(np.sum(all_counts[idx_k,:]) - np.sum(time_anno))<1e-10

    
    return all_counts

def get_all_data(dir_data):
    file_data = os.path.join(dir_data,'Film_data_.csv')
    data_dict = read_anno_file(file_data)
    data_dict = clean_data(data_dict)
    all_aus = get_all_aus(data_dict)
    return data_dict, all_aus

def script_pca():
    dir_data = '../data'
    out_dir = '../experiments'
    out_dir = os.path.join(out_dir,'pca_film_data_12')
    util.mkdir(out_dir)

    # file_data = os.path.join(dir_data,'Film_data_.csv')
    # data_dict = read_anno_file(file_data)
    # data_dict = clean_data(data_dict)
    # all_aus = get_all_aus(data_dict)

    data_dict, all_aus = get_all_data(dir_data)
    key_arr = range(1,13)
    no_pain = np.array([3,6,7,8,9,10])
    pain = np.array([1,2,4,5,11,12])
    bin_pain = np.in1d(key_arr, pain)
    bin_no_pain = np.in1d(key_arr, no_pain)

    
    for data_type in ['frequency', 'duration', 'both', 'both_normalized','duration_normalized']:
        
        all_counts = get_data_by_type(data_dict, all_aus, key_arr, data_type)

        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(all_counts)
        data_pca = scaler.transform(all_counts)
        pca = sklearn.decomposition.PCA( whiten=True)
        data_pca = pca.fit_transform(data_pca)
        variance = pca.explained_variance_ratio_
        
        first_two = data_pca[:,:2]
        x_label = 'First Component %.2f' % (variance[0]*100)
        y_label = 'Second Component %.2f' % (variance[1]*100)
        
        print data_type
        print x_label
        print y_label

        file_str = data_type
        title = ' '.join([val.title() for val in file_str.split('_')])
        out_file = os.path.join(out_dir,file_str+'.jpg')
        visualize.plotSimple([(data_pca[:,0],data_pca[:,1])],title = title, xlabel = x_label, ylabel = y_label, out_file = out_file, noline = True)

        file_str = data_type+'_pain_no_pain'
        title = ' '.join([val.title() for val in file_str.split('_')])
        out_file = os.path.join(out_dir,file_str+'.jpg')
        legend_entries = ['Pain','No Pain']
        data_plot = []
        for bin_curr in [bin_pain, bin_no_pain]:
            data_plot.append((data_pca[bin_curr,0],data_pca[bin_curr,1]))
        visualize.plotSimple(data_plot,title = title, xlabel = x_label, ylabel = y_label, out_file = out_file, noline = True, legend_entries = legend_entries)    

    visualize.writeHTMLForFolder(out_dir)
    
def get_data_by_type(data_dict, all_aus, key_arr, data_type):
    if data_type == 'frequency':
        all_counts = make_data_mat_frequency(data_dict, all_aus, key_arr)
    elif data_type =='duration':
        all_counts = make_data_mat_duration(data_dict, all_aus, key_arr)
    elif data_type =='both_normalized':
        all_counts = make_data_mat_frequency(data_dict, all_aus, key_arr)
        all_counts_d = make_data_mat_duration(data_dict, all_aus, key_arr)
        div_d = all_counts
        div_d[div_d==0]=1
        all_counts_d = all_counts_d/div_d
        all_counts = np.concatenate([all_counts, all_counts_d], axis = 1)
    elif data_type == 'duration_normalized':
        all_counts = make_data_mat_frequency(data_dict, all_aus, key_arr)
        all_counts_d = make_data_mat_duration(data_dict, all_aus, key_arr)
        div_d = all_counts
        div_d[div_d==0]=1
        all_counts = all_counts_d/div_d
    elif data_type =='both':           
        all_counts = make_data_mat_frequency(data_dict, all_aus, key_arr)
        all_counts_d = make_data_mat_duration(data_dict, all_aus, key_arr)
        all_counts = np.concatenate([all_counts, all_counts_d], axis = 1)
    else:
        raise ValueError('data_type '+str(data_type)+' no valid')

    return all_counts



def fit_lda(all_counts, class_pain, norm, priors = None):
    if norm =='l2':
        scaler = sklearn.preprocessing.Normalizer()
    elif norm=='l2_mean':
        scaler = sklearn.pipeline.Pipeline([('l2',sklearn.preprocessing.Normalizer()),
            ('mean',sklearn.preprocessing.StandardScaler(with_std = False))])
    elif norm=='l2_mean_std':
        scaler = sklearn.pipeline.Pipeline([('l2',sklearn.preprocessing.Normalizer()),
            ('mean',sklearn.preprocessing.StandardScaler())])
    elif norm=='mean':
        scaler = sklearn.preprocessing.StandardScaler(with_std = False)
    else:
        scaler = sklearn.preprocessing.StandardScaler()
    
    scaler.fit(all_counts)
    data_lda = scaler.transform(all_counts)
        
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(priors = priors)
    lda.fit(data_lda, class_pain)
    return lda, scaler, data_lda

def script_lda(norm = 'l2', feat_keep = None):
    dir_data = '../data'
    out_dir = '../experiments'
    # norm = 'mean'
    # norm = 'mean_std'

    out_dir_curr = ['lda_film_data_12_withLabels',norm]
    if feat_keep is not None:
        out_dir_curr = out_dir_curr+feat_keep
    
    out_dir_curr = '_'.join(out_dir_curr)
    out_dir = os.path.join(out_dir,out_dir_curr)
    util.mkdir(out_dir)

    # file_data = os.path.join(dir_data,'Film_data_.csv')
    # data_dict = read_anno_file(file_data)
    # data_dict = clean_data(data_dict)
    # all_aus = get_all_aus(data_dict)

    data_dict, all_aus = get_all_data(dir_data)

    if feat_keep is not None:
        bin_au = np.zeros((len(feat_keep), len(all_aus)))
        for idx_val,val in enumerate(all_aus): 
            for idx_feat, feat in enumerate(feat_keep):
                if feat in val:
                    bin_au[idx_feat, idx_val]=1
        bin_au = np.sum(bin_au,axis = 0)>0
        all_aus = [val for idx_val,val in enumerate(all_aus) if bin_au[idx_val]]
        assert len(all_aus)==np.sum(bin_au)

    print all_aus

    key_arr = range(1,13)
    # range(1,13)
    no_pain = np.array([3,6,7,8,9,10])
    pain = np.array([1,2,4,5,11,12])
    bin_pain = np.in1d(key_arr, pain)
    bin_no_pain = np.in1d(key_arr, no_pain)


    label_pain = [str(val) for val in np.array(key_arr)[bin_pain]]
    label_no_pain = [str(val) for val in np.array(key_arr)[bin_no_pain]]

    
    class_pain = np.zeros(bin_pain.shape)
    class_pain[bin_pain] = 1
    class_pain = class_pain.astype(int)
    
    data_types = ['frequency', 'duration', 'both', 'both_normalized','duration_normalized']
    
    for data_type in data_types:
        
        all_counts = get_data_by_type(data_dict, all_aus, key_arr, data_type)
        print all_counts.shape
        # all_counts = all_counts[:,bin_au]
        # print all_counts.shape

        # raw_input()
        
        
        # if norm =='l2':
        #     scaler = sklearn.preprocessing.Normalizer()
        # elif norm=='mean':
        #     scaler = sklearn.preprocessing.StandardScaler(with_std = False)
        # else:
        #     scaler = sklearn.preprocessing.StandardScaler()
        
        # scaler.fit(all_counts)
        # data_pca = scaler.transform(all_counts)
        
        # pca = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        # pca.fit(data_pca, class_pain)
        
        pca, scaler, data_pca = fit_lda(all_counts, class_pain, norm)
        score =pca.score(data_pca,class_pain) 
        preds = pca.predict_proba(data_pca) 
        weight = pca.coef_[0]
    
        print data_type, score
        
        
        x_label = 'Probability No Pain'
        y_label = 'Probability Pain'

        file_str = data_type+'_pain_no_pain'
        title = ' '.join([val.title() for val in file_str.split('_')]+['%.2f' % score])
        out_file = os.path.join(out_dir,file_str+'.jpg')
        legend_entries = ['Pain','No Pain']
        data_plot = []
        for bin_curr in [bin_pain, bin_no_pain]:
            data_plot.append((preds[bin_curr,0],preds[bin_curr,1]))
        visualize.plotSimple(data_plot,title = title, xlabel = x_label, ylabel = y_label, out_file = out_file, noline = True, legend_entries = legend_entries, mark_labels = [label_pain, label_no_pain])  

        out_dir_analysis = os.path.join(out_dir,data_type+'_weight_analysis')
        util.mkdir(out_dir_analysis)
        if 'both' in data_type:
            all_aus_curr = [val+' f' for val in all_aus]+[val+' d' for val in all_aus]
        else:
            all_aus_curr = all_aus

        do_weight_analysis(out_dir_analysis, data_pca, bin_pain, preds, weight, key_arr, all_aus_curr)

        # break
    visualize.writeHTMLForFolder(out_dir)


def do_weight_analysis(out_dir, data_pca, bin_pain, preds, weight_ac, horse_nums, all_aus ):
    # score = pca.score(data_pca,class_pain)
    # preds = pca.predict_proba(data_pca) 
    # weight = pca.coef_
    # bias = pca.intercept_
    # weight = weight[0]
    # print 'bias',bias
    
    avg_val = np.mean(data_pca, axis = 0)
    weight_rw = avg_val*weight_ac
    # print avg_val.shape, weight_ac.shape, weight_rw.shape
    # weight_rw = np.abs(weight_rw)
    for pre_str, weight in zip(['','RW '],[weight_ac, weight_rw]):
        pos_bin = weight>0
        neg_bin = weight<=0
        for idx_pos, bin_curr in enumerate([pos_bin, neg_bin]):
            pos_au = np.array(all_aus)[bin_curr]
            pos_val = np.abs(weight[bin_curr])
            pos_val = pos_val/np.sum(pos_val)
            
            idx_sort = np.argsort(pos_val)[::-1]
            pos_au = pos_au[idx_sort]
            pos_val = pos_val[idx_sort]
            ylabel = 'Percentage'
            xtick_labels = pos_au
            
            if idx_pos==0:
                title = pre_str+'Positive Pain Correlation'
                colors = ['b']
                dict_vals = {'Pos':pos_val}
                legend_vals = ['Pos']

            else:
                title = pre_str+'Negative Pain Correlation'
                colors = ['r']
                dict_vals = {'Neg':pos_val}
                legend_vals = ['Neg']


            out_file_str = '_'.join([val.lower() for val in title.split(' ')])
            out_file = os.path.join(out_dir,out_file_str+'.jpg')
            visualize.plotGroupBar(out_file ,dict_vals,xtick_labels,legend_vals,colors,xlabel='',ylabel = ylabel,title=title,width=1,ylim=None,loc=None) 

    weight = weight_ac
    for horse_num in horse_nums:
        twelve_val = data_pca[horse_num-1,:]



        dict_vals = {'Weight':weight/np.linalg.norm(weight),'Horse '+str(horse_num):twelve_val/np.linalg.norm(twelve_val)}
        legend_vals = ['Weight','Horse '+str(horse_num)]
        colors = ['b','r']
        ylabel = 'Weight'
        title = 'GT %d Pred %.2f' % (int(bin_pain[horse_num-1]),preds[horse_num-1,1])

        out_file = os.path.join(out_dir, 'weight_vs_'+str(horse_num)+'.jpg')
        xtick_labels = all_aus
        visualize.plotGroupBar(out_file ,dict_vals,xtick_labels,legend_vals,colors,xlabel='',ylabel = ylabel,title=title,width=0.4,ylim=None,loc=None)
       

def script_comparing_features():
    dir_data = '../data'
    out_dir = '../experiments'
    out_dir = os.path.join(out_dir,'lda_film_data_12_weight_analysis')
    util.mkdir(out_dir)


    file_data = os.path.join(dir_data,'Film_data_.csv')
    data_dict = read_anno_file(file_data)
    data_dict = clean_data(data_dict)

    all_aus = get_all_aus(data_dict)
    key_arr = range(1,13)
    # range(1,13)
    no_pain = np.array([3,6,7,8,9,10])
    pain = np.array([1,2,4,5,11,12])
    bin_pain = np.in1d(key_arr, pain)
    bin_no_pain = np.in1d(key_arr, no_pain)


    label_pain = [str(val) for val in np.array(key_arr)[bin_pain]]
    label_no_pain = [str(val) for val in np.array(key_arr)[bin_no_pain]]

    
    class_pain = np.zeros(bin_pain.shape)
    class_pain[bin_pain] = 1
    class_pain = class_pain.astype(int)+1

    data_type = 'duration'
    
    # print 'bias',bias
    
    all_counts = get_data_by_type(data_dict, all_aus, key_arr, data_type)
    
    data_type = 'duration_normalized'

    all_counts_dn = get_data_by_type(data_dict, all_aus, key_arr, data_type)

    for horse_num in [12]:
        twelve_val = all_counts_dn[horse_num-1,:]
        weight = all_counts[horse_num-1,:]

        dict_vals = {'Duration':weight/np.linalg.norm(weight),'Duration Normalized':twelve_val/np.linalg.norm(twelve_val)}
        legend_vals = ['Duration','Duration Normalized']
        colors = ['b','r']
        ylabel = 'Normalized Feature Value'
        title = 'Horse '+str(horse_num)+' Features'

        out_file = os.path.join(out_dir, 'duration_vs_duration_normalized_'+str(horse_num)+'.jpg')
        xtick_labels = all_aus
        visualize.plotGroupBar(out_file ,dict_vals,xtick_labels,legend_vals,colors,xlabel='',ylabel = ylabel,title=title,width=0.4,ylim=None,loc=None)


def verifying_data():
    dir_data = '../data'
    file_data = os.path.join(dir_data,'Film_data_.csv')
    data_dict = read_anno_file(file_data)
    data_dict = clean_data(data_dict)

    all_aus = get_all_aus(data_dict)
    key_arr = range(1,13)
    data_types = ['frequency', 'duration', 'both', 'both_normalized','duration_normalized']
    
    for data_type in data_types:
        
        all_counts = get_data_by_type(data_dict, all_aus, key_arr, data_type)
        print data_type
        for idx_r, r in enumerate(all_counts):
            print 'Video',idx_r+1
            for idx_r_curr,r_curr in enumerate(r):
                if r_curr>0:
                    print all_aus[idx_r_curr],r_curr
            print '__'
            raw_input()

def main():

    
    print 'hello'

    script_lda(feat_keep=['au','ad'])
    # script_lda_weight_analysis()
    # print 'pca.coef_',pca.coef_
    # print 'pca.intercept_',pca.intercept_


    # return

    




if __name__=='__main__':
    main()

