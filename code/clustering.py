import sklearn
from sklearn import preprocessing,decomposition,discriminant_analysis
from helpers import util, visualize
from read_in_data import *

def make_data_mat_frequency(data_dict, all_aus, key_arr ):
    all_counts = np.zeros((len(key_arr),len(all_aus)))
    for idx_k, k in enumerate(key_arr):
        au_anno = data_dict[k][0]
        
        for idx_au, au_curr in enumerate(all_aus):
            all_counts[idx_k,idx_au] = au_anno.count(au_curr)

        assert sum(all_counts[idx_k,:])==len(au_anno)

    
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

        assert np.abs(np.sum(all_counts[idx_k,:]) - np.sum(time_anno))<1e-10

    
    return all_counts

def script_pca():
    dir_data = '../data'
    out_dir = '../experiments'
    out_dir = os.path.join(out_dir,'pca_film_data_12')
    util.mkdir(out_dir)

    file_data = os.path.join(dir_data,'Film_data_.csv')
    data_dict = read_anno_file(file_data)
    data_dict = clean_data(data_dict)

    all_aus = get_all_aus(data_dict)
    key_arr = range(1,13)
    no_pain = np.array([3,6,7,8,9,10])
    pain = np.array([1,2,4,5,11,12])
    bin_pain = np.in1d(key_arr, pain)
    bin_no_pain = np.in1d(key_arr, no_pain)

    
    for data_type in ['frequency', 'duration', 'both', 'both_normalized','duration_normalized']:
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
    

def main():

    # script_pca()

    # return

    dir_data = '../data'
    out_dir = '../experiments'
    out_dir = os.path.join(out_dir,'lda_film_data_12_withLabels')
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

    # data_types = [ 'frequency','duration','both', 'both_normalized', 'duration_normalized']
    data_types = [ 'duration_normalized']
    for data_type in data_types:
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

        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(all_counts)
        data_pca = scaler.transform(all_counts)
        
        # scaler = sklearn.preprocessing.Normalizer()
        # scaler.fit(data_pca)
        # data_pca = scaler.transform(data_pca)

        pca = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        pca.fit(data_pca, class_pain)
        
        print pca.score(data_pca,class_pain)
        
        data_pca = pca.predict_proba(data_pca) 
        # data_pca = np.concatenate([data_pca, data_pca],axis = 1)
        # print data_pca.shape
        
        variance = pca.explained_variance_ratio_
        print 'variance',variance
        print 'pca.coef_',pca.coef_
        print 'pca.intercept_',pca.intercept_

        # first_two = data_pca[:,:2]
        x_label = 'Probability No Pain'
        y_label = 'Probability Pain'
        
        print data_type
        print x_label
        print y_label

        # file_str = data_type
        # title = ' '.join([val.title() for val in file_str.split('_')])
        # out_file = os.path.join(out_dir,file_str+'.jpg')
        # visualize.plotSimple([(data_pca[:,0],data_pca[:,1])],title = title, xlabel = x_label, ylabel = y_label, out_file = out_file, noline = True)

        file_str = data_type+'_pain_no_pain'
        title = ' '.join([val.title() for val in file_str.split('_')])
        out_file = os.path.join(out_dir,file_str+'.jpg')
        legend_entries = ['Pain','No Pain']
        data_plot = []
        for bin_curr in [bin_pain, bin_no_pain]:
            data_plot.append((data_pca[bin_curr,0],data_pca[bin_curr,1]))
        visualize.plotSimple(data_plot,title = title, xlabel = x_label, ylabel = y_label, out_file = out_file, noline = True, legend_entries = legend_entries, mark_labels = [label_pain, label_no_pain])    
        # break
    visualize.writeHTMLForFolder(out_dir)




if __name__=='__main__':
    main()

