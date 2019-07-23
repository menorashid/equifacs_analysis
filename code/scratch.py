# dir_meta = '../experiments/comparing_frequency_duration_kunz_core'
    pre_path = '\includegraphics[width = 0.19\\textwidth]{'
    dir_meta = './frequency_duration'
    aus = ['au17','ad38','au47','au101','ead104']
    dirs = [os.path.join(dir_meta,'count'),dir_meta,
    os.path.join(dir_meta,'max_duration'),dir_meta,
        os.path.join(dir_meta,'duration')]
    
    # print len(dirs)
    # raw_input()

    pre_im = ['count_','','duration_','dur_','duration_']
    post_im = ['','_per_horse_hist',
    '','_per_horse_hist',
    '']
    post_im = [val+'.jpg' for val in post_im]

    for idx_row in range(len(aus)):
        print aus[idx_row].upper()+' & '
        for idx_col in range(len(dirs)):
            path_curr = os.path.join(dirs[idx_col],pre_im[idx_col]+aus[idx_row]+post_im[idx_col])

            path_curr = pre_path+path_curr+'}'
            if idx_col+1==len(dirs):
                print path_curr+' \\\\'
                print '\\hline'
            else:
                print path_curr+' & '