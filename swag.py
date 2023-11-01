cp.ifov=32
for cp.ifov in cp.completed_fovs:#range(100):
    try:
        cp.set_ = ''
        fls = cp.dic_fls[cp.ifov]
        fls = reorder_on_R(fls)
        cp.fov = get_fullfov(fls[0])
        save_fld_cell = cp.save_folder+os.sep+'best_per_cell'
        cp.save_fl = save_fld_cell+os.sep+cp.fov+'--'+cp.set_+'__XHfs_finedrft.npz'
        save_fl = cp.save_fl
        save_flf= save_fl.replace('.npz','_save.npz')
        
        
        XH_fs = np.load(save_fl)['XH_fs']
        iRs = XH_fs[:,-2].astype(int)
        #XH_fs[:,:3]+=(Xf_dif/cp.pix)[iRs]
        get_X_cands(cp,keep_best_per_cell(XH_fs,8),nchr_=3,pix=[0.200,0.1083,0.1083],
                             radius_chr = 2,enhanced_radius=2,radius_cand =2,fr_th=0.4,nelems=4,plt_val = False)
        
        
        uRs = np.unique(XH_fs[:,-2],axis=-1).astype(int)
        initialize_with_max_brightness(cp,nkeep = 8000,Rs_u = uRs)
        
        normalize_color_brightnesses(cp)
        
        run_EM(cp,nkeep = 80000,niter = 4,Rs_u = uRs)
        
        
        get_scores_and_threshold(cp,th_score = -2.5)
        plot_matrix(cp,th_score=-2.5,lazy_color_correction = True,vmin=0,vmax=0.5)
        plt.show()
        
        if False:
            for niter in range(0):
                ncol=3
                Xf = np.array(cp.zxys_f)
                bad = np.log(cp.scores_f)<-2.5
                Xf[bad] = np.nan
                cms = np.nanmean(Xf,axis=1)
                Xf_dif = np.nanmean(Xf-cms[:,np.newaxis],axis=0)
            
                elems = np.nanmean([Xf_dif[icol::ncol] for icol in np.arange(ncol)],axis=0)
                Xf_dif = np.array([e for e in elems for icol in np.arange(ncol)])[:len(Xf_dif)]
            
            
                XH_fs[:,:3]-=(Xf_dif/cp.pix)[iRs]
                #rad = 1.5+0.5**niter
                get_X_cands(cp,keep_best_per_cell(XH_fs,8),nchr_=3,pix=[0.200,0.1083,0.1083],
                                 radius_chr = 2,enhanced_radius=2,radius_cand =2,fr_th=0.4,nelems=4,plt_val = False)
            
                uRs = np.unique(XH_fs[:,-2],axis=-1).astype(int)
                initialize_with_max_brightness(cp,
                                               nkeep = 8000,Rs_u = uRs)
            
                normalize_color_brightnesses(cp)
            
                run_EM(cp,nkeep = 80000,niter = 4,Rs_u = uRs)
            
            
                get_scores_and_threshold(cp,th_score = -2.5)
                #plot_matrix(cp,th_score=-2.5,lazy_color_correction = True)
                plot_matrix(cp,th_score=-2.5,lazy_color_correction = True,vmin=0,vmax=0.5)
                plt.show()
        cells = cp.icell_cands
        ifov = get_fov(save_fl)
        cells_f = np.array(cells)+ifov*10**5
        np.savez(save_flf,zxys_f = cp.zxys_f,hs_f = cp.hs_f,cells_f=cells_f,scores_f=cp.scores_f)
    except:
        print("Failed!!  ",cp.ifov)
