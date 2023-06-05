#activate cellpose&&python D:\Carlos\Scripts\worker.py


from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys,numpy as np

master_analysis_folder = r'D:\Carlos\Scripts'
sys.path.append(master_analysis_folder)
from ioMicro import *


def compute_drift(save_folder,fov,all_flds,set_,redo=False):
    """
    save_folder where to save to - analysis_fodler
    fov - i.e. Conv_zscan_005.zarr
    all_flds - folders that contain eithger the MERFISH bits or control bits or sm
   
    """
    #print(len(all_flds))
    #print(all_flds)
    drift_fl = save_folder+os.sep+'drift_'+fov.split('.')[0]+'--'+set_+'.pkl'
    iiref = None
    previous_drift = {}
    if not os.path.exists(drift_fl):
        redo = True
    else:
        drifts_,all_flds_,fov_ = pickle.load(open(drift_fl,'rb'))
        all_tags_ = np.array([os.path.basename(fld)for fld in all_flds_])
        all_tags = np.array([os.path.basename(fld)for fld in all_flds])
        iiref = np.argmin([np.sum(np.abs(drift[0]))for drift in drifts_])
        previous_drift = {tag:drift for drift,tag in zip(drifts_,all_tags_)}
        if not (len(all_tags_)==len(all_tags)):
            redo = True
        else:
            if not np.all(np.sort(all_tags_)==np.sort(all_tags)):
                redo = True
    if redo:
        print("Computing drift...")
        ims = [read_im(fld+os.sep+fov) for fld in all_flds] #map the image
        ncols,sz,sx,sy = ims[0].shape
        if iiref is None: iiref = len(ims)//2
        im_ref = np.array(ims[iiref][-1],dtype=np.float32)
        all_tags = np.array([os.path.basename(fld)for fld in all_flds])
        drifts = [previous_drift.get(tag,get_txyz(im[-1],im_ref,sz_norm=20, sz=400))
                    for im,tag in zip(tqdm(ims),all_tags)]
       
        pickle.dump([drifts,all_flds,fov],open(drift_fl,'wb'))
def compute_fits(save_folder,fov,all_flds,redo=False,ncols=4):
    for fld in tqdm(all_flds):
       
        #ncols = len(im_)
        for icol in range(ncols-1):
            #icol=2
            tag = os.path.basename(fld)
            save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npz'
            if not os.path.exists(save_fl) or redo:
                im_ = read_im(fld+os.sep+fov)
                #print("Reading image")
                im__ = np.array(im_[icol],dtype=np.float32)
               
                im_n = norm_slice(im__,s=30)
                #print("Fitting image")
                Xh = get_local_max(im_n,500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,dbscan=True,
                      return_centers=False,mins=None,sigmaZ=1,sigmaXY=1.5)
                np.savez_compressed(save_fl,Xh=Xh)
def compute_decoding(save_folder,fov,set_):
    dec = decoder_simple(save_folder,fov,set_)
    complete = dec.check_is_complete()
    if complete==0:
        dec.get_XH(fov,set_,ncols=3)#number of colors match
        dec.XH = dec.XH[dec.XH[:,-4]>0.25] ### keep the spots that are correlated with the expected PSF for 60X
        dec.load_library(lib_fl = r'\\192.168.0.10\bbfishdc13\codebook_0_New_DCBB-300_MERFISH_encoding_2_21_2023.csv',nblanks=-1)
        dec.get_inters(dinstance_th=2,enforce_color=True)# enforce_color=False
        dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=-1)
def main_f(set_ifov):
    save_folder =r'\\merfish7\merfish7v1\DNA_FISH\Induced_cardio_05_04_2023\Analysis' ### save folder
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    set__ = ''
    all_flds = glob.glob(r'\\merfish7\merfish7v1\DNA_FISH\Induced_cardio_05_04_2023\H*R*'+set__)
    all_flds += glob.glob(r'\\merfish7\merfish7v1\DNA_FISH\Induced_cardio_05_04_2023\H*I*'+set__)
    all_flds += glob.glob(r'\\merfish7\merfish7v1\DNA_FISH\Induced_cardio_05_04_2023\H*Q*'+set__)
    all_flds += glob.glob(r'\\merfish7\merfish7v1\DNA_FISH\Induced_cardio_05_04_2023\H*B*'+set__)
    set_,ifov = set_ifov
    all_flds = [fld.replace(set__,set_) for fld in all_flds]
    fovs_fl = save_folder+os.sep+'fovs__'+set_+'.npy'
    if not os.path.exists(fovs_fl):
        fls = glob.glob(all_flds[0]+os.sep+'*.zarr')
        fovs = [os.path.basename(fl) for fl in fls]
        np.save(fovs_fl,fovs)
    else:
        fovs = np.load(fovs_fl)
    if ifov<len(fovs):
        fov = fovs[ifov]
        try:
            print("Computing fitting on: "+str(fov))
            print(len(all_flds),all_flds)
            compute_fits(save_folder,fov,all_flds,redo=False)
            print("Computing drift on: "+str(fov))
            compute_drift(save_folder,fov,all_flds,set_,redo=False)
           
            #compute_decoding(save_folder,fov,set_)
        except:
            print("Failed:",fov,set_)
    return set_ifov
if __name__ == '__main__':
    # start 4 worker processes
    items = [(set_,ifov)for set_ in ['']
                        for ifov in range(1000)]
                       
    #main_f(['',47])
    if True:
        with Pool(processes=8) as pool:
            print('starting pool')
            result = pool.map(main_f, items)