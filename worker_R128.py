# conda activate cellpose2&&python D:\Carlos\Scripts\workerDECONVOLUTIOND104rna.py
#
######### Please add documentation
### This is experiemnt D104, we are analyzing the big tad to supplement the samll tad. Date started: 9/15/23 @ 8:02pm by Carlos

# Have you copmputed the PSF?
#################################################################
import glob
import sys
import os
import numpy as np

from ioMicro import *

# standard is 4, its number of colors +1
ncols = 4

psf_file = r"psf_D103_B.npy"
#save_folder = r"/mnt/merfish11/20230919_R128_N5S1MERNeuro_analysis/RMERFISH/"
#save_folder = r"/mnt/merfish11/20230919_R128_N5S1MERControl_analysis/RMERFISH/"
save_folder = r"/mnt/renimaging2/02_20_2024_BigHeartR133S2_analysis/"
flat_field_fl = r"R128__med_col_raw"
lib_fl = "/home/plt3/MERlin_Parameters/codebooks/codebook_code_color2__QZBB_P3P4Heart_blank.csv"

def get_icodesV3(dec,nmin_bits=3,iH=-3):
    import time
    start = time.time()
    lens = dec.lens
    res_unfolder = dec.res_unfolder
    Mlen = np.max(lens)
    print("Calculating indexes within cluster...")
    res_is = np.tile(np.arange(Mlen), len(lens))
    res_is = res_is[res_is < np.repeat(lens, Mlen)]
    print("Calculating index of molecule...")
    ires = np.repeat(np.arange(len(lens)), lens)
    #r0 = np.array([r[0] for r in res for r_ in r])
    print("Calculating index of first molecule...")
    r0i = np.concatenate([[0],np.cumsum(lens)])[:-1]
    r0 = res_unfolder[np.repeat(r0i, lens)]
    print("Total time unfolded molecules:",time.time()-start)
    
    ### torch
    ires = torch.from_numpy(ires.astype(np.int64))
    res_unfolder = torch.from_numpy(res_unfolder.astype(np.int64))
    res_is = torch.from_numpy(res_is.astype(np.int64))
    
    import time
    start = time.time()
    print("Computing score...")
    scoreF = torch.from_numpy(dec.XH[:,iH])[res_unfolder]
    print("Total time computing score:",time.time()-start)
    
    
    ### organize molecules in blocks for each cluster
    def get_asort_scores():
        val = torch.max(scoreF)+2
        scoreClu = torch.zeros([len(lens),Mlen],dtype=torch.float64)+val
        scoreClu[ires,res_is]=scoreF
        asort = scoreClu.argsort(-1)
        scoreClu = torch.gather(scoreClu,dim=-1,index=asort)
        scoresF2 = scoreClu[scoreClu<val-1]
        return asort,scoresF2
    def get_reorder(x,val=-1):
        if type(x) is not torch.Tensor:
            x = torch.from_numpy(np.array(x))
        xClu = torch.zeros([len(lens),Mlen],dtype=x.dtype)+val
        xClu[ires,res_is] = x
        xClu = torch.gather(xClu,dim=-1,index=asort)
        xf = xClu[xClu>val]
        return xf
    
    
    import time
    start = time.time()
    print("Computing sorting...")
    asort,scoresF2 = get_asort_scores()
    res_unfolder2 = get_reorder(res_unfolder,val=-1)
    del asort
    del scoreF
    print("Total time sorting molecules by score:",time.time()-start)
    
    
    
    import time
    start = time.time()
    print("Finding best bits per molecules...")
    
    Rs = dec.XH[:,-1].astype(np.int64)
    Rs = torch.from_numpy(Rs)
    Rs_U = Rs[res_unfolder2]
    nregs,nbits = dec.codes_01.shape
    score_bits = torch.zeros([len(lens),nbits],dtype=scoresF2.dtype)-1
    score_bits[ires,Rs_U]=scoresF2
    
    
    codes_lib = torch.from_numpy(np.array(dec.codes__))
    
    
    codes_lib_01 = torch.zeros([len(codes_lib),nbits],dtype=score_bits.dtype)
    for icd,cd in enumerate(codes_lib):
        codes_lib_01[icd,cd]=1
    codes_lib_01 = codes_lib_01/torch.norm(codes_lib_01,dim=-1)[:,np.newaxis]
    print("Finding best code...")
    batch = 10000
    icodes_best = torch.zeros(len(score_bits),dtype=torch.int64)
    dists_best = torch.zeros(len(score_bits),dtype=torch.float32)
    from tqdm import tqdm
    for i in tqdm(range((len(score_bits)//batch)+1)):
        score_bits_ = score_bits[i*batch:(i+1)*batch]
        if len(score_bits_)>0:
            score_bits__ = score_bits_.clone()
            score_bits__[score_bits__==-1]=0
            score_bits__ = score_bits__/torch.norm(score_bits__,dim=-1)[:,np.newaxis]
            Mul = torch.matmul(score_bits__,codes_lib_01.T)
            max_ = torch.max(Mul,dim=-1)
            icodes_best[i*batch:(i+1)*batch] = max_.indices
            dists_best[i*batch:(i+1)*batch] = 2-2*max_.values
    
    
    keep_all_bits = torch.sum(score_bits.gather(1,codes_lib[icodes_best])>=0,-1)>=nmin_bits
    dists_best_ = dists_best[keep_all_bits]
    score_bits = score_bits[keep_all_bits]
    icodes_best_ = icodes_best[keep_all_bits]
    icodesN=icodes_best_
    
    indexMols_ = torch.zeros([len(lens),nbits],dtype=res_unfolder2.dtype)-1
    indexMols_[ires,Rs_U]=res_unfolder2
    indexMols_ = indexMols_[keep_all_bits]
    indexMols_ = indexMols_.gather(1,codes_lib[icodes_best_])
    
    # make unique
    indexMols_,rinvMols = get_unique_ordered(indexMols_)
    icodesN = icodesN[rinvMols]
    
    XH = torch.from_numpy(dec.XH)
    XH_pruned = XH[indexMols_]
    XH_pruned[indexMols_==-1]=np.nan
    
    dec.dist_best = dists_best_[rinvMols].numpy()
    dec.XH_pruned=XH_pruned.numpy()
    dec.icodesN=icodesN.numpy()
    np.savez_compressed(dec.decoded_fl,XH_pruned=dec.XH_pruned,icodesN=dec.icodesN,gns_names = np.array(dec.gns_names),dist_best=dec.dist_best)
    print("Total time best bits per molecule:",time.time()-start)

def compute_decoding(save_folder,fov,set_,redo=False):
    dec = decoder_simple(save_folder,fov,set_)
    #self.decoded_fl = self.decoded_fl.replace('decoded_','decodedNew_')
    complete = dec.check_is_complete()
    if complete==0 or redo:
        #compute_drift(save_folder,fov,all_flds,set_,redo=False,gpu=False)
        dec = decoder_simple(save_folder,fov=fov,set_=set_)
        dec.get_XH(fov,set_,ncols=3,nbits=16,th_h=0)#number of colors match 
        print(dec.XH.shape)
        dec.XH = dec.XH[dec.XH[:,-4]>0.25] ### keep the spots that are correlated with the expected PSF for 60X
        dec.load_library(lib_fl,nblanks=-1)
        
        dec.ncols = 3
        #dec.get_inters(dinstance_th=2,enforce_color=True)# enforce_color=False
        dec.get_inters(dinstance_th=2,nmin_bits=3,enforce_color=True,redo=True)
        #dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=None,nbits=24)#,is_unique=False)
        get_icodesV3(dec,nmin_bits=3,iH=-3)
        #get_icodesV3(dec,iH=-3,redo=False,norm_brightness=False,nbits=48,is_unique=True)

"""
def get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5):
    
    obj = get_dapi_features(fl,save_folder,set_)
    obj_ref = get_dapi_features(fl_ref,save_folder,set_)
    tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus = np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),0,0
    if (len(obj.Xh_plus)>0) and (len(obj.Xh_minus)>0) and  (len(obj_ref.Xh_plus)>0) and (len(obj_ref.Xh_minus)>0):
        X = obj.Xh_plus[:,:3]
        X_ref = obj_ref.Xh_plus[:,:3]
        tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)

        X = obj.Xh_minus[:,:3]
        X_ref = obj_ref.Xh_minus[:,:3]
        tzxy_minus,N_minus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
        
        tzxyf = -(tzxy_plus+tzxy_minus)/2
        
    

    return [tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus]
"""

def get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5):
    
    obj = get_dapi_features(fl,save_folder,set_)
    obj_ref = get_dapi_features(fl_ref,save_folder,set_)
    tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus = np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),0,0
    if (len(obj.Xh_plus)>0) and (len(obj_ref.Xh_plus)>0):
        X = obj.Xh_plus[:,:3]
        X_ref = obj_ref.Xh_plus[:,:3]
        tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if (len(obj.Xh_minus)>0) and (len(obj_ref.Xh_minus)>0):
        X = obj.Xh_minus[:,:3]
        X_ref = obj_ref.Xh_minus[:,:3]
        tzxy_minus,N_minus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if np.max(np.abs(tzxy_minus-tzxy_plus))<=2:
        tzxyf = -(tzxy_plus*N_plus+tzxy_minus*N_minus)/(N_plus+N_minus)
    else:
        tzxyf = -[tzxy_plus,tzxy_minus][np.argmax([N_plus,N_minus])]
    

    return [tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus]


def compute_drift_V2(save_folder,fov,all_flds,set_,redo=False,gpu=False):
    drift_fl = save_folder+os.sep+'driftNew_'+fov.split('.')[0]+'--'+set_+'.pkl'
    if not os.path.exists(drift_fl) or redo:
        fls = [fld+os.sep+fov for fld in all_flds]
        fl_ref = fls[len(fls)//2]
        newdrifts = []
        for fl in fls:
            drft = get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5)
            print(drft)
            newdrifts.append(drft)
        pickle.dump([newdrifts,all_flds,fov,fl_ref],open(drift_fl,'wb'))


def compute_drift_features(save_folder, fov, all_flds, set_, redo=True, gpu=True):
    fls = [fld + os.sep + fov for fld in all_flds]
    for fl in fls:
        get_dapi_features(fl, save_folder, set_, gpu=gpu, im_med_fl=flat_field_fl + r"3.npz", psf_fl=psf_file)


def main_do_compute_fits(save_folder, fld, fov, icol, save_fl, psf, old_method, icol_flat, gpu):
    im_ = read_im(fld + os.sep + fov)
    im__ = np.array(im_[icol], dtype=np.float32)

    if old_method:
        ### previous method
        im_n = norm_slice(im__, s=30)
        # Xh = get_local_max(im_n,500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,dbscan=True,
        #      return_centers=False,mins=None,sigmaZ=1,sigmaXY=1.5)
        Xh = get_local_maxfast_tensor(
            im_n, th_fit=500, im_raw=im__, dic_psf=None, delta=1, delta_fit=3, sigmaZ=1, sigmaXY=1.5, gpu=False
        )
    else:
        fl_med = flat_field_fl + str(icol) + ".npy"

        if os.path.exists(fl_med):
            im_med = np.array(np.load(fl_med)["im"], dtype=np.float32)
            im_med = cv2.blur(im_med, (20, 20))
            im__ = im__ / im_med * np.median(im_med)

        Xh = get_local_max_tile(
            im__,
            th=3600,
            s_=512,
            pad=100,
            psf=psf,
            plt_val=None,
            snorm=30,
            gpu=gpu,
            deconv={"method": "wiener", "beta": 0.0001},
            delta=1,
            delta_fit=3,
            sigmaZ=1,
            sigmaXY=1.5,
        )

    np.savez_compressed(save_fl, Xh=Xh)


####
def compute_fits(
    save_folder,
    fov,
    all_flds,
    redo=True,
    ncols=ncols,
    psf_file=psf_file,
    try_mode=True,
    old_method=False,
    redefine_color=None,
    gpu=True,
):
    psf = np.load(psf_file)
    for ifld, fld in enumerate(tqdm(all_flds, desc=f"Computing fitting on {fov} using {'GPU' if gpu else 'CPU'}")):
        if redefine_color is not None:
            ncols = len(redefine_color[ifld])
        for icol in range(ncols - 1):
            ### new method
            if redefine_color is None:
                icol_flat = icol
            else:
                # print("ifld is: "+str(ifld))
                # print("icol is: "+str(icol))
                icol_flat = redefine_color[ifld][icol]
            tag = os.path.basename(fld.strip("/"))
            save_fl = save_folder + os.sep + fov.split(".")[0] + "--" + tag + "--col" + str(icol) + "__Xhfits.npz"
            if not os.path.exists(save_fl) or redo:
                if try_mode:
                    try:
                        main_do_compute_fits(save_folder, fld, fov, icol, save_fl, psf, old_method, redefine_color, gpu)
                    except Exception as e:
                        print("Failed", fld, fov, icol, e)
                else:
                    main_do_compute_fits(save_folder, fld, fov, icol, save_fl, psf, old_method, redefine_color, gpu)



def main_f(fov, try_mode=True, gpu=True):
    print("Computing fitting on: " + str(fov))
    #compute_fits(save_folder, fov, all_flds, redo=False, try_mode=try_mode, redefine_color=redefine_color, gpu=gpu)
    #print("Computing drift on: " + str(fov))
    #try:
        #compute_drift_features(save_folder, fov, all_flds, "", redo=False, gpu=False)
    compute_drift_V2(save_folder,fov,all_flds,"",redo=False)
    compute_decoding(save_folder,fov,"",redo=False)
    #except Exception as e:
    #    print("corruption at: " + fov, e)
    
def cleanup_decoded_drift(save_folder):
    all_fls = glob.glob(save_folder+os.sep+'decoded*')
    all_fls += glob.glob(save_folder+os.sep+'drift*')
    for fl in tqdm(all_fls):
        os.remove(fl)
if __name__ == "__main__":
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    set__ = "set2"

    all_flds = glob.glob(r"/mnt/renimaging2/02_20_2024_BigHeartR133S2/H*" + set__ + os.sep)
    #all_flds += glob.glob(r"/mnt/merfish12/20231024_RD129_N5S2heart/RMERFISH/H*RMER*" + set__ + os.sep)
    redefine_color = [[0, 1, 2, 3]] * len(all_flds)  #### three signal color

    # print(all_flds)
    fovs_fl = save_folder + os.sep + "fovs__" + set__ + ".npy"
    if not os.path.exists(fovs_fl):
        fls = glob.glob(all_flds[0] + os.sep + "*.zarr")
        fovs = [os.path.basename(fl) for fl in fls]

        np.save(fovs_fl, fovs)
    else:
        fovs = np.load(fovs_fl)

    fovs=np.sort(fovs)
    #cleanup_decoded_drift(save_folder)
    #main_f(fovs[5])
    if True:

        from queue import Empty

        def worker(queue, gpu):
            try:
                while True:
                    try:
                        fov = queue.get(False)
                        main_f(fov, gpu=gpu)
                    except FileNotFoundError as e:
                        print(fov, e)
            except Empty:
                return

        import multiprocessing

        queue = multiprocessing.Queue()
        for item in sorted(fovs):
            queue.put(item)

        workers = [ ### Add however many GPU and CPU workers as you want here
            multiprocessing.Process(target=worker, kwargs={"queue": queue, "gpu": True}),
            multiprocessing.Process(target=worker, kwargs={"queue": queue, "gpu": False}),
            multiprocessing.Process(target=worker, kwargs={"queue": queue, "gpu": False}),
            multiprocessing.Process(target=worker, kwargs={"queue": queue, "gpu": False}),
            multiprocessing.Process(target=worker, kwargs={"queue": queue, "gpu": False}),
            multiprocessing.Process(target=worker, kwargs={"queue": queue, "gpu": False}),
            multiprocessing.Process(target=worker, kwargs={"queue": queue, "gpu": False}),
            multiprocessing.Process(target=worker, kwargs={"queue": queue, "gpu": False}),
            multiprocessing.Process(target=worker, kwargs={"queue": queue, "gpu": False}),
        ]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
                worker.close()
        except KeyboardInterrupt:
            for worker in workers:
                worker.terminate()
                worker.close()
