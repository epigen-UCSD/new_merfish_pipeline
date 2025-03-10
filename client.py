import xmlrpc.client
import time
import os
import multiprocessing
from ioMicro import read_im, get_local_max_tile, get_dapi_features, get_best_translation_points
import pickle
import numpy as np
import json
import logging

config = json.load(open("client.json"))


def compute_fits(image_file, icol, save_fl, psf, im_med, gpu, name):
    im_ = read_im(image_file)
    im__ = np.array(im_[icol], dtype=np.float32)
    im__ = im__ / im_med * np.median(im_med)

    if im__.shape[-1] == 2048:
        s_ = 512
    else:
        s_ = 500

    Xh = get_local_max_tile(
        im__,
        th=3600,
        s_=s_,
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

    np.savez_compressed(save_fl, Xh=Xh, author=name)


def get_best_translation_pointsV2(fl, fl_ref, save_folder, set_, resc=5):
    obj = get_dapi_features(fl, save_folder, set_)
    obj_ref = get_dapi_features(fl_ref, save_folder, set_)
    tzxyf, tzxy_plus, tzxy_minus, N_plus, N_minus = np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), 0, 0
    if (len(obj.Xh_plus) > 0) and (len(obj_ref.Xh_plus) > 0):
        X = obj.Xh_plus[:, :3]
        X_ref = obj_ref.Xh_plus[:, :3]
        tzxy_plus, N_plus = get_best_translation_points(X, X_ref, resc=resc, return_counts=True)
    if (len(obj.Xh_minus) > 0) and (len(obj_ref.Xh_minus) > 0):
        X = obj.Xh_minus[:, :3]
        X_ref = obj_ref.Xh_minus[:, :3]
        tzxy_minus, N_minus = get_best_translation_points(X, X_ref, resc=resc, return_counts=True)
    if np.max(np.abs(tzxy_minus - tzxy_plus)) <= 2:
        tzxyf = -(tzxy_plus * N_plus + tzxy_minus * N_minus) / (N_plus + N_minus)
    else:
        tzxyf = -[tzxy_plus, tzxy_minus][np.argmax([N_plus, N_minus])]

    return [tzxyf, tzxy_plus, tzxy_minus, N_plus, N_minus]


def compute_drift_V2(save_fl, fov, all_flds, set_, redo=False, gpu=False):
    save_folder = os.path.dirname(save_fl)
    fov = os.path.basename(save_fl).split("__")[-1].split("--")[0]
    #drift_fl = save_folder + os.sep + "driftNew_" + fov.split(".")[0] + "--" + set_ + ".pkl"
    if not os.path.exists(save_fl) or redo:
        fls = [fld + os.sep + fov for fld in all_flds]
        fl_ref = fls[len(fls) // 2]
        newdrifts = []
        for fl in fls:
            drft = get_best_translation_pointsV2(fl, fl_ref, save_folder, set_, resc=5)
            print(drft)
            newdrifts.append(drft)
        pickle.dump([newdrifts, all_flds, fov, fl_ref], open(drift_fl, "wb"))


def worker(name, gpu):
    server = xmlrpc.client.ServerProxy(f"http://{config['server-hostname']}:{config['server-port']}")
    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s -- {name} -- %(message)s", datefmt="%Y/%m/%d %I:%M:%S %p"
    )
    logging.info("Client started")

    psfs = {}
    meds = {}

    while True:
        try:
            img_nas, img_file, save_nas, save_folder, save_fl, psf_file, med_file, icol = server.request(name, gpu)
            img_file = os.path.join(*img_file)
            save_fl = os.path.join(config["nas-mapping"][save_nas], save_folder, save_fl)
            if os.path.exists(save_fl):
                continue
            logging.info(f"Creating {save_fl} from {img_file}")
            img_file = os.path.join(config["nas-mapping"][img_nas], img_file)
            psf_file = os.path.join(config["nas-mapping"][save_nas], save_folder, psf_file)
            med_file = os.path.join(config["nas-mapping"][save_nas], save_folder, med_file)
            if psf_file not in psfs:
                psfs[psf_file] = np.load(psf_file)
            if med_file not in meds:
                meds[med_file] = np.load(med_file)
            if "Xhfits" in save_fl:
                compute_fits(img_file, icol, save_fl, psfs[psf_file], meds[med_file], gpu, name)
            elif "dapiFeatures" in save_fl:
                get_dapi_features(
                    img_file,
                    os.path.dirname(save_fl),
                    "",
                    gpu=False,
                    im_med_fl=meds[med_file],
                    psf_fl=psf_file,
                    name=name,
                )
            elif "drift" in save_fl:
                fov = os.path.basename(save_fl).split("__")[-1].split("--")[0]
                compute_drift_V2(save_fl, fov)
            elif "decoded" in save_fl:
                pass
            server.complete(name)
        except xmlrpc.client.Fault:
            time.sleep(1)
        except ConnectionRefusedError:
            time.sleep(1)
        except TimeoutError:
            time.sleep(1)
        except Exception:
            server.abort(name)


if __name__ == "__main__":
    workers = []
    for i in range(1, config["gpu-workers"] + 1):
        kwargs = {"name": f"{config['client-name']}-GPU.{i}", "gpu": True}
        workers.append(multiprocessing.Process(target=worker, kwargs=kwargs))
    for i in range(1, config["cpu-workers"] + 1):
        kwargs = {"name": f"{config['client-name']}-CPU.{i}", "gpu": False}
        workers.append(multiprocessing.Process(target=worker, kwargs=kwargs))

    for p in workers:
        p.start()

    for p in workers:
        p.join()
        p.close()
