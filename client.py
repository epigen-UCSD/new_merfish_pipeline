import xmlrpc.client
import time
import os
import multiprocessing
from ioMicro import read_im, get_local_max_tile, get_dapi_features
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
            img_nas, img_file, save_nas, save_fl, psf_file, med_file, icol = server.request(name, gpu)
            save_fl = os.path.join(config["nas-mapping"][save_nas], save_fl)
            if os.path.exists(save_fl):
                continue
            logging.info(f"Creating {save_fl} from {img_file}")
            img_file = os.path.join(config["nas-mapping"][img_nas], img_file)
            psf_file = os.path.join(config["nas-mapping"][save_nas], psf_file)
            med_file = os.path.join(config["nas-mapping"][save_nas], med_file)
            if psf_file not in psfs:
                psfs[psf_file] = np.load(psf_file)
            if med_file not in meds:
                meds[med_file] = np.load(med_file)
            if "Xhfits" in save_fl:
                compute_fits(img_file, icol, save_fl, psfs[psf_file], meds[med_file], gpu, name)
            elif "dapiFeatures" in save_fl:
                get_dapi_features(
                    img_file, os.path.dirname(save_fl), "", gpu=False, im_med_fl=meds[med_file], psf_fl=psf_file, name=name
                )
            server.complete(name)
        except xmlrpc.client.Fault:
            time.sleep(1)
        except ConnectionRefusedError:
            time.sleep(1)


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
