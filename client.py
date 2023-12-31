import xmlrpc.client
import time
import os
import multiprocessing
from ioMicro import read_im, get_local_max_tile, get_dapi_features
import numpy as np
import cv2
import logging

client_name = "pumpkin"
gpu_workers = 1
cpu_workers = 2

hostname = "breadfruit.ucsd.edu"
port = 8000

naspath = {
    "merfish9": "/mnt/merfish9",
    "merfish10": "/mnt/merfish10",
    "merfish11": "/mnt/merfish11",
    "merfish12": "/mnt/merfish12",
    "merfish13": "/mnt/merfish13",
    "/home/plt3": "/home/plt3"
}


def compute_fits(image_file, icol, save_fl, psf, im_med, gpu):
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

    np.savez_compressed(save_fl, Xh=Xh)


def worker(name, gpu):
    server = xmlrpc.client.ServerProxy(f"http://{hostname}:{port}")
    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s -- {name} -- %(message)s", datefmt="%Y/%m/%d %I:%M:%S %p"
    )
    logging.info("Client started")

    psfs = {}
    meds = {}

    while True:
        try:
            img_nas, img_file, save_nas, save_fl, psf_file, med_file, icol = server.request(name, gpu)
            img_file = os.path.join(naspath[img_nas], img_file)
            save_fl = os.path.join(naspath[save_nas], save_fl)
            if not os.path.exists(os.path.dirname(save_fl)):
                os.mkdir(os.path.dirname(save_fl))
            logging.info(f"Creating {save_fl} from {img_file}")
            if psf_file not in psfs:
                if os.path.exists(psf_file):
                    psfs[psf_file] = np.load(psf_file)
                else:
                    logging.info(f"Getting PSF {psf_file}")
                    psfs[psf_file] = np.array(server.get_psf(psf_file))
                    psfs[psf_file].save(psf_file)
            if (med_file, icol) not in meds:
                if os.path.exists(f"{med_file}{icol}.npz"):
                    meds[med_file, icol] = np.array(np.load(f"{med_file}{icol}.npz")["im"], dtype=np.float32)
                    meds[med_file, icol] = cv2.blur(meds[med_file, icol], (20, 20))
                else:
                    logging.info(f"Getting median image {med_file} for color {icol}")
                    meds[med_file, icol] = np.array(server.get_flat_field(med_file, icol))
            if "Xhfits" in save_fl:
                compute_fits(img_file, icol, save_fl, psfs[psf_file], meds[med_file, icol], gpu)
            elif "dapiFeatures" in save_fl:
                get_dapi_features(
                    img_file,
                    os.path.dirname(save_fl),
                    "",
                    gpu=False,
                    im_med_fl=meds[med_file, icol],
                    psf_fl=psfs[psf_file],
                )
        except xmlrpc.client.Fault:
            time.sleep(1)
        except ConnectionRefusedError:
            time.sleep(1)


workers = []
for i in range(1, gpu_workers + 1):
    kwargs = {"name": f"{client_name}-GPU.{i}", "gpu": True}
    workers.append(multiprocessing.Process(target=worker, kwargs=kwargs))
for i in range(1, cpu_workers + 1):
    kwargs = {"name": f"{client_name}-CPU.{i}", "gpu": False}
    workers.append(multiprocessing.Process(target=worker, kwargs=kwargs))

for p in workers:
    p.start()

for p in workers:
    p.join()
    p.close()
