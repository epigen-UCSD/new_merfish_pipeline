import curses
import glob
import time
import datetime
import os
import re
import numpy as np
import multiprocessing
import logging
import cv2
from functools import lru_cache

from ioMicro import read_im, get_local_max_tile, get_dapi_features


data_folders = r"/mnt/merfish10/20231107_D106LuoRMER/RNA/H*"
skip = ["/mnt/merfish10/20231107_D106LuoRMER/RNA/H0/"]
save_folder = r"/mnt/merfish10/20231107_D106LuoRMER_analysis"
psf_file = r"psf_647_Kiwi.npy"
flat_field_fl = r"D106_RMER_repeat__med_col_raw"
set__ = ""

gpu_workers = 2
cpu_workers = 3

# standard is 4, its number of colors +1
ncols = 4

logging.basicConfig(filename="worker.log", level=logging.INFO)


def compute_fits(image_file, icol, save_fl, psf, gpu):
    im_ = read_im(image_file)
    im__ = np.array(im_[icol], dtype=np.float32)

    fl_med = flat_field_fl + str(icol) + ".npz"

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


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


@lru_cache(maxsize=10)
def check_image(filename):
    try:
        read_im(filename)
        return True
    except Exception:
        return False


def add_new_images_to_queue(queue, messages):
    all_flds = glob.glob(data_folders + set__ + os.sep)
    all_flds.sort(key=natural_keys)
    for fld in all_flds:
        if fld in skip:
            continue
        messages.put(f"Scanner--Scanning {os.path.split(fld.strip(os.sep))[-1]}")
        files = sorted(glob.glob(fld + os.sep + "*.zarr"))
        for i, filename in enumerate(files):
            messages.put(f"Scanner--Scanning {os.path.split(fld.strip(os.sep))[-1]} ({len(files) - i} images left)")
            fov = os.path.basename(filename).split(".")[0]
            hyb = filename.split("/")[-2]
            prefix = os.path.join(save_folder, f"{fov}--{hyb}")
            for icol in range(ncols - 1):
                save_fl = f"{prefix}--col{icol}__Xhfits.npz"
                if not os.path.exists(save_fl) and check_image(filename):
                    queue.put([filename, save_fl])
            save_fl = f"{prefix}--dapiFeatures.npz"
            if not os.path.exists(save_fl) and check_image(filename):
                queue.put([filename, save_fl])


def manager(queue, messages):
    while True:
        add_new_images_to_queue(queue, messages)
        if queue.empty():  # No new images found, wait an hour
            messages.put("Scanner--No new images found on last scan, waiting 1 hour")
            time.sleep(60 * 60)
        # Wait until the queue is empty
        messages.put("Scanner--Waiting for tasks to complete before scanning for new images")
        while not queue.empty():
            time.sleep(60 * 3)


def worker(queue, messages, gpu, name):
    psf = np.load(psf_file)
    while True:
        messages.put(f"Worker {name} Waiting")
        image_fl, save_fl = queue.get()
        start_time = time.time()
        if save_fl.endswith("Xhfits.npz"):
            messages.put(f"Worker {name} {os.path.split(save_fl)[-1]}")
            icol = int(re.search(r"--col(\d+)__", save_fl)[1])
            try:
                compute_fits(image_fl, icol, save_fl, psf, gpu)
            except Exception as e:
                messages.put([e, image_fl, save_fl, name])
        elif save_fl.endswith("dapiFeatures.npz"):
            if gpu:
                queue.put([image_fl, save_fl])
                time.sleep(0.1)
                continue
            messages.put(f"Worker {name} {os.path.split(save_fl)[-1]}")
            try:
                get_dapi_features(
                    image_fl,
                    save_folder,
                    "",
                    gpu=False,
                    im_med_fl=f"{flat_field_fl}3.npz",
                    psf_fl=psf_file,
                )
            except Exception as e:
                messages.put([e, image_fl, save_fl, name])
        messages.put(f"Elapsed {time.time() - start_time}")


def main(stdscr):
    # Setup
    stdscr.clear()
    curses.curs_set(0)
    stdscr.nodelay(True)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    tasks = multiprocessing.Queue()
    messages = multiprocessing.Queue()
    manager_process = multiprocessing.Process(target=manager, kwargs={"queue": tasks, "messages": messages})
    manager_process.start()

    gpu_kwargs = {"queue": tasks, "messages": messages, "gpu": True}
    cpu_kwargs = {"queue": tasks, "messages": messages, "gpu": False}
    workers = []
    for i in range(1, gpu_workers + 1):
        gpu_kwargs["name"] = f"GPU-{i}"
        workers.append(multiprocessing.Process(target=worker, kwargs=gpu_kwargs))
    for i in range(1, cpu_workers + 1):
        cpu_kwargs["name"] = f"CPU-{i}"
        workers.append(multiprocessing.Process(target=worker, kwargs=cpu_kwargs))
    for p in workers:
        p.start()

    scanning = ""
    worker_status = {}
    worker_timestamp = {}
    durations = []
    done = None
    errors = 0
    while True:
        while not messages.empty():
            message = messages.get()
            if isinstance(message, list):
                error, image_fl, save_fl, name = message
                logging.info(f"{name}, {save_fl}, {error}")
                errors += 1
            elif message.startswith("Scanner"):
                scanning = message.split("--")[1]
            elif message.startswith("Worker"):
                message = message[4:]
                _, name, status = message.split()
                worker_status[name] = status
                worker_timestamp[name] = time.time()
            elif message.startswith("Elapsed"):
                duration = float(message.split()[1])
                durations.append(duration)
                if len(durations) >= 5:
                    avg = sum(durations[:100]) / len(durations[:100])
                    secs_left = (avg * tasks.qsize()) / len(workers)
                    td = datetime.timedelta(seconds=int(secs_left))
                    done = datetime.datetime.now() + td
        stdscr.erase()
        stdscr.addstr(0, 0, f"Fitting images in {data_folders}")
        stdscr.addstr(1, 0, f"Saving fits to {save_folder}")
        stdscr.addstr(2, 0, f"Deconvolving with {psf_file}")
        stdscr.addstr(3, 0, f"Flat field correcting with {flat_field_fl}")
        stdscr.addstr(4, 0, "-----------")
        stdscr.addstr(5, 0, f"{tasks.qsize():,d} files left to generate")
        if scanning.startswith("Scanning"):
            stdscr.addstr(6, 0, "Estimated completion: TBD (waiting for scanning to complete)")
        elif done:
            stdscr.addstr(6, 0, f"Estimated completion in {td} at {done}")
        else:
            stdscr.addstr(6, 0, "Estimated completion: TBD (wait for more tasks to complete)")
        if scanning:
            stdscr.addstr(7, 0, scanning)
        stdscr.addstr(8, 0, "-----------")
        for i, (name, status) in enumerate(worker_status.items(), start=9):
            duration = datetime.timedelta(seconds=int(time.time() - worker_timestamp[name]))
            stdscr.addstr(i, 0, f"{name}: {status} ({duration})")
        stdscr.addstr(10 + len(workers), 0, f"Press Q to quit, {errors} errors")
        stdscr.refresh()
        key = stdscr.getch()
        if key >= 0:
            key = chr(key)
            if key in {"q", "Q"}:
                break
        time.sleep(0.5)

    manager_process.terminate()
    manager_process.join()
    manager_process.close()
    for p in workers:
        p.terminate()
        p.join()
        p.close()


if __name__ == "__main__":
    curses.wrapper(main)
