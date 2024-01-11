from xmlrpc.server import SimpleXMLRPCServer
import glob
import os
import re
import time
import datetime
import numpy as np
import multiprocessing
import cv2
import curses

data_nas = "merfish11"
data_folders = r"20230919_R128_N5S1MERNeuro/RMERFISH/H*"
skip = []
save_nas = "merfish11"
save_folder = "20230919_R128_N5S1MERNeuro_analysis3"
psf_file = r"psfs/psf_D103_B.npy"
flat_field_fl = r"flat_field/R128__med_col_raw"
# If order="hyb", fitting will be done for all FOVs in H1, then all FOVs in H2, etc.
# If order="fov", fitting will be done for all hybs of FOV 0, then all hybs of FOV 1, etc.
order = "hyb"
ncol = 4

hostname = "breadfruit.ucsd.edu"
port = 8000

naspath = {
    "merfish9": "/mnt/merfish9",
    "merfish10": "/mnt/merfish10",
    "merfish11": "/mnt/merfish11",
    "merfish12": "/mnt/merfish12",
    "merfish13": "/mnt/merfish13",
}


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def parse_fov(task):
    return task[1].split("__")[-1].split(".")[0]


class TaskServer:
    def __init__(self, messages):
        self.messages = messages

    def scan_folders(self):
        all_flds = glob.glob(os.path.join(naspath[data_nas], data_folders) + os.sep)
        all_flds.sort(key=natural_keys)
        self.tasks = []
        self.dapi_features = []
        for fld in all_flds:
            self.messages.put(["Scanning", fld])
            if fld in skip:
                continue
            for filename in sorted(glob.glob(fld + os.sep + "*.zarr")):
                filename_no_nas = filename.split(naspath[data_nas])[-1].strip(os.sep)
                fov = os.path.basename(filename).split(".")[0]
                hyb = filename.split("/")[-2]
                prefix = os.path.join(save_folder, f"{fov}--{hyb}")
                for icol in range(ncol - 1):
                    save_fl = f"{prefix}--col{icol}__Xhfits.npz"
                    if not os.path.exists(os.path.join(naspath[save_nas], save_fl)):
                        self.tasks.append((data_nas, filename_no_nas, save_nas, save_fl, psf_file, flat_field_fl, icol))
                save_fl = f"{prefix}--dapiFeatures.npz"
                if not os.path.exists(os.path.join(naspath[save_nas], save_fl)):
                    self.tasks.append((data_nas, filename_no_nas, save_nas, save_fl, psf_file, flat_field_fl, ncol - 1))
                self.messages.put(["Tasks", len(self.tasks)])
        if order == "fov":
            self.tasks.sort(key=parse_fov)
        self.messages.put(["Scanning", ""])

    def request(self, client, nodapi=False):
        i = 0
        if nodapi:
            while "dapiFeatures" in self.tasks[i][3]:
                i += 1
                if i == len(self.tasks):
                    i = 0
                    break
        data = self.tasks.pop(i)
        self.messages.put(["Assigning", data, client])
        return data

    def get_psf(self, psf_file):
        return np.load(psf_file).tolist()

    def get_flat_field(self, flat_field_fl, icol):
        fl_med = f"{flat_field_fl}{icol}.npz"
        im_med = np.array(np.load(fl_med)["im"], dtype=np.float32)
        return cv2.blur(im_med, (20, 20)).tolist()

    def start(self):
        self.scan_folders()
        with SimpleXMLRPCServer((hostname, port), logRequests=False) as server:
            server.register_function(self.request)
            server.register_function(self.get_psf)
            server.register_function(self.get_flat_field)
            server.serve_forever()


def server_process(messages):
    server = TaskServer(messages)
    server.start()


def interface(stdscr, messages):
    stdscr.clear()
    curses.curs_set(0)
    stdscr.nodelay(True)

    scanning = ""
    worker_status = {}
    worker_started = {}
    durations = []
    done = None
    tasks = 0
    while True:
        while not messages.empty():
            message = messages.get()
            if message[0] == "Scanning":
                scanning = message[1]
            elif message[0] == "Assigning":
                if message[2] in worker_started:
                    durations.append(time.time() - worker_started[message[2]])
                    avg = sum(durations[:100]) / len(durations[:100])
                    secs_left = (avg * tasks) / len(worker_status)
                    td = datetime.timedelta(seconds=int(secs_left))
                    done = datetime.datetime.now() + td
                    tasks -= 1
                worker_status[message[2]] = message[1][3]
                worker_started[message[2]] = time.time()
            elif message[0] == "Tasks":
                tasks = message[1]
        stdscr.erase()
        stdscr.addstr(0, 0, f"Fitting images on {data_nas} in {data_folders}")
        stdscr.addstr(1, 0, f"Saving fits to {save_nas} in {save_folder}")
        stdscr.addstr(2, 0, f"Deconvolving with {psf_file}")
        stdscr.addstr(3, 0, f"Flat field correcting with {flat_field_fl}")
        stdscr.addstr(4, 0, "-----------")
        if scanning:
            stdscr.addstr(5, 0, f"{tasks} files left to generate, scanning {scanning}")
        else:
            stdscr.addstr(5, 0, f"{tasks} files left to generate")
        if done:
            stdscr.addstr(6, 0, f"Estimated completion in {td} at {done}")
        else:
            stdscr.addstr(6, 0, "Estimated completion: TBD")
        stdscr.addstr(7, 0, "-----------")
        for i, (name, status) in enumerate(worker_status.items(), start=8):
            duration = datetime.timedelta(seconds=int(time.time() - worker_started[name]))
            stdscr.addstr(i, 0, f"{name}: {status} ({duration})")
        stdscr.addstr(10 + len(worker_status), 0, "Press Q to quit")
        stdscr.refresh()
        key = stdscr.getch()
        if key >= 0:
            key = chr(key)
            if key in {"q", "Q"}:
                break
        time.sleep(0.5)


messages = multiprocessing.Queue()
p = multiprocessing.Process(target=server_process, kwargs={"messages": messages})
p.start()
curses.wrapper(interface, messages)
p.terminate()
p.join()
p.close()
