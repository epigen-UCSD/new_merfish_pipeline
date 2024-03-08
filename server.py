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
import json
import sys

config = json.load(open(sys.argv[1]))
data_nas = config["nas-mapping"][config["data_nas"]]
save_nas = config["nas-mapping"][config["save_nas"]]


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
        all_folders = glob.glob(os.path.join(data_nas, config["data_folders"]) + os.sep)
        all_folders.sort(key=natural_keys)
        self.tasks = []
        total = 0
        for folder in all_folders:
            self.messages.put(["Scanning", folder])
            if folder in config["skip_hybs"]:
                continue
            for filename in sorted(glob.glob(folder + os.sep + "*.zarr")):
                total += self.scan_file(filename)
                self.messages.put(["Tasks", len(self.tasks), total])
        if config["order"] == "fov":
            self.tasks.sort(key=parse_fov)
        self.messages.put(["Scanning", ""])

    def create_task(self, image_file, save_fl, icol):
        self.tasks.append(
            (
                config["data_nas"],
                os.path.split(image_file),
                config["save_nas"],
                config["save_folder"],
                save_fl,
                os.path.basename(config["psf_file"]),
                os.path.basename(config["flat_field_fl"]) + f"{icol}.npy",
                icol,
            )
        )

    def scan_file(self, filename):
        total = 0
        image_file = filename.split(data_nas)[-1].strip(os.sep)
        fov = os.path.basename(filename).split(".")[0]
        hyb = filename.split("/")[-2]
        prefix = os.path.join(config["save_folder"], f"{fov}--{hyb}")
        for icol in range(config["ncol"] - 1):
            save_fl = f"{prefix}--col{icol}__Xhfits.npz"
            total += 1
            if not os.path.exists(os.path.join(save_nas, save_fl)):
                self.create_task(image_file, save_fl, icol)
        save_fl = f"{prefix}--dapiFeatures.npz"
        total += 1
        if not os.path.exists(os.path.join(save_nas, save_fl)):
            self.create_task(image_file, save_fl, config["ncol"] - 1)
        return total

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

    def complete(self, client):
        self.messages.put(["Completed", client])

    def start(self):
        self.scan_folders()
        save_folder = os.path.join(save_nas, config["save_folder"])
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
            os.chmod(save_folder, 0o777)
        psf = np.load(config["psf_file"])
        np.save(os.path.join(save_folder, os.path.basename(config["psf_file"])), psf)
        for icol in range(config["ncol"]):
            fl_med = f"{config['flat_field_fl']}{icol}.npz"
            im_med = np.array(np.load(fl_med)["im"], dtype=np.float32)
            im_med = cv2.blur(im_med, (20, 20))
            np.save(os.path.join(save_folder, os.path.basename(config["flat_field_fl"]) + f"{icol}.npy"), im_med)
        with SimpleXMLRPCServer((config["server-hostname"], config["server-port"]), logRequests=False) as server:
            server.register_function(self.request)
            server.register_function(self.complete)
            server.serve_forever()


def server_process(messages):
    server = TaskServer(messages)
    server.start()


class AnalysisStatus:
    def __init__(self):
        self.scanning = ""
        self.worker_status = {}
        self.worker_started = {}
        self.durations = []
        self.done = None
        self.tasks = 0
        self.total = 1

    def parse_message(self, message):
        if message[0] == "Scanning":
            self.scanning = message[1]
        elif message[0] == "Assigning":
            self.worker_status[message[2]] = message[1][3]
            self.worker_started[message[2]] = time.time()
        elif message[0] == "Completed":
            if message[1] in self.worker_started:
                self.durations.append(time.time() - self.worker_started[message[1]])
                self.update_time_remaining()
                del self.worker_started[message[1]]
            if message[1] in self.worker_status:
                del self.worker_status[message[1]]
        elif message[0] == "Tasks":
            self.tasks = message[1]
            self.total = message[2]

    def update_time_remaining(self):
        avg = sum(self.durations[-100:]) / len(self.durations[-100:])
        secs_left = (avg * self.tasks) / len(self.worker_status)
        self.td = datetime.timedelta(seconds=int(secs_left))
        self.done = datetime.datetime.now() + self.td
        self.tasks -= 1

    def progress(self):
        if self.scanning:
            return f"{self.tasks} files left to generate, scanning {self.scanning}"
        pct = 100 * (self.total - self.tasks) / self.total
        return f"{self.tasks} files left to generate, {pct:0.1f}% complete"

    def completion(self):
        if self.done:
            return f"Estimated completion in {self.td} at {self.done}"
        return "Estimated completion: TBD"

    def workers(self):
        if not self.worker_status:
            yield "Waiting for task requests"
        for name, status in self.worker_status.items():
            duration = datetime.timedelta(seconds=int(time.time() - self.worker_started[name]))
            yield f"{name}: {status} ({duration})"


def main(stdscr, messages):
    stdscr.clear()
    curses.curs_set(0)
    stdscr.nodelay(True)
    status = AnalysisStatus()
    clear_counter = 0
    while True:
        while not messages.empty():
            status.parse_message(messages.get())
        if clear_counter > 10:
            stdscr.clear()
            clear_counter = 0
        clear_counter += 1
        stdscr.erase()
        stdscr.addstr(0, 0, f"Fitting images on {config['data_nas']} in {config['data_folders']}")
        stdscr.addstr(1, 0, f"Saving fits to {config['save_nas']} in {config['save_folder']}")
        stdscr.addstr(2, 0, f"Deconvolving with {config['psf_file']}")
        stdscr.addstr(3, 0, f"Flat field correcting with {config['flat_field_fl']}")
        stdscr.addstr(4, 0, "-----------")
        stdscr.addstr(5, 0, status.progress())
        stdscr.addstr(6, 0, status.completion())
        stdscr.addstr(7, 0, "-----------")
        for i, line in enumerate(status.workers(), start=8):
            stdscr.addstr(i, 0, line)
        stdscr.addstr(i + 2, 0, "Press Q to quit")
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
curses.wrapper(main, messages)
p.terminate()
p.join()
p.close()
