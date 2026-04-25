import ctypes
import datetime
import logging
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time
import traceback
import __main__
 
import concurrent_log_handler
import heapdict
import torch

from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers import Observer

from inference import (
    load_jtp3_model, preprocess_jtp3, run_inference_jtp3
)

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10

MODEL_PATH = "classifiers/JTP-3/jtp-3-hydra.safetensors"
LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
INTERNAL_THRESHOLD = 0.01

DEFAULT_CONFIG = {
    "delay_seconds": 1,
    "ignore_seconds": 1,
    "score_cutoff": 0.30,
    "everything_walk": False,
    "everything_retry_seconds": 60,
    "classified_tag_model": 'e621-model-{0}',
    "classified_tag_score": 'e621-score-{0}',
    "include_folders": [ "D:/Downloads/yiffy" ],
    "include_patterns": ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'],
    "exclude_patterns": [],
    "logging": {
        "file_size": 1024 * 1024,
        "max_files": 10,
        "level": "info",
    }
}

config = {}

class MonitorEventHandler(PatternMatchingEventHandler):
    def __init__(self, image_queue, **kwargs):
        super().__init__(**kwargs)
        self.image_queue = image_queue

    def on_created(self, event: FileSystemEvent) -> None:
        logging.debug('Event: image created: queuing %s', event.src_path)
        self.image_queue.put(FileEntry(event.src_path, time.time(), FileEntry.EVENT_CREATED))

    def on_modified(self, event: FileSystemEvent) -> None:
        logging.debug('Event: image modified: queueing %s', event.src_path)
        self.image_queue.put(FileEntry(event.src_path, time.time(), FileEntry.EVENT_MODIFIED))

class Classifier:
    def __init__(self, model_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info("CUDA (GPU) is available and selected.")
        else:
            self.device = torch.device("cpu")
            logging.info("Using CPU.")

        self.model, self.allowed_tags = load_jtp3_model(
            model_path=model_path,
            device=self.device
        )
        logging.debug("Loaded JTP-3 inference module.")

    def preprocess(self, image_path):
        start_preprocess = time.time()
        patches, coords, valid = preprocess_jtp3(image_path)
        time_preprocess = time.time() - start_preprocess
        return time_preprocess, patches, coords, valid

    def infer(self, patches, coords, valid):
        start_inference = time.time()
        probabilities = run_inference_jtp3(
            model=self.model,
            patches=patches,
            coords=coords,
            valid=valid,
            device=self.device
        )
        time_inference = time.time() - start_inference
        return time_inference, probabilities

    def postprocess(self, probabilities, score_cutoff):
        # 1. Thresholding (find indices above threshold)
        # Move to CPU for thresholding/indexing
        probabilities_cpu = probabilities.cpu()
        # Filter out only extremely unlikely tags
        indices = torch.where(probabilities_cpu > INTERNAL_THRESHOLD)[0]
        values = probabilities_cpu[indices]

        # 2. Map indices to tags and store scores
        results = []
        for i in range(indices.size(0)):
            tag_index = indices[i].item()
            if 0 <= tag_index < len(self.allowed_tags):
                tag_name = self.allowed_tags[tag_index]
                score = values[i].item()
                if score >= score_cutoff:
                    results.append((tag_name, score))
            else:
                logging.warning("Warning: Index %s out of bounds for allowed tags.", tag_index)

        # 3. Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def classify_image(self, image_path, score_cutoff):
        logging.debug("Loading and preprocessing image %s...", image_path)
        time_preprocess, patches, coords, valid = self.preprocess(image_path)
        logging.debug("Running JTP-3 inference...")
        time_inference, probabilities = self.infer(patches, coords, valid)
        logging.debug("Post-processing results...")
        results = self.postprocess(probabilities, score_cutoff)
        logging.debug("Found %d tags above INTERNAL threshold %.2f and with a score above %.2f.",
            len(results), INTERNAL_THRESHOLD, score_cutoff)
        return [result[0] for result in results], time_preprocess, time_inference

class Everything:
    EVERYTHING3_PROPERTY_ID_PATH = 1 
    EVERYTHING3_PROPERTY_ID_DATE_MODIFIED = 5
    EVERYTHING3_PROPERTY_ID_PATH_AND_NAME = 240
    EVERYTHING3_ERROR_IPC_PIPE_NOT_FOUND = 0xE0000002
    PATH_BUFFER_SIZE = 260
    WINDOWS_TICKS = int(1/10**-7)
    WINDOWS_EPOCH = datetime.datetime.strptime('1601-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    POSIX_EPOCH = datetime.datetime.strptime('1970-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    EPOCH_DIFF = (POSIX_EPOCH - WINDOWS_EPOCH).total_seconds()
    WINDOWS_TICKS_TO_POSIX_EPOCH = EPOCH_DIFF * WINDOWS_TICKS

    def __init__(self, executable_path, instance_name=None):
        dll_filepath = os.path.join(executable_path, "everything_sdk\\dll\\Everything3_x64.dll")
        logging.info("Everything DLL path: %s", dll_filepath)
        self.instance_name = instance_name
        self.everything_dll = ctypes.WinDLL(dll_filepath)
        self.everything_dll.Everything3_ConnectW.argtypes = [ctypes.c_wchar_p]
        self.everything_dll.Everything3_ConnectW.restype = ctypes.c_void_p
        self.everything_dll.Everything3_CreateSearchState.restype = ctypes.c_void_p
        self.everything_dll.Everything3_SetSearchTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self.everything_dll.Everything3_AddSearchPropertyRequest.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.everything_dll.Everything3_AddSearchPropertyRequest.restype = ctypes.c_bool
        self.everything_dll.Everything3_Search.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.everything_dll.Everything3_Search.restype = ctypes.c_void_p
        self.everything_dll.Everything3_GetResultListFileCount.argtypes = [ctypes.c_void_p]
        self.everything_dll.Everything3_GetResultListFileCount.restype = ctypes.c_size_t
        self.everything_dll.Everything3_GetResultFullPathNameW.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_wchar_p, ctypes.c_size_t]
        self.everything_dll.Everything3_GetResultFullPathNameW.restype = ctypes.c_size_t
        self.everything_dll.Everything3_GetResultDateModified.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.everything_dll.Everything3_GetResultDateModified.restype = ctypes.c_uint64
        self.everything_dll.Everything3_DestroyResultList.argtypes = [ctypes.c_void_p]
        self.everything_dll.Everything3_DestroySearchState.argtypes = [ctypes.c_void_p]
        self.everything_dll.Everything3_DestroyClient.argtypes = [ctypes.c_void_p]
        self.everything_dll.Everything3_GetLastError.restype = ctypes.c_uint

    def get_timestamp(self, filetime):
        """Convert windows filetime winticks to python datetime.datetime."""
        return (filetime - self.WINDOWS_TICKS_TO_POSIX_EPOCH) / self.WINDOWS_TICKS

    def query(self, query):
        client = None
        state = None
        result_list = None
        results = []

        try:
            client = self.everything_dll.Everything3_ConnectW(self.instance_name)
            state = self.everything_dll.Everything3_CreateSearchState()
            self.everything_dll.Everything3_SetSearchTextW(state, query)
            self.everything_dll.Everything3_AddSearchPropertyRequest(state, self.EVERYTHING3_PROPERTY_ID_PATH_AND_NAME)
            self.everything_dll.Everything3_AddSearchPropertyRequest(state, self.EVERYTHING3_PROPERTY_ID_PATH)
            self.everything_dll.Everything3_AddSearchPropertyRequest(state, self.EVERYTHING3_PROPERTY_ID_DATE_MODIFIED)

            result_list = self.everything_dll.Everything3_Search(client, state)
            if result_list is None:
                error = self.everything_dll.Everything3_GetLastError()
                logging.debug("Error communicating with Everything: 0x%x", error)
                return None

            num_results = self.everything_dll.Everything3_GetResultListFileCount(result_list)
            filename = ctypes.create_unicode_buffer(self.PATH_BUFFER_SIZE)

            for i in range(num_results):
                self.everything_dll.Everything3_GetResultFullPathNameW(result_list, i, filename, self.PATH_BUFFER_SIZE)
                date_modified_filetime = self.everything_dll.Everything3_GetResultDateModified(result_list, i)
                results.append((ctypes.wstring_at(filename), self.get_timestamp(date_modified_filetime)))

        except Exception as e:
            logging.error("Failed to connect to Everything DLL: %s", e)

        finally:
            if result_list is not None:
                self.everything_dll.Everything3_DestroyResultList(result_list)
            if state is not None:
                self.everything_dll.Everything3_DestroySearchState(state)
            if client is not None:
                self.everything_dll.Everything3_DestroyClient(client)

        return results

class FileEntry(object):
    EVENT_CREATED = "created"
    EVENT_MODIFIED = "modified"
    EVENT_WALKED = "walked"
    EVENT_QUIT = "quit"

    def __init__(self, filepath, timestamp, event):
        self.filepath = str(filepath)
        self.timestamp = timestamp
        self.event = event

    def __repr__(self):
        return f'FileEntry: {self.filepath} {self.timestamp} {self.event}'

    def __lt__(self, other):
        if self.event == other.event:
            return self.timestamp < other.timestamp
        else:
            return self.event != self.EVENT_WALKED

class FolderWalkEntries:
    def __init__(self, entries_queue, classified_tag, executable_path):
        self.start_walk = time.time()
        self.entries_queue = entries_queue
        self.classified_tag = classified_tag
        self.executable_path = executable_path
        if sys.platform == "win32" and config["everything_walk"]:
            self.everything_walk()
        else:
            self.os_walk()

    def everything_walk_worker(self, entries_queue):
        everything = Everything(self.executable_path, config.get("everything_instance"))
        entries = {}
        for folder in config["include_folders"]:
            query = f"!tags:{self.classified_tag} files: \"{folder}\""
            while True:
                results = everything.query(query)
                if results is None:
                    time.sleep(config["everything_retry_seconds"])
                else:
                    break
            logging.debug(f"Everything query: '%s' (found %d files)", query, len(results))
            for result in results:
                if self.pattern_allowed(pathlib.Path(result[0])):
                    entries[result[0]] = result[1]
        logging.info("Walked folders with Everything, found %d files.", len(entries))
        for filepath, timestamp in entries.items():
            entries_queue.put(FileEntry(filepath, timestamp, FileEntry.EVENT_WALKED))

    def everything_walk(self):
        self.walk_thread = threading.Thread(target=self.everything_walk_worker, args=(self.entries_queue,), daemon=True)
        self.walk_thread.start()

    def os_walk(self):
        entries = []
        for folder in config["include_folders"]:
            for root, dirs, files in os.walk(folder):
                logging.debug("Complete walk: %s (%d files)", root, len(files))
                for file in files:
                    filepath = pathlib.Path(os.path.join(root, file))
                    if self.pattern_allowed(filepath):
                        entries.append(FileEntry(filepath, os.path.getmtime(filepath), FileEntry.EVENT_WALKED))
        logging.info("Walked folders with OS, found %d files.", len(entries))
        for entry in entries:
            self.entries_queue.put(entry)

    def pattern_allowed(self, filepath):
        include_match = any((filepath.match(pattern) for pattern in config["include_patterns"]))
        exclude_match = any((filepath.match(pattern) for pattern in config["exclude_patterns"]))
        return include_match and not exclude_match

class TemporarySet:
    def __init__(self):
        self.heap = heapdict.heapdict()

    def add(self, item, until):
        self.heap[item] = until

    def check(self, item, now):
        until = self.heap.get(item)
        return until is not None and now < until 

    def drain(self, before):
        while len(self.heap) > 0:
            _, until = self.heap.peekitem()
            if until < before:
                self.heap.popitem()
            else:
                break

class Application:
    def __init__(self):
        self.executable_path = os.path.split(__main__.__file__)[0]
        self.config_filepath = os.path.join(self.executable_path, os.path.splitext(__name__)[0] + '.toml')
        self.log_filepath = os.path.join(self.executable_path, os.path.splitext(__name__)[0] + '.log')
        self.log_level = None

    def get_config_filename(self):
        return os.path.basename(self.config_filepath)

    def set_config_path(self, config_path):
        self.config_filepath = os.path.join(config_path, os.path.basename(self.config_filepath))

    def set_log_level(self, log_level):
        self.log_level = log_level

    def run_command(self, args, input_buffer=None) -> bool:
        process = subprocess.run(
            args,
            input=input_buffer,
            encoding='utf-8',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True)
        return process.returncode, process.stdout.strip(), process.stderr.strip()

    def check_has_xmp_tag(self, image_path, tag):
        start_check = time.time()
        _, stdout, stderr = self.run_command(['exiv2', '-px', 'pr', image_path])
        for error in stderr.splitlines():
            logging.warning('%s (while checking tags on: %s)', error, image_path)
        has_tag = tag in stdout
        time_check = time.time() - start_check
        logging.debug('%s JTP-3 tag: %s (%.2fs)', "Has" if has_tag else "Missing", image_path, time_check)
        return has_tag, time_check

    def write_xmp_tags(self, image_path, tags):
        start_write = time.time()
        if len(tags) > 0:
            # Preserve the current modified time.
            modified_time = os.path.getmtime(image_path)
            keywords = [tag.replace('_', ' ') for tag in tags]
            keywords_buffer = '\n'.join([f'set Xmp.dc.subject {kw}' for kw in keywords])
            _, _, stderr = self.run_command(
                ['exiv2', '-m-', image_path],
                input_buffer=keywords_buffer)
            for error in stderr.splitlines():
                logging.warning("%s (while setting tags on: %s)", error, image_path)
            logging.debug("Wrote XMP keywords to: %s", image_path)
            access_time = os.path.getatime(image_path)
            os.utime(image_path, times=(access_time, modified_time))
        return time.time() - start_write

    def image_processor(self, image_queue):
        classifier = Classifier(model_path=MODEL_PATH)
        ignore_set = TemporarySet()
        found_walk = False

        while True:
            entry = image_queue.get()
            if entry.event == FileEntry.EVENT_QUIT:
                break

            start_job = time.time()
            time_delay = start_job - entry.timestamp
            ignore_set.drain(entry.timestamp)

            if not found_walk:
                if entry.event == FileEntry.EVENT_WALKED:
                    found_walk = True
                    start_walk = time.time()
            elif image_queue.empty():
                time_walk = time.time() - start_walk
                logging.info("Finished processing existing files from a recursive walk. (%dm %ds)",
                    time_walk / 60, int(time_walk) % 60)
                found_walk = False

            if not ignore_set.check(entry.filepath, entry.timestamp):
                if time_delay < config["delay_seconds"]:
                    image_queue.put(entry)
                    time.sleep(config["delay_seconds"] - time_delay)
                else:
                    logging.debug("Checking %s...", entry.filepath)

                    try:
                        has_tag, time_check = self.check_has_xmp_tag(entry.filepath, self.tag_model)
                        if not has_tag:
                            tags, time_preprocess, time_inference = \
                                classifier.classify_image(entry.filepath, config["score_cutoff"])
                            tags.extend([self.tag_model, self.tag_score])
                            time_write = self.write_xmp_tags(entry.filepath, tags)
                            time_job = time.time() - start_job
                            logging.info("%8s %.2fs %.2fs (%.2fs %.2fs %.2fs %.2fs) %3d %s",
                                entry.event, time_delay, time_job, time_check, time_preprocess,
                                time_inference, time_write, len(tags), entry.filepath)
                        ignore_set.add(entry.filepath, time.time() + config["ignore_seconds"])

                    except subprocess.CalledProcessError as e:
                        logging.error("Called process '%s' failed: %s (return code: %d)",
                            ' '.join(e.cmd), e.stderr.strip(), e.returncode)

                    except Exception as e:
                        logging.error("Image processing failed: %s (%s)", e, entry.filepath)
                        logging.debug("Stack trace:\n%s", traceback.format_exc().strip())

    def start(self, log_level=None):
        global config
        with open(self.config_filepath, "rb") as f:
            config = DEFAULT_CONFIG | tomllib.load(f)

        logging.basicConfig(
            handlers=[
                logging.StreamHandler(sys.stdout),
                concurrent_log_handler.ConcurrentRotatingFileHandler(
                    self.log_filepath,
                    mode='a',
                    maxBytes=config["logging"]["file_size"],
                    backupCount=config["logging"]["max_files"],
                    use_gzip=True),
            ],
            level=(self.log_level if self.log_level else config["logging"]["level"]).upper(),
            format=LOG_FORMAT)

        for folder_path in config["include_folders"]:
            if not os.path.isdir(folder_path):
                logging.error("Include folder does not exist: %s", folder_path)
                return False

        self.tag_model = config["classified_tag_model"].format("jtp3")
        self.tag_score = config["classified_tag_score"].format(f"{config['score_cutoff']:.4f}")

        os.chdir(self.executable_path) 
        logging.debug("Working directory: %s", os.getcwd())

        try:
            self.image_queue = queue.PriorityQueue()
            self.walker = FolderWalkEntries(self.image_queue, self.tag_model, self.executable_path)
            self.worker_thread = threading.Thread(target=self.image_processor, args=(self.image_queue,))
            self.worker_thread.start()

            self.observer = Observer()
            for folder in config["include_folders"]:
                event_handler = MonitorEventHandler(
                    self.image_queue,
                    patterns=config["include_patterns"],
                    ignore_patterns=config["exclude_patterns"],
                    ignore_directories=True,
                    case_sensitive=False)
                self.observer.schedule(event_handler, folder, recursive=True)
            self.observer.start()

        except Exception as e:
            logging.error(e)

        return True

    def stop(self):
        self.observer.stop()
        self.observer.join()
        self.image_queue.put(FileEntry("", 0.0, event=FileEntry.EVENT_QUIT))
        self.worker_thread.join()
