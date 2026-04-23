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
    "classified_tag": 'e621-jtp3',
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
        self.image_queue.put((event.src_path, time.time(), "created"))

    def on_modified(self, event: FileSystemEvent) -> None:
        logging.debug('Event: image modified: queueing %s', event.src_path)
        self.image_queue.put((event.src_path, time.time(), "modified"))

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

class DelayQueue:
    def __init__(self, input_queue):
        self.input_queue = input_queue
        self.delay_queue = heapdict.heapdict()
        self.delay_info = {}

    def dequeue(self):
        entry = self.input_queue.get()
        if entry is None:
            return False
        (item, timestamp, info) = entry
        self.delay_queue[item] = timestamp
        self.delay_info[item] = info
        return True

    def update(self, latent_entries=None) -> bool:
        while not self.input_queue.empty():
            if not self.dequeue():
                return False
        if len(self.delay_queue) == 0:
            if latent_entries is not None and len(latent_entries) > 0:
                (item, timestamp, info) = latent_entries.pop()
                self.delay_queue[item] = timestamp
                self.delay_info[item] = info
                return True
            if not self.dequeue():
                return False
        return True

    def peek(self):
        (item, timestamp) = self.delay_queue.peekitem()
        return item, timestamp, self.delay_info[item]

    def pop(self):
        (item, timestamp) = self.delay_queue.popitem()
        return item, timestamp, self.delay_info.pop(item)

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

def run_command(args, input_buffer=None) -> bool:
    process = subprocess.run(
        args,
        input=input_buffer,
        encoding='utf-8',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True)
    return process.returncode, process.stdout.strip(), process.stderr.strip()

def check_has_xmp_tag(image_path, tag):
    start_check = time.time()
    _, stdout, stderr = run_command(['exiv2', '-px', 'pr', image_path])
    for error in stderr.splitlines():
        logging.warning('%s (while checking tags on: %s)', error, image_path)
    has_tag = tag in stdout
    time_check = time.time() - start_check
    logging.debug('%s JTP-3 tag: %s (%.2fs)', "Has" if has_tag else "Missing", image_path, time_check)
    return has_tag, time_check

def write_xmp_tags(image_path, tags):
    start_write = time.time()
    if len(tags) > 0:
        # Preserve the current modified time.
        modified_time = os.path.getmtime(image_path)
        keywords = [tag.replace('_', ' ') for tag in tags]
        keywords_buffer = '\n'.join([f'set Xmp.dc.subject {kw}' for kw in keywords])
        _, _, stderr = run_command(
            ['exiv2', '-m-', image_path],
            input_buffer=keywords_buffer)
        for error in stderr.splitlines():
            logging.warning("%s (while setting tags on: %s)", error, image_path)
        logging.debug("Wrote XMP keywords to: %s", image_path)
        access_time = os.path.getatime(image_path)
        os.utime(image_path, times=(access_time, modified_time))
    return time.time() - start_write

def get_walk_entries():
    walk_entries = []
    for include_folder in config["include_folders"]:
        for root, dirs, files in os.walk(include_folder):
            logging.debug("Complete walk: %s (%d files)", root, len(files))
            for file in files:
                filepath = pathlib.Path(os.path.join(root, file))
                include_match = any([filepath.match(pattern) for pattern in config["include_patterns"]])
                exclude_match = any([filepath.match(pattern) for pattern in config["exclude_patterns"]])
                if include_match and not exclude_match: 
                    walk_entries.append((filepath, os.path.getmtime(filepath), "walked"))
    logging.debug("Filtered walk: %s files", len(walk_entries))
    walk_entries.sort(key=lambda entry: entry[1])
    return walk_entries

def image_processor(image_queue):
    delay_queue = DelayQueue(image_queue)
    classifier = Classifier(model_path=MODEL_PATH) 
    ignore_set = TemporarySet()
    start_walk = time.time()
    walk_entries = get_walk_entries()
    walking = True

    while delay_queue.update(walk_entries):
        if walking and len(walk_entries) == 0:
            logging.info("Finished processing existing files from a recursive walk. (%.2fs)",
                time.time() - start_walk)
            walking = False

        image_path, timestamp, event = delay_queue.peek()
        start_job = time.time()
        time_delay = start_job - timestamp
        ignore_set.drain(timestamp)

        if ignore_set.check(str(image_path), timestamp):
            delay_queue.pop()
        elif time_delay < config["delay_seconds"]:
            time.sleep(config["delay_seconds"] - time_delay)
        else:
            image_path, timestamp, event = delay_queue.pop()
            logging.debug("Checking %s...", image_path)

            try:
                has_tag, time_check = check_has_xmp_tag(image_path, config["classified_tag"])
                if not has_tag:
                    tags, time_preprocess, time_inference = \
                        classifier.classify_image(image_path, config["score_cutoff"])
                    tags.append(config["classified_tag"])
                    time_write = write_xmp_tags(image_path, tags)
                    time_job = time.time() - start_job
                    logging.info("%8s %.2fs %.2fs (%.2fs %.2fs %.2fs %.2fs) %3d %s",
                        event, time_delay, time_job, time_check, time_preprocess,
                        time_inference, time_write, len(tags), image_path)
                ignore_set.add(str(image_path), time.time() + config["ignore_seconds"])

            except subprocess.CalledProcessError as e:
                logging.error("Called process '%s' failed: %s (return code: %d)",
                    ' '.join(e.cmd), e.stderr.strip(), e.returncode)

            except Exception as e:
                logging.error("Image processing failed: %s (%s)", e, image_path)
                logging.debug("Stack trace:\n%s", traceback.format_exc().strip())

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

        # TODO: Get rid of this hard-coded folder.
        os.chdir(self.executable_path) 
        logging.debug("Working directory: %s", os.getcwd())

        try:
            self.image_queue = queue.Queue()
            self.worker_thread = threading.Thread(target=image_processor, args=(self.image_queue,))
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
        self.image_queue.put(None)
        self.worker_thread.join()
