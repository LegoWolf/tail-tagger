import argparse
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import __main__

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

MODEL_PATH="classifiers/JTP-3/jtp-3-hydra.safetensors"
LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
INTERNAL_THRESHOLD = 0.01

DEFAULT_CONFIG = {
    "delay_seconds": 1,
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

    def update(self) -> bool:
        while not self.input_queue.empty():
            if not self.dequeue():
                return False
        if len(self.delay_queue) == 0:
            if not self.dequeue():
                return False
        return True

    def peek(self):
        (item, timestamp) = self.delay_queue.peekitem()
        return item, timestamp, self.delay_info[item]

    def pop(self):
        (item, timestamp) = self.delay_queue.popitem()
        return item, timestamp, self.delay_info.pop(item)

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
    _, stdout, stderr = run_command(['exiv2', '-px', 'pr', image_path])
    for error in stderr.splitlines():
        logging.warning('%s (while checking tags on: %s)', error, image_path)
    has_tag = tag in stdout
    logging.debug('%s JTP-3 tag: %s', "Has" if has_tag else "Missing", image_path)
    return has_tag

def write_xmp_tags(image_path, tags):
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

def image_processor(image_queue):
    delay_queue = DelayQueue(image_queue)
    classifier = Classifier(model_path=MODEL_PATH)

    while delay_queue.update():
        image_path, timestamp, event = delay_queue.peek()
        time_delay = time.time() - timestamp

        if time_delay < config["delay_seconds"]:
            time.sleep(config["delay_seconds"] - time_delay)
        else:
            image_path, timestamp, event = delay_queue.pop()
            logging.debug("Checking %s...", image_path)

            try:
                start_job = time.time()
                if not check_has_xmp_tag(image_path, config["classified_tag"]):
                    tags, time_preprocess, time_inference = \
                        classifier.classify_image(image_path, config["score_cutoff"])
                    tags.append(config["classified_tag"])
                    write_xmp_tags(image_path, tags)
                    time_job = time.time() - start_job
                    logging.info("%8s %.2fs %.2fs (%.2fs %.2fs) %3d %s",
                        event, time_delay, time_job, time_preprocess, time_inference,
                        len(tags), image_path)

            except subprocess.CalledProcessError as e:
                logging.error("Called process '%s' failed: %s (return code: %d)",
                    ' '.join(e.cmd), e.stderr.strip(), e.returncode)

            except Exception as e:
                logging.error("Image processing failed: %s (%s)", e, image_path)
                logging.debug("Stack trace:\n%s", traceback.format_exc().strip())

def main():
    global config
    config_filepath = os.path.splitext(__main__.__file__)[0] + '.toml'
    log_filepath = os.path.splitext(__main__.__file__)[0] + '.log'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default='.',
        help=f"path to the {os.path.basename(config_filepath)} file.")
    parser.add_argument(
        "--loglevel",
        choices=["debug", "info", "warning", "error"],
        help="minimum level of messages to log")
    args = parser.parse_args()

    with open(os.path.join(args.config, config_filepath), "rb") as f:
        config = DEFAULT_CONFIG | tomllib.load(f)

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                log_filepath,
                mode='a',
                maxBytes=config["logging"]["file_size"],
                backupCount=config["logging"]["max_files"]),
        ],
        level=(args.loglevel if args.loglevel else config["logging"]["level"]).upper(),
        format=LOG_FORMAT)

    for folder_path in config["include_folders"]:
        if not os.path.isdir(folder_path):
            logging.error("Include folder does not exist: %s", folder_path)
            return

    try:
        image_queue = queue.Queue()
        worker_thread = threading.Thread(target=image_processor, args=(image_queue,))
        worker_thread.start()

        observer = Observer()
        for folder in config["include_folders"]:
            event_handler = MonitorEventHandler(
                image_queue,
                patterns=config["include_patterns"],
                ignore_patterns=config["exclude_patterns"],
                ignore_directories=True,
                case_sensitive=False)
            observer.schedule(event_handler, folder, recursive=True)
        observer.start()

        try:
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            logging.info("Process aborted at keyboard!")

        except Exception as e:
            logging.error(e)

        finally:
            observer.stop()
            observer.join()
            image_queue.put(None)
            worker_thread.join()

    except Exception as e:
        logging.error(e)

if __name__ == '__main__':
    main()
