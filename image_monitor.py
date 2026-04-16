import time
import threading
import queue
import torch 
import heapdict
import os
import subprocess
import logging
import traceback
import argparse
import sys

from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers import Observer

from inference import (
    load_jtp3_model, preprocess_jtp3, run_inference_jtp3
)

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10

class MonitorEventHandler(PatternMatchingEventHandler):
    def __init__(self, image_queue, **kwargs):
        super().__init__(**kwargs)
        self.image_queue = image_queue

    def on_created(self, event: FileSystemEvent) -> None:
        logging.debug(f'Event: image created: queuing {event.src_path}')
        self.image_queue.put((event.src_path, time.time(), "created"))

    def on_modified(self, event: FileSystemEvent) -> None:
        logging.debug(f'Event: image modified: queueing {event.src_path}')
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

    def classify_image(self, image_path, score_cutoff):
        logging.debug(f"Loading and preprocessing image {image_path}...")
        start_preprocess = time.time()
        patches, coords, valid = preprocess_jtp3(image_path)
        end_preprocess = time.time()
        logging.debug(f"Preprocessing took {end_preprocess - start_preprocess:.3f} seconds.")

        # --- Run Inference using provided function ---
        logging.debug("Running JTP-3 inference...")
        start_inference = time.time()
        probabilities = run_inference_jtp3(
            model=self.model,
            patches=patches,
            coords=coords,
            valid=valid,
            device=self.device
        )
        end_inference = time.time()

        logging.debug("Post-processing results...")
        # 1. Thresholding (find indices above threshold)
        probabilities_cpu = probabilities.cpu()  # Move to CPU for thresholding/indexing
        INTERNAL_THRESHOLD = 0.01  # Filter out only extremely unlikely tags
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
                logging.warning(f"Warning: Index {tag_index} out of bounds for allowed tags.")

        # 3. Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        logging.debug(f"Found {len(results)} tags above INTERNAL threshold {INTERNAL_THRESHOLD} and with a score above {score_cutoff}.")
        return [result[0] for result in results], end_preprocess - start_preprocess, end_inference - start_inference

class DelayQueue:
    def __init__(self, queue):
        self.queue = queue
        self.delay_queue = heapdict.heapdict()
        self.delay_info = {}

    def dequeue(self):
        entry = self.queue.get()
        if entry is None:
            return False
        (item, timestamp, info) = entry
        self.delay_queue[item] = timestamp
        self.delay_info[item] = info
        return True

    def update(self) -> bool:
        while not self.queue.empty():
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

def run_command(args, input=None) -> bool:
    process = subprocess.run(
        args,
        input=input,
        encoding='utf-8',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True)
    return process.returncode, process.stdout.strip(), process.stderr.strip()

def check_has_xmp_tag(image_path, tag):
    return_code, stdout, stderr = run_command(['exiv2', '-px', 'pr', image_path])
    for error in stderr.splitlines():
        logging.warning(f'{error} (while checking tags on: {image_path})')
    has_tag = tag in stdout
    logging.debug(f'{"Has" if has_tag else "Missing"} JTP-3 tag: {image_path}')
    return has_tag

def write_xmp_tags(image_path, tags):
    if len(tags) > 0:
        # Preserve the current modified time.
        modified_time = os.path.getmtime(image_path)
        keywords = [tag.replace('_', ' ') for tag in tags]
        keywords_buffer = '\n'.join([f'set Xmp.dc.subject {kw}' for kw in keywords])
        return_code, stdout, stderr = run_command(['exiv2', '-m-', image_path], input=keywords_buffer)
        for error in stderr.splitlines():
            logging.warning(f'{error} (while setting tags on: {image_path})')
        logging.debug(f"Wrote XMP keywords to: {image_path}")
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
            logging.debug(f'Checking {image_path}...')

            try:
                if not check_has_xmp_tag(image_path, config["classified_tag"]):
                    tags, time_preprocess, time_inference = classifier.classify_image(image_path, config["score_cutoff"])
                    tags.append(config["classified_tag"])
                    write_xmp_tags(image_path, tags)
                    logging.info(f'{event:8} {time_delay:2.2f}s {time_preprocess:2.2f}s {time_inference:2.2f}s {len(tags):3} {image_path}')

            except subprocess.CalledProcessError as e:
                logging.error(f"Called process '{' '.join(e.cmd)}' failed: {e.stderr.strip()} (return code: {e.returncode})")

            except Exception as e:
                logging.error(f"Image processing failed: {e} ({image_path})")
                logging.debug(f"Stack trace:\n{traceback.format_exc().strip()}")

def main():
    import __main__
    config_filename = os.path.splitext(__main__.__file__)[0] + '.toml'
    log_filename = os.path.splitext(__main__.__file__)[0] + '.log'
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='.', help=f"path to the {config_filename} file.")
    parser.add_argument("--loglevel", default="info", choices=["debug", "info", "warning", "error"], help="minimum level of messages to log")
    args = parser.parse_args()

    with open(os.path.join(args.config, config_filename), "rb") as f:
        config = tomllib.load(f)

    logging.basicConfig(filename=log_filename, format=LOG_FORMAT, level=args.loglevel.upper())
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

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

        except KeyboardInterrupt as e:
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

MODEL_PATH="classifiers/JTP-3/jtp-3-hydra.safetensors"
LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'

config = {
    "delay_seconds": 1,
    "score_cutoff": 0.30,
    "classified_tag": 'e621-jtp3',
    "include_folders": [ "D:/Downloads/yiffy" ],
    "include_patterns": ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'],
    "exclude_patterns": []
}

if __name__ == '__main__':
    main()
