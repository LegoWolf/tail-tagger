import time
import threading
import queue
import torch 
import heapdict
import os
import subprocess

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from inference import (
    load_jtp3_model, preprocess_jtp3, run_inference_jtp3
)

DELAY = 1
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
SCORE_CUTOFF = 0.30
CLASSIFIED_TAG = 'e621-jtp3'

def is_image_file(filepath) -> bool:
    return os.path.splitext(filepath)[1].lower() in IMAGE_EXTENSIONS

class MyEventHandler(FileSystemEventHandler):
    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and is_image_file(event.src_path):
            # print(f'Image created: queuing {event.src_path}')
            image_queue.put((event.src_path, time.time()))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and is_image_file(event.src_path):
            # print(f'Image modified: queueing {event.src_path}')
            image_queue.put((event.src_path, time.time()))

class Classifier:
    def __init__(self, model_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("CUDA (GPU) is available and selected.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU.")

        self.model, self.allowed_tags = load_jtp3_model(
            model_path=model_path,
            device=self.device
        )
        print("Loaded JTP-3 inference module.")

    def classify_image(self, image_path, score_cutoff):
        print(f"Loading and preprocessing image {image_path}...")
        start_preprocess = time.time()
        patches, coords, valid = preprocess_jtp3(image_path)
        end_preprocess = time.time()
        print(f"Preprocessing took {end_preprocess - start_preprocess:.3f} seconds.")

        # --- Run Inference using provided function ---
        print("Running JTP-3 inference...")
        probabilities = run_inference_jtp3(
            model=self.model,
            patches=patches,
            coords=coords,
            valid=valid,
            device=self.device
        )

        print("Post-processing results...")
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
                print(f"Warning: Index {tag_index} out of bounds for allowed tags.")

        # 3. Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"Found {len(results)} tags above INTERNAL threshold {INTERNAL_THRESHOLD} and with a score above {score_cutoff}.")
        return [result[0] for result in results]

def check_has_xmp_tag(image_path, tag):
    try:
        return tag in subprocess.check_output(['exiv2', '-px', 'pr', image_path]).decode('utf-8')
    except CalledProcessError as e:
        print(f'Error: failed to find tag "{tag}" in {image_path}')
        return True

def write_xmp_tags(image_path, tags):
    try:
        if len(tags) > 0:
            keywords = [tag.replace('_', ' ') for tag in tags]
            keywords_stream = '\n'.join([f'set Xmp.dc.subject {kw}' for kw in keywords])
            p = subprocess.run(['exiv2', '-m-', image_path], input=keywords_stream, encoding='utf-8')
            if p.returncode == 0:
                print(f"Wrote XMP keywords to: {image_path}")
            else:
                print(f"Error writing XMP keywords to {image_path}: return code {p.returncode}")
    except Exception as e:
        print(f"Error writing to {image_path}: {e}")

def image_processor(image_queue):
    delay_queue = heapdict.heapdict()
    classifier = Classifier(model_path="classifiers/JTP-3/jtp-3-hydra.safetensors")

    while True:
        while not image_queue.empty():
            (image_path, timestamp) = image_queue.get()
            delay_queue[image_path] = timestamp
        if len(delay_queue) == 0:
            (image_path, timestamp) = image_queue.get()
            delay_queue[image_path] = timestamp

        (image_path, timestamp) = delay_queue.peekitem()
        now = time.time()

        if now - timestamp > DELAY:
            delay_queue.popitem()
            print(f'Checking {image_path}...')
            try:
                if not check_has_xmp_tag(image_path, CLASSIFIED_TAG):
                    tags = classifier.classify_image(image_path, SCORE_CUTOFF)
                    tags.append(CLASSIFIED_TAG)
                    write_xmp_tags(image_path, tags)

            except Exception as e:
                print(f"ERROR during JTP-3 analysis: {e}")
                import traceback
                traceback.print_exc()
        else:
            time.sleep(now - timestamp + 0.5)

image_queue = queue.Queue()
worker_thread = threading.Thread(target=image_processor, args=(image_queue,))
worker_thread.start()

event_handler = MyEventHandler()
observer = Observer()
observer.schedule(event_handler, "D:\\Downloads\\yiffy", recursive=True)
observer.start()
try:
    while True:
        time.sleep(1)
finally:
    observer.stop()
    observer.join()
    image_queue.put(None)
    worker_thread.join()