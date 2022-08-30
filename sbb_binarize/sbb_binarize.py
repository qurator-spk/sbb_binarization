import argparse
import gc
import itertools
import math
import os
from pathlib import Path
from typing import Union, List, Any

import cv2
import numpy as np
import tensorflow as tf
from mpire import WorkerPool
from mpire.utils import make_single_arguments
from tensorflow.python.keras.saving.save import load_model


class SbbBinarizer:
    def __init__(self) -> None:
        super().__init__()
        self.model: Any = None
        self.model_height: int = 0
        self.model_width: int = 0
        self.n_classes: int = 0

    def load_model(self, model_dir: Union[str, Path]):
        model_dir = Path(model_dir)
        self.model = load_model(str(model_dir.absolute()), compile=False)
        self.model_height = self.model.layers[len(self.model.layers) - 1].output_shape[1]
        self.model_width = self.model.layers[len(self.model.layers) - 1].output_shape[2]
        self.n_classes = self.model.layers[len(self.model.layers) - 1].output_shape[3]

    def binarize_image(self, image_path: Path, save_path: Path):
        if not image_path.exists():
            raise ValueError(f"Image not found: {str(image_path)}")

        # noinspection PyUnresolvedReferences
        img = cv2.imread(str(image_path))
        original_image_height, original_image_width, image_channels = img.shape

        # Padded images must be multiples of model size
        padded_image_height = math.ceil(original_image_height / self.model_height) * self.model_height
        padded_image_width = math.ceil(original_image_width / self.model_width) * self.model_width
        padded_image = np.zeros((padded_image_height, padded_image_width, image_channels))
        padded_image[0:original_image_height, 0:original_image_width, :] = img[:, :, :]

        image_batch = np.expand_dims(padded_image, 0)  # Create the batch dimension
        patches = tf.image.extract_patches(
            images=image_batch,
            sizes=[1, self.model_height, self.model_width, 1],
            strides=[1, self.model_height, self.model_width, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )

        number_of_horizontal_patches = patches.shape[1]
        number_of_vertical_patches = patches.shape[2]
        total_number_of_patches = number_of_horizontal_patches * number_of_vertical_patches
        target_shape = (total_number_of_patches, self.model_height, self.model_width, image_channels)
        # Squeeze all image patches (n, m, width, height, channels) into a single big batch (b, width, height, channels)
        image_patches = tf.reshape(patches, target_shape)
        # Normalize the image to values between 0.0 - 1.0
        image_patches = image_patches / float(255.0)

        predicted_patches = self.model.predict(image_patches)
        # We have to manually call garbage collection and clear_session here to avoid memory leaks.
        # Taken from https://medium.com/dive-into-ml-ai/dealing-with-memory-leak-issue-in-keras-model-training-e703907a6501
        gc.collect()
        tf.keras.backend.clear_session()

        binary_patches = np.invert(np.argmax(predicted_patches, axis=3).astype(bool)).astype(np.uint8) * 255
        full_image_with_padding = self._patches_to_image(
            binary_patches,
            padded_image_height,
            padded_image_width,
            self.model_height,
            self.model_width
        )
        full_image = full_image_with_padding[0:original_image_height, 0:original_image_width]
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # noinspection PyUnresolvedReferences
        cv2.imwrite(str(save_path), full_image)

    def _patches_to_image(self, patches: np.ndarray, image_height: int, image_width: int, patch_height: int, patch_width: int):
        height = math.ceil(image_height / patch_height) * patch_height
        width = math.ceil(image_width / patch_width) * patch_width

        image_reshaped = np.reshape(
            np.squeeze(patches),
            [height // patch_height, width // patch_width, patch_height, patch_width]
        )
        image_transposed = np.transpose(a=image_reshaped, axes=[0, 2, 1, 3])
        image_resized = np.reshape(image_transposed, [height, width])
        return image_resized


def split_list_into_worker_batches(files: List[Any], number_of_workers: int) -> List[List[Any]]:
    """ Splits any given list into batches for the specified number of workers and returns a list of lists. """
    batches = []
    batch_size = math.ceil(len(files) / number_of_workers)
    batch_start = 0
    for i in range(1, number_of_workers + 1):
        batch_end = i * batch_size
        file_batch_to_delete = files[batch_start: batch_end]
        batches.append(file_batch_to_delete)
        batch_start = batch_end
    return batches


def batch_predict(input_data):
    model_dir, input_images, output_images, worker_number = input_data
    print(f"Setting visible cuda devices to {str(worker_number)}")
    # Each worker thread will be assigned only one of the available GPUs to allow multiprocessing across GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_number)

    binarizer = SbbBinarizer()
    binarizer.load_model(model_dir)

    for image_path, output_path in zip(input_images, output_images):
        binarizer.binarize_image(image_path=image_path, save_path=output_path)
        print(f"Binarized {image_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', default="model_2021_03_09", help="Path to the directory where the TF model resides or path to an h5 file.")
    parser.add_argument('-i', '--input-path', required=True)
    parser.add_argument('-o', '--output-path', required=True)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    model_directory = args.model_dir

    if input_path.is_dir():
        print(f"Enumerating all PNG files in {str(input_path)}")
        all_input_images = list(input_path.rglob("*.png"))
        print(f"Filtering images that have already been binarized in {str(output_path)}")
        input_images = [i for i in all_input_images if not (output_path / (i.relative_to(input_path))).exists()]
        output_images = [output_path / (i.relative_to(input_path)) for i in input_images]
        input_images = [i for i in input_images]

        print(f"Starting batch-binarization of {len(input_images)} images")

        number_of_gpus = len(tf.config.list_physical_devices('GPU'))
        number_of_workers = max(1, number_of_gpus)
        image_batches = split_list_into_worker_batches(input_images, number_of_workers)
        output_batches = split_list_into_worker_batches(output_images, number_of_workers)

        # Must use spawn to create completely new process that has its own resources to properly multiprocess across GPUs
        with WorkerPool(n_jobs=number_of_workers, start_method='spawn') as pool:
            model_dirs = itertools.repeat(model_directory, len(image_batches))
            input_data = zip(model_dirs, image_batches, output_batches, range(number_of_workers))
            contents = pool.map_unordered(
                batch_predict,
                make_single_arguments(input_data),
                iterable_len=number_of_workers,
                progress_bar=False
            )
    else:
        binarizer = SbbBinarizer()
        binarizer.load_model(model_directory)
        binarizer.binarize_image(image_path=input_path, save_path=output_path)
