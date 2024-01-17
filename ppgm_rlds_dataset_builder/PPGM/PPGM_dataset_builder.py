from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os
import cv2
import h5py
import json
import io
from collections import defaultdict
import random
from copy import deepcopy
from PIL import Image

CAMERA_TYPE_DICT = {
    'wrist_camera_id': 0,
    'static_camera_id': 1,
}

CAMERA_TYPE_TO_STRING_DICT = {
    0: "wrist_camera",
    1: "static_camera",
}


def get_camera_type(cam_id):
    if cam_id not in CAMERA_TYPE_DICT:
        return None
    type_int = CAMERA_TYPE_DICT[cam_id]
    type_str = CAMERA_TYPE_TO_STRING_DICT[type_int]
    return type_str


class MP4Reader:
    def __init__(self, filepath, serial_number, grayscale=False):
        # Save Parameters #
        self.serial_number = serial_number
        self.grayscale = grayscale
        self._index = 0

        # Open Video Reader #
        self._mp4_reader = cv2.VideoCapture(filepath)
        if not self._mp4_reader.isOpened():
            raise RuntimeError("Corrupted MP4 File")


    def set_reading_parameters(
        self,
        image=True,
        resolution=(0, 0),
    ):
        # Save Parameters #
        self.image = image
        self.resolution = resolution
        self.skip_reading = not image
        if self.skip_reading:
            return

    def get_frame_resolution(self):
        width = self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        return (width, height)

    def get_frame_count(self):
        if self.skip_reading:
            return 0
        frame_count = int(self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        return frame_count

    def set_frame_index(self, index):
        if self.skip_reading:
            return

        if index < self._index:
            self._mp4_reader.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
            self._index = index

        while self._index < index:
            self.read_camera(ignore_data=True)

    def _process_frame(self, frame):
        frame = deepcopy(frame)
        return frame

    def read_camera(self, ignore_data=False, correct_timestamp=None):
        # Skip if Read Unnecesary #
        if self.skip_reading:
            return {}

        # Read Camera #
        success, frame = self._mp4_reader.read()
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to 1-channel image
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert to RGB

        self._index += 1
        if not success:
            return None
        if ignore_data:
            return None

        # Return Data #
        data_dict = {}
        data_dict["image"] = {self.serial_number: self._process_frame(frame)}

        return data_dict

    def disable_camera(self):
        if hasattr(self, "_mp4_reader"):
            self._mp4_reader.release()


class RecordedMultiCameraWrapper:
    def __init__(self, recording_folderpath, camera_kwargs={}):
        # Save Camera Info #
        self.camera_kwargs = camera_kwargs

        # Open Camera Readers #
        mp4_filepaths = glob.glob(recording_folderpath + "/*.mp4")
        svo_filepaths = []
        all_filepaths = svo_filepaths + mp4_filepaths

        self.camera_dict = {}
        for f in all_filepaths:
            serial_number = f.split("/")[-1][:-4]
            cam_type = get_camera_type(serial_number)
            camera_kwargs.get(cam_type, {})

            if f.endswith("_depth.mp4"):
                self.camera_dict[serial_number] = MP4Reader(f, serial_number, grayscale=True) # depth is black and white
            elif f.endswith(".mp4"):
                self.camera_dict[serial_number] = MP4Reader(f, serial_number)
            else:
                raise ValueError


    def read_cameras(self, index=None, camera_type_dict={}, timestamp_dict={}):
        full_obs_dict = defaultdict(dict)

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        #random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            cam_type = camera_type_dict[cam_id]
            curr_cam_kwargs = self.camera_kwargs.get(cam_type, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

            timestamp = timestamp_dict.get(cam_id + "_frame_received", None)
            if index is not None:
                self.camera_dict[cam_id].set_frame_index(index)

            data_dict = self.camera_dict[cam_id].read_camera(correct_timestamp=timestamp)

            # Process Returned Data #
            if data_dict is None:
                return None
            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])

        return full_obs_dict



def get_hdf5_length(hdf5_file, keys_to_ignore=[]):
    length = None

    for key in hdf5_file.keys():
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            curr_length = get_hdf5_length(curr_data, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            curr_length = len(curr_data)
        else:
            raise ValueError

        if length is None:
            length = curr_length
        assert curr_length == length

    return length


def load_hdf5_to_dict(hdf5_file, index, keys_to_ignore=[]):
    data_dict = {}

    for key in hdf5_file.keys():
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            data_dict[key] = load_hdf5_to_dict(curr_data, index, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            data_dict[key] = curr_data[index]
        else:
            raise ValueError

    return data_dict



class TrajectoryReader:
    def __init__(self, filepath, read_images=True):
        self._hdf5_file = h5py.File(filepath, "r")
        is_video_folder = "observations/videos" in self._hdf5_file
        self._read_images = read_images and is_video_folder
        self._length = get_hdf5_length(self._hdf5_file)
        self._video_readers = {}
        self._index = 0

    def length(self):
        return self._length

    def read_timestep(self, index=None, keys_to_ignore=[]):
        # Make Sure We Read Within Range #
        if index is None:
            index = self._index
        else:
            assert not self._read_images
            self._index = index
        assert index < self._length

        # Load Low Dimensional Data #
        keys_to_ignore = [*keys_to_ignore.copy(), "videos"]
        timestep = load_hdf5_to_dict(self._hdf5_file, self._index, keys_to_ignore=keys_to_ignore)

        # Increment Read Index #
        self._index += 1

        # Return Timestep #
        return timestep

    def get_target_label(self):
        return self._hdf5_file.attrs['current_task']

    def close(self):
        self._hdf5_file.close()


def crawler(dirname, filter_func=None):
    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    traj_files = [f.path for f in os.scandir(dirname) if (f.is_file() and "trajectory.h5" in f.path)]

    if len(traj_files):
        # Only Save Desired Data #
        if filter_func is None:
            use_data = True
        else:
            hdf5_file = h5py.File(traj_files[0], "r")
            use_data = filter_func(hdf5_file.attrs)
            hdf5_file.close()

        if use_data:
            return [dirname]

    all_folderpaths = []
    for child_dirname in subfolders:
        child_paths = crawler(child_dirname, filter_func=filter_func)
        all_folderpaths.extend(child_paths)

    return all_folderpaths


def load_trajectory(
    filepath,
    wrist_cam_id,
    static_cam_id,
    read_cameras=True,
    recording_folderpath=None,
    camera_kwargs={},
    remove_skipped_steps=False,
    num_samples_per_traj=None,
    num_samples_per_traj_coeff=1.5,
):
    read_hdf5_images = read_cameras and (recording_folderpath is None) # read images from hdf5 file
    read_recording_folderpath = read_cameras and (recording_folderpath is not None) # read images from MP4

    traj_reader = TrajectoryReader(filepath, read_images=read_hdf5_images)
    if read_recording_folderpath:
        camera_reader = RecordedMultiCameraWrapper(recording_folderpath, camera_kwargs)

    horizon = traj_reader.length()
    timestep_list = []

    # Choose Timesteps To Save #
    if num_samples_per_traj:
        num_to_save = num_samples_per_traj
        if remove_skipped_steps:
            num_to_save = int(num_to_save * num_samples_per_traj_coeff)
        max_size = min(num_to_save, horizon)
        indices_to_save = np.sort(np.random.choice(horizon, size=max_size, replace=False))
    else:
        indices_to_save = np.arange(horizon)

    # Iterate Over Trajectory #
    for i in indices_to_save:
        # Get HDF5 Data #
        timestep = traj_reader.read_timestep(index=i)

        # If Applicable, Get Recorded Data #
        if read_recording_folderpath:
            timestamp_dict = timestep["observation"]["timestamp"]["cameras"]
            # The camera_type dictionaries are incorrectly populated, so we overwrite them to contain the same camera IDs.
            timestep["observation"]["camera_type"] = {str(wrist_cam_id): 0, f"{wrist_cam_id}_depth": 0, str(static_cam_id): 1, f"{static_cam_id}_depth": 1}
            camera_type_dict = {
                k: CAMERA_TYPE_TO_STRING_DICT[v] for k, v in timestep["observation"]["camera_type"].items()
            }
            camera_obs = camera_reader.read_cameras(
                index=i, camera_type_dict=camera_type_dict, timestamp_dict=timestamp_dict
            )
            camera_failed = camera_obs is None

            # Add Data To Timestep If Successful #
            if camera_failed:
                print(f"Failed to read camera")
                break
            else:
                timestep["observation"].update(camera_obs)
        
        # Filter Steps #
        step_skipped = not timestep["observation"]["controller_info"].get("movement_enabled", True)
        delete_skipped_step = step_skipped and remove_skipped_steps

        # Save Filtered Timesteps #
        if delete_skipped_step:
            del timestep
        else:
            timestep_list.append(timestep)

    # Remove Extra Transitions #
    timestep_list = np.array(timestep_list)
    if (num_samples_per_traj is not None) and (len(timestep_list) > num_samples_per_traj):
        ind_to_keep = np.random.choice(len(timestep_list), size=num_samples_per_traj, replace=False)
        timestep_list = timestep_list[ind_to_keep]

    # Get Target Label #
    target_label = traj_reader.get_target_label()

    # Close Readers #
    traj_reader.close()

    # Return Data #
    return timestep_list, target_label


class PPGM(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Pre-Trained Panda Grasping Model (PPGM)."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
            'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'wrist_image': tfds.features.Image(
                            shape=(270, 360, 3), # downsampled from (480, 640, 3)
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB image',
                        ),
                        'wrist_depth_image': tfds.features.Image(
                            shape=(270, 360, 1), # downsampled from (480, 640, 1)
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera depth image',
                        ),
                        'static_image': tfds.features.Image(
                            shape=(270, 360, 3), # downsampled from (480, 640, 3)
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Static third-person camera RGB image',
                        ),
                        'static_depth_image': tfds.features.Image(
                            shape=(270, 360, 1), # downsampled from (480, 640, 1)
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Static third-person camera depth image',
                        ),
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Robot Cartesian state',
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float32,
                            doc='Gripper position statae',
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Joint position state'
                        )
                    }),
                    'action_dict': tfds.features.FeaturesDict({
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Commanded Cartesian position'
                        ),
                        'cartesian_velocity': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Commanded Cartesian velocity'
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float32,
                            doc='Commanded gripper position'
                        ),
                        'gripper_velocity': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float32,
                            doc='Commanded gripper velocity'
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Commanded joint position'
                        ),
                        'joint_velocity': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Commanded joint velocity'
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [6x joint velocities, \
                            1x gripper position].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'recording_folderpath': tfds.features.Text(
                        doc='Path to the folder of recordings.'
                    )
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/scr/moojink/data/R2D2/data_debug_2023-11-01_47-demos/', wrist_cam_id=138422074005, static_cam_id=140122076178), # TODO remove hardcode
            #'val': self._generate_examples(''),
        }


    def _generate_examples(self, path, wrist_cam_id, static_cam_id) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _resize_and_encode(image, size):
            assert len(size) == 2
            # In PIL, image size is (W, H, C) as opposed to (H, W, C), so flip the size if needed.
            if size[0] < size[1]:
                size = size[::-1]
            image = Image.fromarray(image)
            image = image.resize(size, Image.Resampling.LANCZOS)
            image = np.array(image)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1) # (H, W) to (H, W, 1)
            return image

        def _parse_example(episode_path):
            FRAMESKIP = 1

            h5_filepath = os.path.join(episode_path, 'trajectory.h5')
            recording_folderpath = os.path.join(episode_path, 'recordings', 'MP4')

            traj, target_label = load_trajectory(h5_filepath, wrist_cam_id=wrist_cam_id, static_cam_id=static_cam_id, read_cameras=True, recording_folderpath=recording_folderpath)
            data = traj[::FRAMESKIP]

            assert all(t.keys() == data[0].keys() for t in data) # check that all steps have the same dict keys
            # Resize and encode all images.
            for t in range(len(data)):
                for key in data[0]['observation']['image'].keys():
                    data[t]['observation']['image'][key] = _resize_and_encode(data[t]['observation']['image'][key], size=(270, 360))

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                obs = step['observation']
                action = step['action']
                language_instruction = f"pick {target_label}"
                camera_type_dict = obs['camera_type']
                wrist_cam_ids = [k for k, v in camera_type_dict.items() if v == 0]
                static_cam_ids = [k for k, v in camera_type_dict.items() if v != 0]

                episode.append({
                    'observation': {
                        'wrist_image': obs['image'][f'{wrist_cam_ids[0]}'],
                        'wrist_depth_image': obs['image'][f'{wrist_cam_ids[1]}'],
                        'static_image': obs['image'][f'{static_cam_ids[0]}'],
                        'static_depth_image': obs['image'][f'{static_cam_ids[1]}'],
                        'cartesian_position': np.array(obs['robot_state']['cartesian_position'], dtype=np.float32),
                        'joint_position': np.array(obs['robot_state']['joint_positions'], dtype=np.float32),
                        'gripper_position': np.array([obs['robot_state']['gripper_position']], dtype=np.float32),
                    },
                    'action_dict': {
                        'cartesian_position': np.array(action['cartesian_position'], dtype=np.float32),
                        'cartesian_velocity': np.array(action['cartesian_velocity'], dtype=np.float32),
                        'gripper_position': np.array([action['gripper_position']], dtype=np.float32),
                        'gripper_velocity': np.array([action['gripper_velocity']], dtype=np.float32),
                        'joint_position': np.array(action['joint_position'], dtype=np.float32),
                        'joint_velocity': np.array(action['joint_velocity'], dtype=np.float32),
                    },
                    'action': np.concatenate((action['cartesian_velocity'], [action['gripper_velocity']]), dtype=np.float32),
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': language_instruction,
                })
            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': h5_filepath,
                    'recording_folderpath': recording_folderpath
                }
            }
            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = crawler(path)
        episode_paths = [p for p in episode_paths if os.path.exists(p + '/trajectory.h5') and \
                os.path.exists(p + '/recordings/MP4')]

        # # for smallish datasets, use single-thread parsing
        # for sample in episode_paths:
        #     yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        beam = tfds.core.lazy_imports.apache_beam
        return (
                beam.Create(episode_paths)
                | beam.Map(_parse_example)
        )


