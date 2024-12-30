import torch
from torch.utils.data import Dataset
import json
import os
import statistics
from PIL import Image, ImageFilter
from torchvision.transforms import v2
import numpy as np
import io


class FfppDataset_temporal_whole_diff(Dataset):
    """

    """
    data_root = '/cluster/home/xuyi/xuyi/FF++'
    data_list = {
        'test': '/cluster/home/xuyi/xuyi/FF++/splits/test_tsne.json',
        # 'train': '/cluster/home/xuyi/xuyi/FF++/splits/train.json',
        # 'val': '/cluster/home/xuyi/xuyi/FF++/splits/val.json',
        'train': '/cluster/home/xuyi/xuyi/FF++/splits/train_test.json',
        'val': '/cluster/home/xuyi/xuyi/FF++/splits/val_test.json',
        # 'test': '/cluster/home/xuyi/xuyi/FF++/splits/val_test.json',
        # 'tsne': '/cluster/home/xuyi/xuyi/FF++/splits/test_tsne.json'
    }

    def __init__(self, dataset='ALL', transform=None, mode='train', step=1, quality='c23', segment_len=2,
                 square_scale=1.0, frames=None, interval=1, gn=0, gb=0, cp=100):
        self.mode = mode
        self.dataset = dataset
        self.transform = transform
        self.segment_len = segment_len
        self.square_scale = square_scale
        self.interval = interval
        self.gn = gn
        self.gb = gb
        self.cp = cp

        default_frames = {'test': 100, 'val': 100, 'train': 200, 'val_test': 2, 'train_test': 5}
        self.frames = frames if frames is not None else default_frames

        for transform in self.transform.transforms:
            # Check if the current transformation is an instance of the Resize operation
            if isinstance(transform, v2.CenterCrop):
                self.resize_size = transform.size

        with open(self.data_list[mode], 'r') as fd:
            data = json.load(fd)
            self.img_lines = []

            for pair in data:
                r1, r2 = pair
                for i in range(0, self.frames[mode], step):
                    self.img_lines.append(('original_sequences/{}/{}/frames/{}'.format('youtube', quality, r1), i, 0))
                    self.img_lines.append(('original_sequences/{}/{}/frames/{}'.format('youtube', quality, r2), i, 0))

                    if dataset == 'ALL':
                        for fake_d in ['Deepfakes', 'NeuralTextures', 'FaceSwap', 'Face2Face']:
                            self.img_lines.append(
                                ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r1, r2), i, 1))
                            self.img_lines.append(
                                ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r2, r1), i, 1))

                    else:
                        for ds in dataset:
                            self.img_lines.append(
                                ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r1, r2), i, 1))
                            self.img_lines.append(
                                ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r2, r1), i, 1))


        self.usable_imgs = []
        for (name, idx, label) in self.img_lines:
            is_valid, bbox = self.segment_sanity_check(name, idx)
            if is_valid:
                # img_path = '{}/{}/{:06d}.jpg'.format(self.data_root, name, int(idx))
                self.usable_imgs.append((name, idx, segment_len, label, bbox))

    def segment_sanity_check(self, name, idx):
        most_left = []
        most_up = []
        most_right = []
        most_bottom = []
        try:
            for i in range(idx, idx + (self.segment_len-1) * self.interval, self.interval):
                score, landmark = self.load_landmark(name, i)  # Use i for current index

                if score is not None:
                    mean = statistics.mean(score)
                    if mean < 0.5:
                        # If any frame's mean score is below 0.5, the segment is invalid
                        return False, None
                else:
                    # If there's no score, the frame is considered invalid
                    return False, None

                most_left.append(min(landmark, key=lambda x: x[0]))
                most_right.append(max(landmark, key=lambda x: x[0]))
                most_up.append(min(landmark, key=lambda x: x[1]))
                most_bottom.append(max(landmark, key=lambda x: x[1]))

            left = min(most_left, key=lambda x: x[0])
            right = max(most_right, key=lambda x: x[0])
            up = min(most_up, key=lambda x: x[1])
            bottom = max(most_bottom, key=lambda x: x[1])
            return True, [left[0], right[0], up[1], bottom[1]]

        except Exception as e:
            # In case of any exception, consider the segment invalid
            return False, None

    def load_face_square(self, name, bbox):
        frame = Image.open(name)
        most_right, most_bottom = frame.size
        left, right, up, bottom = bbox

        height = bottom - up
        width = right - left
        up_padding = max(int(up - height * (self.square_scale - 1) // 2), 0)
        bottom_padding = min(int(bottom + height * (self.square_scale - 1) // 2), most_bottom)
        left_padding = max(int(left - width * (self.square_scale - 1) // 2), 0)
        right_padding = min(int(right + width * (self.square_scale - 1) // 2), most_right)
        face_img = frame.crop((left_padding, up_padding, right_padding, bottom_padding))
        return face_img

    def load_landmark(self, name, idx):
        landmark_json = os.path.join(os.path.join(self.data_root, name).replace('frames', 'landmark'), 'landmarks.json')
        with open(landmark_json, 'r') as j:
            video_info = json.loads(j.read())
        score = video_info[str(idx)]['score']
        landmark = video_info[str(idx)]['ldm']

        return score, landmark

    def add_gaussian_noise(self, image, mean=0, std=1):
        """
        Add Gaussian noise to an image.

        Parameters:
            image (np.array): The original image.
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.

        Returns:
            np.array: Image with Gaussian noise added.
        """
        gaussian = np.random.normal(mean, std, image.shape)
        noisy_image = image + gaussian
        noisy_image = np.clip(noisy_image, 0, 255)  # Ensure values are valid for an image
        return noisy_image.astype(image.dtype)

    def add_gaussian_blur(self, image, radius=1):
        """
        Apply Gaussian blur to the given image using a specified radius.

        Args:
        image (PIL.Image.Image): Input image to which Gaussian blur will be applied.
        radius (int or float): Radius of the Gaussian blur.

        Returns:
        PIL.Image.Image: Blurred image.
        """
        image_pil = Image.fromarray(image)
        blurred_image = image_pil.filter(ImageFilter.GaussianBlur(radius))
        blurred_image_np = np.array(blurred_image)
        return blurred_image_np

    def add_compression(self, image, quality=100):
        """
        Apply JPEG compression to the image to simulate compression artifacts.

        Parameters:
        - image: Input image (NumPy array).
        - quality: Compression quality level (lower means more compression, default is 25).

        Returns:
        - Compressed image as a NumPy array.
        """
        # Convert the NumPy array to a PIL Image
        pil_image = Image.fromarray(image)

        # Save the image to a bytes buffer with JPEG compression
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)

        # Load the image back from the buffer
        buffer.seek(0)
        compressed_image = Image.open(buffer)

        # Convert the compressed image back to a NumPy array
        compressed_image_np = np.array(compressed_image)

        return compressed_image_np

    def __getitem__(self, index):
        segment = torch.zeros(self.segment_len - 1, 3, self.resize_size[0], self.resize_size[1])

        folder_name, idx, segment_len, label, bbox = self.usable_imgs[index]

        for i, frame_idx in enumerate(range(idx, idx + (self.segment_len-1) * self.interval, self.interval)):
            img_path_0 = os.path.join(self.data_root, folder_name, f"{frame_idx:06d}.jpg")
            img_path_1 = os.path.join(self.data_root, folder_name, f"{frame_idx+self.interval:06d}.jpg")
            # print(i, img_path_0, img_path_1)
            face_img_0 = self.load_face_square(img_path_0, bbox)
            face_img_1 = self.load_face_square(img_path_1, bbox)

            face_img_0 = np.array(face_img_0)
            face_img_1 = np.array(face_img_1)

            if self.gn:
                face_img_0 = self.add_gaussian_noise(face_img_0, std=self.gn)
                face_img_1 = self.add_gaussian_noise(face_img_1, std=self.gn)
            if self.gb:
                face_img_0 = self.add_gaussian_blur(face_img_0, radius=self.gb)
                face_img_1 = self.add_gaussian_blur(face_img_1, radius=self.gb)
            if self.cp != 100:
                assert 0 <= self.cp < 100
                face_img_0 = self.add_compression(face_img_0, quality=self.cp)
                face_img_1 = self.add_compression(face_img_1, quality=self.cp)

            face_img_diff = abs(face_img_0 - face_img_1)

            img = self.transform(face_img_diff)

            segment[i] = img

        segment = segment.permute(1, 0, 2, 3)
        return segment, label, folder_name, idx, segment_len

    def __len__(self):
        return len(self.usable_imgs)


class FfppDataset_temporal_whole_seg(Dataset):
    """

    """
    data_root = '/cluster/home/xuyi/xuyi/FF++'
    data_list = {
        'test': '/cluster/home/xuyi/xuyi/FF++/splits/test.json',
        'train': '/cluster/home/xuyi/xuyi/FF++/splits/train.json',
        'val': '/cluster/home/xuyi/xuyi/FF++/splits/val.json'
        # 'train': '/cluster/home/xuyi/xuyi/FF++/splits/train_test.json',
        # 'val': '/cluster/home/xuyi/xuyi/FF++/splits/val_test.json'
    }

    def __init__(self, dataset='ALL', transform=None, mode='train', step=1, quality='c23', segment_len=2,
                 square_scale=1.0, frames=None):
        self.mode = mode
        self.dataset = dataset
        self.transform = transform
        self.segment_len = segment_len
        self.square_scale = square_scale

        default_frames = {'test': 100, 'val': 100, 'train': 200}
        self.frames = frames if frames is not None else default_frames

        for transform in self.transform.transforms:
            # Check if the current transformation is an instance of the Resize operation
            if isinstance(transform, v2.CenterCrop):
                self.resize_size = transform.size

        with open(self.data_list[mode], 'r') as fd:
            data = json.load(fd)
            self.img_lines = []

            for pair in data:
                r1, r2 = pair
                for i in range(0, self.frames[mode], step):
                    self.img_lines.append(('original_sequences/{}/{}/frames/{}'.format('youtube', quality, r1), i, 0))
                    self.img_lines.append(('original_sequences/{}/{}/frames/{}'.format('youtube', quality, r2), i, 0))

                    if self.mode == 'train':
                        if dataset == 'ALL':
                            if i > self.frames[mode] // 4:
                                continue
                            for fake_d in ['Deepfakes', 'NeuralTextures', 'FaceSwap', 'Face2Face']:
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r2, r1), i, 1))

                        else:
                            for ds in dataset:
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r2, r1), i, 1))

                    else:
                        if dataset == 'ALL':
                            if i > self.frames[mode] // 4:
                                continue
                            for fake_d in ['Deepfakes', 'NeuralTextures', 'FaceSwap', 'Face2Face']:
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r2, r1), i, 1))
                        else:
                            for ds in dataset:
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r2, r1), i, 1))

        self.usable_imgs = []
        for (name, idx, label) in self.img_lines:
            is_valid, bbox = self.segment_sanity_check(name, idx, self.segment_len)
            if is_valid:
                # img_path = '{}/{}/{:06d}.jpg'.format(self.data_root, name, int(idx))
                self.usable_imgs.append((name, idx, segment_len, label, bbox))

    def segment_sanity_check(self, name, idx, segment_len):
        most_left = []
        most_up = []
        most_right = []
        most_bottom = []
        try:
            for i in range(idx, idx + segment_len):
                score, landmark = self.load_landmark(name, i)  # Use i for current index

                if score is not None:
                    mean = statistics.mean(score)
                    if mean < 0.5:
                        # If any frame's mean score is below 0.5, the segment is invalid
                        return False, None
                else:
                    # If there's no score, the frame is considered invalid
                    return False, None

                most_left.append(min(landmark, key=lambda x: x[0]))
                most_right.append(max(landmark, key=lambda x: x[0]))
                most_up.append(min(landmark, key=lambda x: x[1]))
                most_bottom.append(max(landmark, key=lambda x: x[1]))

            left = min(most_left, key=lambda x: x[0])
            right = max(most_right, key=lambda x: x[0])
            up = min(most_up, key=lambda x: x[1])
            bottom = max(most_bottom, key=lambda x: x[1])
            return True, [left[0], right[0], up[1], bottom[1]]

        except Exception as e:
            # In case of any exception, consider the segment invalid
            return False, None

    def load_face_square(self, name, bbox):
        frame = Image.open(name)
        most_right, most_bottom = frame.size
        left, right, up, bottom = bbox

        height = bottom - up
        width = right - left
        up_padding = max(int(up - height * (self.square_scale - 1) // 2), 0)
        bottom_padding = min(int(bottom + height * (self.square_scale - 1) // 2), most_bottom)
        left_padding = max(int(left - width * (self.square_scale - 1) // 2), 0)
        right_padding = min(int(right + width * (self.square_scale - 1) // 2), most_right)
        face_img = frame.crop((left_padding, up_padding, right_padding, bottom_padding))
        return face_img

    def load_landmark(self, name, idx):
        landmark_json = os.path.join(os.path.join(self.data_root, name).replace('frames', 'landmark'), 'landmarks.json')
        with open(landmark_json, 'r') as j:
            video_info = json.loads(j.read())
        score = video_info[str(idx)]['score']
        landmark = video_info[str(idx)]['ldm']

        return score, landmark

    def __getitem__(self, index):
        segment = torch.zeros(self.segment_len - 1, 3, self.resize_size[0], self.resize_size[1])

        folder_name, idx, segment_len, label, bbox = self.usable_imgs[index]

        for i, frame_idx in enumerate(range(idx, idx + self.segment_len)):
            img_path = os.path.join(self.data_root, folder_name, f"{frame_idx:06d}.jpg")
            face_img = self.load_face_square(img_path, bbox)
            img = self.transform(face_img)
            segment[i] = img

        segment = segment.permute(1, 0, 2, 3)
        return segment, label, folder_name, idx, segment_len

    def __len__(self):
        return len(self.usable_imgs)



class FfppDataset_temporal_whole_diff_fix1frame(Dataset):
    """

    """
    data_root = '/cluster/home/xuyi/xuyi/FF++'
    data_list = {
        'test': '/cluster/home/xuyi/xuyi/FF++/splits/test.json',
        'train': '/cluster/home/xuyi/xuyi/FF++/splits/train.json',
        'val': '/cluster/home/xuyi/xuyi/FF++/splits/val.json'
        # 'train': '/cluster/home/xuyi/xuyi/FF++/splits/train_test.json',
        # 'val': '/cluster/home/xuyi/xuyi/FF++/splits/val_test.json'
    }

    def __init__(self, dataset='ALL', transform=None, mode='train', step=1, quality='c23', segment_len=2,
                 square_scale=1.0, frames=None):
        self.mode = mode
        self.dataset = dataset
        self.transform = transform
        self.segment_len = segment_len
        self.square_scale = square_scale

        default_frames = {'test': 100, 'val': 100, 'train': 200}
        self.frames = frames if frames is not None else default_frames

        for transform in self.transform.transforms:
            # Check if the current transformation is an instance of the Resize operation
            if isinstance(transform, v2.CenterCrop):
                self.resize_size = transform.size

        with open(self.data_list[mode], 'r') as fd:
            data = json.load(fd)
            self.img_lines = []

            for pair in data:
                r1, r2 = pair
                for i in range(0, self.frames[mode], step):
                    self.img_lines.append(('original_sequences/{}/{}/frames/{}'.format('youtube', quality, r1), i, 0))
                    self.img_lines.append(('original_sequences/{}/{}/frames/{}'.format('youtube', quality, r2), i, 0))

                    if self.mode == 'train':
                        if dataset == 'ALL':
                            if i > self.frames[mode] // 4:
                                continue
                            for fake_d in ['Deepfakes', 'NeuralTextures', 'FaceSwap', 'Face2Face']:
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r2, r1), i, 1))

                        else:
                            for ds in dataset:
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r2, r1), i, 1))

                    else:
                        if dataset == 'ALL':
                            if i > self.frames[mode] // 4:
                                continue
                            for fake_d in ['Deepfakes', 'NeuralTextures', 'FaceSwap', 'Face2Face']:
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r2, r1), i, 1))
                        else:
                            for ds in dataset:
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r2, r1), i, 1))

        self.usable_imgs = []
        for (name, idx, label) in self.img_lines:
            is_valid, bbox = self.segment_sanity_check(name, idx, self.segment_len)
            if is_valid:
                # img_path = '{}/{}/{:06d}.jpg'.format(self.data_root, name, int(idx))
                self.usable_imgs.append((name, idx, segment_len, label, bbox))

    def segment_sanity_check(self, name, idx, segment_len):
        most_left = []
        most_up = []
        most_right = []
        most_bottom = []
        try:
            for i in range(idx, idx + segment_len):
                score, landmark = self.load_landmark(name, i)  # Use i for current index

                if score is not None:
                    mean = statistics.mean(score)
                    if mean < 0.5:
                        # If any frame's mean score is below 0.5, the segment is invalid
                        return False, None
                else:
                    # If there's no score, the frame is considered invalid
                    return False, None

                most_left.append(min(landmark, key=lambda x: x[0]))
                most_right.append(max(landmark, key=lambda x: x[0]))
                most_up.append(min(landmark, key=lambda x: x[1]))
                most_bottom.append(max(landmark, key=lambda x: x[1]))

            left = min(most_left, key=lambda x: x[0])
            right = max(most_right, key=lambda x: x[0])
            up = min(most_up, key=lambda x: x[1])
            bottom = max(most_bottom, key=lambda x: x[1])
            return True, [left[0], right[0], up[1], bottom[1]]

        except Exception as e:
            # In case of any exception, consider the segment invalid
            return False, None

    def load_face_square(self, name, bbox):
        frame = Image.open(name)
        most_right, most_bottom = frame.size
        left, right, up, bottom = bbox

        height = bottom - up
        width = right - left
        up_padding = max(int(up - height * (self.square_scale - 1) // 2), 0)
        bottom_padding = min(int(bottom + height * (self.square_scale - 1) // 2), most_bottom)
        left_padding = max(int(left - width * (self.square_scale - 1) // 2), 0)
        right_padding = min(int(right + width * (self.square_scale - 1) // 2), most_right)
        face_img = frame.crop((left_padding, up_padding, right_padding, bottom_padding))
        return face_img

    def load_landmark(self, name, idx):
        landmark_json = os.path.join(os.path.join(self.data_root, name).replace('frames', 'landmark'), 'landmarks.json')
        with open(landmark_json, 'r') as j:
            video_info = json.loads(j.read())
        score = video_info[str(idx)]['score']
        landmark = video_info[str(idx)]['ldm']

        return score, landmark

    def __getitem__(self, index):
        segment = torch.zeros(self.segment_len - 1, 3, self.resize_size[0], self.resize_size[1])

        folder_name, idx, segment_len, label, bbox = self.usable_imgs[index]

        for i, frame_idx in enumerate(range(idx, idx + self.segment_len - 1)):
            img_path_0 = os.path.join(self.data_root, folder_name, f"{idx:06d}.jpg")
            img_path_1 = os.path.join(self.data_root, folder_name, f"{idx+i+1:06d}.jpg")
            face_img_0 = self.load_face_square(img_path_0, bbox)
            face_img_1 = self.load_face_square(img_path_1, bbox)

            face_img_0 = np.array(face_img_0)
            face_img_1 = np.array(face_img_1)

            face_img_diff = abs(face_img_0 - face_img_1)

            img = self.transform(face_img_diff)

            segment[i] = img

        segment = segment.permute(1, 0, 2, 3)
        return segment, label, folder_name, idx, segment_len

    def __len__(self):
        return len(self.usable_imgs)



class FfppDataset_temporal_whole_diff_raw_c23(Dataset):
    """

    """
    data_root = '/cluster/home/xuyi/xuyi/FF++'
    data_list = {
        'test': '/cluster/home/xuyi/xuyi/FF++/splits/test.json',
        'train': '/cluster/home/xuyi/xuyi/FF++/splits/train.json',
        'val': '/cluster/home/xuyi/xuyi/FF++/splits/val.json'
        # 'train': '/cluster/home/xuyi/xuyi/FF++/splits/train_test.json',
        # 'val': '/cluster/home/xuyi/xuyi/FF++/splits/val_test.json'
    }

    def __init__(self, dataset='ALL', transform=None, mode='train', step=1, quality='c23', segment_len=2,
                 square_scale=1.0, frames=None):
        self.mode = mode
        self.dataset = dataset
        self.transform = transform
        self.segment_len = segment_len
        self.square_scale = square_scale

        default_frames = {'test': 100, 'val': 100, 'train': 200}
        self.frames = frames if frames is not None else default_frames

        for transform in self.transform.transforms:
            # Check if the current transformation is an instance of the Resize operation
            if isinstance(transform, v2.CenterCrop):
                self.resize_size = transform.size

        with open(self.data_list[mode], 'r') as fd:
            data = json.load(fd)
            self.img_lines = []

            for pair in data:
                r1, r2 = pair
                for i in range(0, self.frames[mode], step):
                    self.img_lines.append(('original_sequences/{}/{}/frames/{}'.format('youtube', quality, r1), i, 0))
                    self.img_lines.append(('original_sequences/{}/{}/frames/{}'.format('youtube', quality, r2), i, 0))
                    self.img_lines.append(('original_sequences/{}/{}/frames/{}'.format('youtube', 'raw', r1), i, 0))
                    self.img_lines.append(('original_sequences/{}/{}/frames/{}'.format('youtube', 'raw', r2), i, 0))

                    if self.mode == 'train':
                        if dataset == 'ALL':
                            if i > self.frames[mode] // 4:
                                continue
                            for fake_d in ['Deepfakes', 'NeuralTextures', 'FaceSwap', 'Face2Face']:
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r2, r1), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, 'raw', r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, 'raw', r2, r1), i, 1))

                        else:
                            for ds in dataset:
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r2, r1), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, 'raw', r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, 'raw', r2, r1), i, 1))

                    else:
                        if dataset == 'ALL':
                            if i > self.frames[mode] // 4:
                                continue
                            for fake_d in ['Deepfakes', 'NeuralTextures', 'FaceSwap', 'Face2Face']:
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r2, r1), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, 'raw', r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, 'raw', r2, r1), i, 1))
                        else:
                            for ds in dataset:
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r2, r1), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, 'raw', r1, r2), i, 1))
                                self.img_lines.append(
                                    ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, 'raw', r2, r1), i, 1))

        self.usable_imgs = []
        for (name, idx, label) in self.img_lines:
            is_valid, bbox = self.segment_sanity_check(name, idx, self.segment_len)
            if is_valid:
                # img_path = '{}/{}/{:06d}.jpg'.format(self.data_root, name, int(idx))
                self.usable_imgs.append((name, idx, segment_len, label, bbox))

    def segment_sanity_check(self, name, idx, segment_len):
        most_left = []
        most_up = []
        most_right = []
        most_bottom = []
        try:
            for i in range(idx, idx + segment_len):
                score, landmark = self.load_landmark(name, i)  # Use i for current index

                if score is not None:
                    mean = statistics.mean(score)
                    if mean < 0.5:
                        # If any frame's mean score is below 0.5, the segment is invalid
                        return False, None
                else:
                    # If there's no score, the frame is considered invalid
                    return False, None

                most_left.append(min(landmark, key=lambda x: x[0]))
                most_right.append(max(landmark, key=lambda x: x[0]))
                most_up.append(min(landmark, key=lambda x: x[1]))
                most_bottom.append(max(landmark, key=lambda x: x[1]))

            left = min(most_left, key=lambda x: x[0])
            right = max(most_right, key=lambda x: x[0])
            up = min(most_up, key=lambda x: x[1])
            bottom = max(most_bottom, key=lambda x: x[1])
            return True, [left[0], right[0], up[1], bottom[1]]

        except Exception as e:
            # In case of any exception, consider the segment invalid
            return False, None

    def load_face_square(self, name, bbox):
        frame = Image.open(name)
        most_right, most_bottom = frame.size
        left, right, up, bottom = bbox

        height = bottom - up
        width = right - left
        up_padding = max(int(up - height * (self.square_scale - 1) // 2), 0)
        bottom_padding = min(int(bottom + height * (self.square_scale - 1) // 2), most_bottom)
        left_padding = max(int(left - width * (self.square_scale - 1) // 2), 0)
        right_padding = min(int(right + width * (self.square_scale - 1) // 2), most_right)
        face_img = frame.crop((left_padding, up_padding, right_padding, bottom_padding))
        return face_img

    def load_landmark(self, name, idx):
        landmark_json = os.path.join(os.path.join(self.data_root, name).replace('frames', 'landmark'), 'landmarks.json')
        with open(landmark_json, 'r') as j:
            video_info = json.loads(j.read())
        score = video_info[str(idx)]['score']
        landmark = video_info[str(idx)]['ldm']

        return score, landmark

    def __getitem__(self, index):
        segment = torch.zeros(self.segment_len - 1, 3, self.resize_size[0], self.resize_size[1])

        folder_name, idx, segment_len, label, bbox = self.usable_imgs[index]

        for i, frame_idx in enumerate(range(idx, idx + self.segment_len - 1)):
            img_path_0 = os.path.join(self.data_root, folder_name, f"{frame_idx:06d}.jpg")
            img_path_1 = os.path.join(self.data_root, folder_name, f"{frame_idx+1:06d}.jpg")
            face_img_0 = self.load_face_square(img_path_0, bbox)
            face_img_1 = self.load_face_square(img_path_1, bbox)

            face_img_0 = np.array(face_img_0)
            face_img_1 = np.array(face_img_1)

            face_img_diff = abs(face_img_0 - face_img_1)

            img = self.transform(face_img_diff)

            segment[i] = img

        segment = segment.permute(1, 0, 2, 3)
        return segment, label, folder_name, idx, segment_len

    def __len__(self):
        return len(self.usable_imgs)





class FfppDataset_temporal_whole_diff_c40(Dataset):
    """

    """
    data_root = '/cluster/home/xuyi/xuyi/FF++'
    data_list = {
        'test': '/cluster/home/xuyi/xuyi/FF++/splits/test.json',
        # 'train': '/cluster/home/xuyi/xuyi/FF++/splits/train.json',
        # 'val': '/cluster/home/xuyi/xuyi/FF++/splits/val.json',
        'train': '/cluster/home/xuyi/xuyi/FF++/splits/train_test.json',
        'val': '/cluster/home/xuyi/xuyi/FF++/splits/val_test.json'
    }

    def __init__(self, dataset='ALL', transform=None, mode='train', step=1, quality='c40', segment_len=2,
                 square_scale=1.0, frames=None):
        self.mode = mode
        self.dataset = dataset
        self.transform = transform
        self.segment_len = segment_len
        self.square_scale = square_scale

        default_frames = {'test': 100, 'val': 100, 'train': 200, 'val_test': 2, 'train_test': 5}
        self.frames = frames if frames is not None else default_frames

        for transform in self.transform.transforms:
            # Check if the current transformation is an instance of the Resize operation
            if isinstance(transform, v2.CenterCrop):
                self.resize_size = transform.size

        with open(self.data_list[mode], 'r') as fd:
            data = json.load(fd)
            self.img_lines = []

            for pair in data:
                r1, r2 = pair
                for i in range(0, self.frames[mode], step):
                    self.img_lines.append(('original_sequences/{}/{}/frames/{}'.format('youtube', quality, r1), i, 0))
                    self.img_lines.append(('original_sequences/{}/{}/frames/{}'.format('youtube', quality, r2), i, 0))

                    if dataset == 'ALL':
                        for fake_d in ['Deepfakes', 'NeuralTextures', 'FaceSwap', 'Face2Face']:
                            self.img_lines.append(
                                ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r1, r2), i, 1))
                            self.img_lines.append(
                                ('manipulated_sequences/{}/{}/frames/{}_{}'.format(fake_d, quality, r2, r1), i, 1))

                    else:
                        for ds in dataset:
                            self.img_lines.append(
                                ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r1, r2), i, 1))
                            self.img_lines.append(
                                ('manipulated_sequences/{}/{}/frames/{}_{}'.format(ds, quality, r2, r1), i, 1))


        self.usable_imgs = []
        for (name, idx, label) in self.img_lines:
            is_valid, bbox = self.segment_sanity_check(name, idx, self.segment_len)
            if is_valid:
                # img_path = '{}/{}/{:06d}.jpg'.format(self.data_root, name, int(idx))
                self.usable_imgs.append((name, idx, segment_len, label, bbox))

    def segment_sanity_check(self, name, idx, segment_len):
        most_left = []
        most_up = []
        most_right = []
        most_bottom = []
        try:
            for i in range(idx, idx + segment_len):
                score, landmark = self.load_landmark(name, i)  # Use i for current index

                if score is not None:
                    mean = statistics.mean(score)
                    if mean < 0.5:
                        # If any frame's mean score is below 0.5, the segment is invalid
                        return False, None
                else:
                    # If there's no score, the frame is considered invalid
                    return False, None

                most_left.append(min(landmark, key=lambda x: x[0]))
                most_right.append(max(landmark, key=lambda x: x[0]))
                most_up.append(min(landmark, key=lambda x: x[1]))
                most_bottom.append(max(landmark, key=lambda x: x[1]))

            left = min(most_left, key=lambda x: x[0])
            right = max(most_right, key=lambda x: x[0])
            up = min(most_up, key=lambda x: x[1])
            bottom = max(most_bottom, key=lambda x: x[1])
            return True, [left[0], right[0], up[1], bottom[1]]

        except Exception as e:
            # In case of any exception, consider the segment invalid
            return False, None

    def load_face_square(self, name, bbox):
        frame = Image.open(name)
        most_right, most_bottom = frame.size
        left, right, up, bottom = bbox

        height = bottom - up
        width = right - left
        up_padding = max(int(up - height * (self.square_scale - 1) // 2), 0)
        bottom_padding = min(int(bottom + height * (self.square_scale - 1) // 2), most_bottom)
        left_padding = max(int(left - width * (self.square_scale - 1) // 2), 0)
        right_padding = min(int(right + width * (self.square_scale - 1) // 2), most_right)
        face_img = frame.crop((left_padding, up_padding, right_padding, bottom_padding))
        return face_img

    def load_landmark(self, name, idx):
        landmark_json = os.path.join(os.path.join(self.data_root, name).replace('frames', 'landmark'), 'landmarks.json')
        with open(landmark_json, 'r') as j:
            video_info = json.loads(j.read())
        score = video_info[str(idx)]['score']
        landmark = video_info[str(idx)]['ldm']

        return score, landmark

    def __getitem__(self, index):
        segment = torch.zeros(self.segment_len - 1, 3, self.resize_size[0], self.resize_size[1])

        folder_name, idx, segment_len, label, bbox = self.usable_imgs[index]

        for i, frame_idx in enumerate(range(idx, idx + self.segment_len - 1)):
            img_path_0 = os.path.join(self.data_root, folder_name, f"{frame_idx:06d}.jpg")
            print(img_path_0)
            img_path_1 = os.path.join(self.data_root, folder_name, f"{frame_idx+1:06d}.jpg")
            face_img_0 = self.load_face_square(img_path_0, bbox)
            face_img_1 = self.load_face_square(img_path_1, bbox)

            face_img_0 = np.array(face_img_0)
            face_img_1 = np.array(face_img_1)

            face_img_diff = abs(face_img_0 - face_img_1)

            img = self.transform(face_img_diff)

            segment[i] = img

        segment = segment.permute(1, 0, 2, 3)
        return segment, label, folder_name, idx, segment_len

    def __len__(self):
        return len(self.usable_imgs)
