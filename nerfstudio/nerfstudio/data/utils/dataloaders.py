# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code for sampling images from a dataset of images.
"""

# for multithreading
import concurrent.futures
import multiprocessing
import random
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sized, Tuple, Union

import torch
from rich.progress import track
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.utils.misc import get_dict_to_torch
from nerfstudio.utils.rich_utils import CONSOLE


class CacheDataloader(DataLoader):
    """Collated image dataset that implements caching of default-pytorch-collatable data.
    Creates batches of the InputDataset return type.

    Args:
        dataset: Dataset to sample from.
        num_samples_to_collate: How many images to sample rays for each batch. -1 for all images.
        num_times_to_repeat_images: How often to collate new images. -1 to never pick new images.
        device: Device to perform computation.
        collate_fn: The function we will use to collate our training data
    """

    def __init__(
        self,
        dataset: Dataset,
        num_images_to_sample_from: int = -1,
        num_times_to_repeat_images: int = -1,
        device: Union[torch.device, str] = "cpu",
        collate_fn: Callable[[Any], Any] = nerfstudio_collate,
        exclude_batch_keys_from_device: Optional[List[str]] = None,
        **kwargs,
    ):
        if exclude_batch_keys_from_device is None:
            exclude_batch_keys_from_device = ["image"]
        self.dataset = dataset  # 保存了训练/验证数据集的SDFDataset类，基于pytorch的dataset类
        assert isinstance(self.dataset, Sized)

        super().__init__(dataset=dataset, **kwargs)  # 初始化pytorch.dataloader类 This will set self.dataset
        self.num_times_to_repeat_images = num_times_to_repeat_images    # -1 
        self.cache_all_images = (num_images_to_sample_from == -1) or (num_images_to_sample_from >= len(self.dataset))   # true
        # len(dataset)返回的是数据集类中样本的数量
        self.num_images_to_sample_from = len(self.dataset) if self.cache_all_images else num_images_to_sample_from      # 49
        self.device = device                                                    # cuda
        self.collate_fn = collate_fn                                            # 自定义的对数据批处理操作
        self.num_workers = kwargs.get("num_workers", 0)                         # 4
        self.exclude_batch_keys_from_device = exclude_batch_keys_from_device    # ['image', 'mask', 'depth', 'normal']
        self.num_repeated = self.num_times_to_repeat_images                     # starting value -1
        self.first_time = True

        self.cached_collated_batch = None
        if self.cache_all_images:   # true
            CONSOLE.print(f"Caching all {len(self.dataset)} images.")
            if len(self.dataset) > 500:
                CONSOLE.print(
                    "[bold yellow]Warning: If you run out of memory, try reducing the number of images to sample from."
                )
            self.cached_collated_batch = self._get_collated_batch()
        elif self.num_times_to_repeat_images == -1:
            CONSOLE.print(
                f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, without resampling."
            )
        else:
            CONSOLE.print(
                f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, "
                f"resampling every {self.num_times_to_repeat_images} iters."
            )

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)
   
    # !! 从数据集中随机采样一定数量的样本
    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""

        assert isinstance(self.dataset, Sized)
        indices = random.sample(range(len(self.dataset)), k=self.num_images_to_sample_from) # 从数据集中随机选择num_images_to_sample_from个索引，其实就是生成同数据量的随机索引
        batch_list = []
        results = []
        
        # * 多线程操作设置
        num_threads = int(self.num_workers) * 4                         # 4x4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1) # 如果cpu核心不足，就设置为cpu核心数
        num_threads = max(num_threads, 1)                               # 至少有1个线程
        
        # * 使用 concurrent.futures.ThreadPoolExecutor 创建一个线程池执行器，最大工作线程数为 num_threads。
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)    # 通过线程池执行器执行数据集的__getitem__()
                results.append(res)                                     # 将任务结果添加到results列表中
           
            # track 函数，该函数通常用于追踪多个并发任务的执行进度
            # results:包含了并发任务的结果的列表
            for res in track(results, description="Loading data batch", transient=True):
                batch_list.append(res.result())
        # ******* 最后返回的结果见base_dataset的get_data()，即self.dataset.__getitem__实际调用的函数
        # 肯定会有:image_idx和image
        # todo：需要添加mask
        return batch_list

    # !! 得到修改存储形式后的数据集
    # 存储形式是字典:{'image_idx': tensor([47,  6, 43, ...='cuda:0'), 'image': tensor([[[[0.0431, 0....1765]]]])}
    def _get_collated_batch(self):
        """Returns a collated batch."""
        # * 得到每张图片的信息[{'image_idx:0,'image':tensor(...)},...,{'image_idx:1,'image':tensor(...)}]
        batch_list = self._get_batch_list()             # 得到所有图片的list
        # * 上面的列表变成词典,每个键保存所有图片对应的信息：{'image_idx': tensor([47,  6, 43, ...='cuda:0'), 'image': tensor([[[[0.0431, 0....1765]]]])}
        collated_batch = self.collate_fn(batch_list)    # 调用的是nerfstudio_collate.py的nerfstudio_collate函数
        # * 把字典数据发送到gpu中去
        collated_batch = get_dict_to_torch(
            collated_batch, device=self.device, exclude=self.exclude_batch_keys_from_device
        )
        return collated_batch

    def __iter__(self):
        while True:
            if self.cache_all_images:
                collated_batch = self.cached_collated_batch
            elif self.first_time or (
                self.num_times_to_repeat_images != -1 and self.num_repeated >= self.num_times_to_repeat_images
            ):
                # trigger a reset
                self.num_repeated = 0
                collated_batch = self._get_collated_batch()
                # possibly save a cached item
                self.cached_collated_batch = collated_batch if self.num_times_to_repeat_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch


class EvalDataloader(DataLoader):
    """Evaluation dataloader base class

    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        self.input_dataset = input_dataset
        self.cameras = input_dataset.cameras.to(device)
        self.device = device
        self.kwargs = kwargs
        super().__init__(dataset=input_dataset)

    @abstractmethod
    def __iter__(self):
        """Iterates over the dataset"""
        return self

    @abstractmethod
    def __next__(self) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data"""

    def get_camera(self, image_idx: int = 0) -> Tuple[Cameras, Dict]:
        """Get camera for the given image index

        Args:
            image_idx: Camera image index
        """
        camera = self.cameras[image_idx : image_idx + 1]
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        assert isinstance(batch, dict)
        return camera, batch

    def get_data_from_image_idx(self, image_idx: int) -> Tuple[RayBundle, Dict]:
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        ray_bundle = self.cameras.generate_rays(camera_indices=image_idx, keep_shape=True)
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        assert isinstance(batch, dict)
        return ray_bundle, batch


class FixedIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns a fixed set of indices.

    Args:
        input_dataset: InputDataset to load data from
        image_indices: List of image indices to load data from. If None, then use all images.
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        image_indices: Optional[Tuple[int]] = None,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
        if image_indices is None:
            self.image_indices = list(range(len(input_dataset)))
        else:
            self.image_indices = image_indices
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.image_indices):
            image_idx = self.image_indices[self.count]
            camera, batch = self.get_camera(image_idx)
            self.count += 1
            return camera, batch
        raise StopIteration


class RandIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns random images.
    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __iter__(self):
        return self

    def __next__(self):
        # choose a random image index
        image_idx = random.randint(0, len(self.cameras) - 1)
        camera, batch = self.get_camera(image_idx)
        return camera, batch
