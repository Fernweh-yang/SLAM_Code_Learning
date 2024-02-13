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
Field for SDF based model, rather then estimating density to generate a surface,
a signed distance function (SDF) for surface representation is used to help with extracting high fidelity surfaces
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Type

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig
from nerfstudio.utils.external import tcnn


class LearnedVariance(nn.Module):
    """Variance network in NeuS

    Args:
        init_val: initial value in NeuS variance network
    """

    variance: Tensor

    def __init__(self, init_val):
        super().__init__()
        self.register_parameter("variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, x: Float[Tensor, "1"]) -> Float[Tensor, "1"]:
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)

    def get_variance(self) -> Float[Tensor, "1"]:
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


@dataclass
class SDFFieldConfig(FieldConfig):
    """SDF Field Config"""

    _target: Type = field(default_factory=lambda: SDFField)
    num_layers: int = 8
    """Number of layers for geometric network"""
    hidden_dim: int = 256
    """Number of hidden dimension of geometric network"""
    geo_feat_dim: int = 256
    """Dimension of geometric feature"""
    num_layers_color: int = 4
    """Number of layers for color network"""
    hidden_dim_color: int = 256
    """Number of hidden dimension of color network"""
    appearance_embedding_dim: int = 32
    """Dimension of appearance embedding"""
    use_appearance_embedding: bool = False
    """Whether to use appearance embedding"""
    bias: float = 0.8
    """Sphere size of geometric initialization"""
    geometric_init: bool = True
    """Whether to use geometric initialization"""
    inside_outside: bool = True
    """Whether to revert signed distance value, set to True for indoor scene"""
    weight_norm: bool = True
    """Whether to use weight norm for linear layer"""
    use_grid_feature: bool = False
    """Whether to use multi-resolution feature grids"""
    divide_factor: float = 2.0
    """Normalization factor for multi-resolution grids"""
    beta_init: float = 0.1
    """Init learnable beta value for transformation of sdf to density"""
    encoding_type: Literal["hash", "periodic", "tensorf_vm"] = "hash"
    num_levels: int = 16
    """Number of encoding levels"""
    max_res: int = 2048
    """Maximum resolution of the encoding"""
    base_res: int = 16
    """Base resolution of the encoding"""
    log2_hashmap_size: int = 19
    """Size of the hash map"""
    features_per_level: int = 2
    """Number of features per encoding level"""
    use_hash: bool = True
    """Whether to use hash encoding"""
    smoothstep: bool = True
    """Whether to use the smoothstep function"""


# ! 用于构建sdf场
class SDFField(Field):
    """
    A field for Signed Distance Functions (SDF).

    Args:
        config: The configuration for the SDF field.
        aabb: An axis-aligned bounding box for the SDF field.
        num_images: The number of images for embedding appearance.
        use_average_appearance_embedding: Whether to use average appearance embedding. Defaults to False.
        spatial_distortion: The spatial distortion. Defaults to None.
    """

    config: SDFFieldConfig

    def __init__(
        self,
        config: SDFFieldConfig,
        aabb: Float[Tensor, "2 3"],
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        # **************** 1.设置一些列参数 ****************
        self.config = config
        
        # 将aabb的参数设为可训练参数
        # pytorch.Parameter: 将张量包装成一个可以被Module对象注册的可训练参数,可以通过优化器进行更新
        # 初始值为：
        # tensor([[-1., -1., -1.],
        #         [ 1.,  1.,  1.]])
        self.aabb = Parameter(aabb, requires_grad=False)

        # mip-nerf360论文中提到的技术：重参数化
        self.spatial_distortion = spatial_distortion

        # 用来训练的图片数
        self.num_images = num_images

        # 为输入图片的索引创建嵌入层：将高维的索引映射到更低的维度上去，得到输入数据的另一种表现形式
        self.embedding_appearance = Embedding(self.num_images, self.config.appearance_embedding_dim)

        # 是否在推理inference阶段使用average appearance embedding)，默认为false
        self.use_average_appearance_embedding = use_average_appearance_embedding

        # 是否使用多分辨率特征网络（multi-resolution feature grids），默认为false
        self.use_grid_feature = self.config.use_grid_feature

        # 多分辨率网格的归一化因子（Normalization factor for multi-resolution grids）
        # 处理多分辨率网格数据时用于归一化的参数或者技术。
        self.divide_factor = self.config.divide_factor

        # max_res=2048，base_res=16，num_levels=16
        growth_factor = np.exp((np.log(config.max_res) - np.log(config.base_res)) / (config.num_levels - 1))

        # **************** 2. 设置编码器 ****************
        if self.config.encoding_type == "hash":
            # feature encoding
            # * 使用tinycudann的encoding来hash编码
            self.encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid" if config.use_hash else "DenseGrid",
                    "n_levels": config.num_levels,
                    "n_features_per_level": config.features_per_level,
                    "log2_hashmap_size": config.log2_hashmap_size,
                    "base_resolution": config.base_res,
                    "per_level_scale": growth_factor,
                    "interpolation": "Smoothstep" if config.smoothstep else "Linear",
                },
            )

        # we concat inputs position ourselves
        # * 位置编码：Multi-scale sinusoidal encodings
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=6, min_freq_exp=0.0, max_freq_exp=5.0, include_input=False
        )

        # * 方向编码：Multi-scale sinusoidal encodings
        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        )

        # **************** 3.创建geometric网络 ****************
        # 对3d点训练得到sdf和sdf特征
        self.initialize_geo_layers()

        # **************** 4.创建Neus中的方差网络 ****************
        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = LearnedVariance(init_val=self.config.beta_init)

        # **************** 4.创建color网络 ****************
        # * color网络每一层的维度
        # self.config.hidden_dim_color = 256
        # self.config.num_layers_color = 2
        dims = [self.config.hidden_dim_color for _ in range(self.config.num_layers_color)]      # [256,256]
        in_dim = (
            3                                           # point
            + self.direction_encoding.get_out_dim()     # view_direction:27
            + 3                                         # normal
            + self.config.geo_feat_dim                  # feature:   256
            + self.embedding_appearance.get_out_dim()   # embedding: 32
        )
        dims = [in_dim] + dims + [3]                    # 每一层的维度：[321,256,256,3]
        self.num_layers_color = len(dims)               # 一共有4层维度
        # * 根据上面计算的维度来创建4-1层mlp
        for layer in range(0, self.num_layers_color - 1):
            # ** 创建mlp
            out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            # ** 对mlp进行权重归一化处理
            # 通过归一化权重矩阵的行或列来加速收敛和提高模型的泛化性能。
            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            # ** 将每一层都加入到模型的属性当中去
            # setattr(object, name, value)
            # 对象：要设置属性的对象。 属性名：要设置的属性的名称。 值：要为属性设置的新值。
            setattr(self, "clin" + str(layer), lin)

        # **************** 5.设置一系列激活函数，之后forward中会用到 ****************
        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        # 退火率
        self._cos_anneal_ratio = 1.0

        if self.use_grid_feature:
            assert self.spatial_distortion is not None, "spatial distortion must be provided when using grid feature"

    # !! geometric网络，用于生成sdf值和sdf特征
    def initialize_geo_layers(self) -> None:
        """
        Initialize layers for geometric network (sdf)
        """
        # **************** 1.计算各层的维度 ****************
        dims = [self.config.hidden_dim for _ in range(self.config.num_layers)]          # [256，256]
        in_dim = 3 + self.position_encoding.get_out_dim() + self.encoding.n_output_dims # 3+36+32
        dims = [in_dim] + dims + [1 + self.config.geo_feat_dim]                         # [71,256,256,257]
        self.num_layers = len(dims)                                                     # 4
        self.skip_in = [4]

        # **************** 2.根据上面的维度创建每一层mlp ****************
        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
            # * 创建线性层
            lin = nn.Linear(dims[layer], out_dim)

            # * 对线性层的权重和偏置参数进行初始化
            # 合适的初始化可以加速模型的收敛，提高模型的性能，并且有助于避免梯度消失或爆炸等问题。
            if self.config.geometric_init:
                if layer == self.num_layers - 2:
                    if not self.config.inside_outside:
                        # torch.nn.init.normal_初始化线性层lin的权重参数
                        # 从正态分布中采样随机数，然后用这些随机数填充 lin.weight，以初始化该层的权重。
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.0001)
                        # 初始化线性层 lin 的偏置（bias）参数。具体来说，它会将偏置参数 lin.bias 的值设置为指定的常数值，这个常数值为 -self.config.bias。
                        # 权重参数通常需要初始化为非常小的随机数以避免过大的梯度，偏置参数通常初始化为零或者接近零的常数值。
                        torch.nn.init.constant_(lin.bias, -self.config.bias)    # bias = 0.5
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, self.config.bias)
                elif layer == 0:    # 第一层
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif layer in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:               # 普通层
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            # * 对mlp进行权重归一化处理
            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "glin" + str(layer), lin)

    def set_cos_anneal_ratio(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def forward_geonetwork(self, inputs: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch geo_features+1"]:
        """forward the geonetwork"""
        if self.use_grid_feature:
            assert self.spatial_distortion is not None, "spatial distortion must be provided when using grid feature"
            positions = self.spatial_distortion(inputs)
            # map range [-2, 2] to [0, 1]
            positions = (positions + 2.0) / 4.0
            feature = self.encoding(positions)
        else:
            feature = torch.zeros_like(inputs[:, :1].repeat(1, self.encoding.n_output_dims))

        pe = self.position_encoding(inputs)

        inputs = torch.cat((inputs, pe, feature), dim=-1)

        # Pass through layers
        outputs = inputs

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(layer))

            if layer in self.skip_in:
                outputs = torch.cat([outputs, inputs], 1) / np.sqrt(2)

            outputs = lin(outputs)

            if layer < self.num_layers - 2:
                outputs = self.softplus(outputs)
        return outputs

    # !! 得到每一根射线的上采样点的sdf值
    # TODO: fix ... in shape annotations.
    def get_sdf(self, ray_samples: RaySamples) -> Float[Tensor, "num_samples ... 1"]:
        """predict the sdf value for ray samples"""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        hidden_output = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, _ = torch.split(hidden_output, [1, self.config.geo_feat_dim], dim=-1)
        return sdf

    def get_alpha(
        self,
        ray_samples: RaySamples,
        sdf: Optional[Float[Tensor, "num_samples ... 1"]] = None,
        gradients: Optional[Float[Tensor, "num_samples ... 1"]] = None,
    ) -> Float[Tensor, "num_samples ... 1"]:
        """compute alpha from sdf as in NeuS"""
        if sdf is None or gradients is None:
            inputs = ray_samples.frustums.get_start_positions()
            inputs.requires_grad_(True)
            with torch.enable_grad():
                hidden_output = self.forward_geonetwork(inputs)
                sdf, _ = torch.split(hidden_output, [1, self.config.geo_feat_dim], dim=-1)
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self._cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        return alpha

    def get_density(self, ray_samples: RaySamples):
        raise NotImplementedError

    def get_colors(
        self,
        points: Float[Tensor, "*batch 3"],
        directions: Float[Tensor, "*batch 3"],
        normals: Float[Tensor, "*batch 3"],
        geo_features: Float[Tensor, "*batch geo_feat_dim"],
        camera_indices: Tensor,
    ) -> Float[Tensor, "*batch 3"]:
        """compute colors"""
        d = self.direction_encoding(directions)

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
            # set it to zero if don't use it
            if not self.config.use_appearance_embedding:
                embedded_appearance = torch.zeros_like(embedded_appearance)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                )

        hidden_input = torch.cat(
            [
                points,
                d,
                normals,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ],
            dim=-1,
        )

        for layer in range(0, self.num_layers_color - 1):
            lin = getattr(self, "clin" + str(layer))

            hidden_input = lin(hidden_input)

            if layer < self.num_layers_color - 2:
                hidden_input = self.relu(hidden_input)

        rgb = self.sigmoid(hidden_input)

        return rgb

    def get_outputs(
        self,
        ray_samples: RaySamples,
        density_embedding: Optional[Tensor] = None,
        return_alphas: bool = False,
    ) -> Dict[FieldHeadNames, Tensor]:
        """compute output of ray samples"""
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        outputs = {}

        camera_indices = ray_samples.camera_indices.squeeze()

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        inputs.requires_grad_(True)
        with torch.enable_grad():
            hidden_output = self.forward_geonetwork(inputs)
            sdf, geo_feature = torch.split(hidden_output, [1, self.config.geo_feat_dim], dim=-1)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf, inputs=inputs, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        rgb = self.get_colors(inputs, directions_flat, gradients, geo_feature, camera_indices)

        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = torch.nn.functional.normalize(gradients, p=2, dim=-1)

        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMALS: normals,
                FieldHeadNames.GRADIENT: gradients,
            }
        )

        if return_alphas:
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        return outputs

    def forward(
        self, ray_samples: RaySamples, compute_normals: bool = False, return_alphas: bool = False
    ) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
            compute normals: not currently used in this implementation.
            return_alphas: Whether to return alpha values
        """
        field_outputs = self.get_outputs(ray_samples, return_alphas=return_alphas)
        return field_outputs
