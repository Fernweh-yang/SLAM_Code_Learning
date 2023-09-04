import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

# 数据加载的脚本，虽然可以导入不同的数据类型，但是最后送入到网络的都是同一类型
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)   # 随机渲染某些光线，不可能所有光线都渲染，否则计算量太大
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

# 实现小批量处理数据
def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    # ******************* 5.2.1.1 位置编码 *******************
    embedded = embed_fn(inputs_flat)   

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    # ******************* 5.2.1.2 网络学习 *******************
    # 以更小的patch-netchunk送进网络跑前向，比如chunk=20，就1-20跑一波，21-40跑一波。。。
    outputs_flat = batchify(fn, netchunk)(embedded)
    # 将list重新拼接reshape为[1024,64,64] 4:rgb alpha
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    在更小的batch上进行渲染,避免超出内存
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        # ******************* 5.2 minibatch 渲染 *******************
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    # 将所有的结果拼接后一起返回
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H 图像高度: int. Height of image in pixels. 

      W 图像宽度: int. Width of image in pixels.  

      focal 针孔相机焦距: float. Focal length of pinhole camera. 

      chunk 同步处理的最多光线数: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.

      rays 每个batch的ray的原点和方向: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.

      c2w 相机到世界坐标系的旋转矩阵: array of shape [3, 4]. Camera-to-world transformation matrix.

      ndc NDC坐标: bool. If True, represent ray origin, direction in NDC coordinates.

      near 光线最近距离: float or array of shape [batch_size]. Nearest distance for a ray.
      
      far 光线最远距离: float or array of shape [batch_size]. Farthest distance for a ray.
      
      use_viewdirs 是否使用方向信息: bool. If True, use viewing direction of a point in space in model.
      
      c2w_staticcam 变换矩阵: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
   
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays. 预测的rgb图
      disp_map: [batch_size]. Disparity map. Inverse of depth. 视差图
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray. 深度图
      extras: dict with everything returned by render_rays(). 其他
    """
    # ******************* 5.0 数据准备 *******************
    if c2w is not None:
        # special case to render full image
        # 如果c2w不为空，重新计算下全图的ray
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        # 否则使用前面计算好的ray
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d   # 视角就是光线的方向
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            # 特殊的情况下，会重新算下光线（用于做实验）
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)    #归一化得到单位向量
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()  #将方向矩阵shape调整为[n_batch,3] eg.[1024,3]

    sh = rays_d.shape # [..., 3] eg.[1024,3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    # 构建batch
    rays_o = torch.reshape(rays_o, [-1,3]).float() #光线起始点[1024,3]
    rays_d = torch.reshape(rays_d, [-1,3]).float() #光线方向[1024,3]

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])   #光线的最近和最远深度均为[1024,1]和光线一样的个数
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # 将上面4个量拼接起来，就是[1024,8]
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)# 再将上面归一化后得到的光线方向单位向量拼接进来,[1024,11]

    # Render and reshape
    # ******************* 5.1 渲染 *******************
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    # ******************* 3.1 位置编码 *******************
    # x,y,z进行位置编码。输入是xyz三维，输出是input_ch=63维的高维空间。embed_fn就是这个转高维怎么算的一个方法
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    # 下面是给方向进行位置编码
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4 # 输出通道数，如果有设置fine在光线上额外采样就是5通道，否则4通道
    skips = [4]
    # ******************* 3.2 NeRF模型初始化 *******************
    # coarse网络初始化
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters()) #取模型参数

    model_fine = None
    # fine网络初始化，就深度和通道数和coarse网络可以不一样
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())
    
    # ******************* 3.3 模型批量处理数据函数 *******************
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # ******************* 3.4 优化器定义 *******************
    # 使用的adam自适应矩估计优化算法
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    # 加载已有模型参数（如果训练中断了，就可以在原有基础上继续训练）
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################
    # 训练需要的参数
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False   # 木有扰动
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # 见论文公式3中alpha公式定义
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    # 同样论文公式3定义：delta_i = t_{i+1}-t_i
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples] alpha值计算
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1] # 权重计算公式见论文公式3
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3] rgb值计算

    depth_map = torch.sum(weights * z_vals, -1) # 深度图计算d=\sum w_iz_i
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)) # 视差图计算
    acc_map = torch.sum(weights, -1)    # 权重求和

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.体素渲染
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin原点, ray direction, min
        dist, max dist, and unit-magnitude viewing direction单位方向.

      network_fn: function. Model for predicting RGB and density at each point
        in space.NeRF网络,用来预测空间中每个点rgb和不透明度

      network_query_fn: function used for passing queries查询 to network_fn.

      N_samples: int. Number of different times to sample along each ray.粗糙采样点数

      retraw: bool. If True, include model's raw, unprocessed predictions.真就返回无压缩数据

      lindisp: bool. If True, sample linearly in inverse depth rather than in depth. 以反深度进行线性采样

      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time. 波动

      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.精细网络增加的采样点数

      network_fine: "fine" network with same spec as network_fn.

      white_bkgd: bool. If True, assume a white background.

      raw_noise_std: ...噪声

      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model. ray的rgb值
      disp_map: [num_rays]. Disparity map. 1 / depth.   视差
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.累积不透明度
      raw: [num_rays, num_samples, 4]. Raw predictions from model. 原始raw数据
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each sample.标准差
    """
    # 下面是取各种数据出来
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1] 每个[N_rays,1]
    # 粗糙网络采样
    t_vals = torch.linspace(0., 1., steps=N_samples) # 在0-1之间线性采样N_samples个值
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples]) # 给N_rays曲线每条都在near-far之间取N_samples个采样 


    # 加扰动
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])  # 均值[N_rays,N_samples-1]
        upper = torch.cat([mids, z_vals[...,-1:]], -1)  # [N_rays,N_samples]
        lower = torch.cat([z_vals[...,:1], mids], -1)   # [N_rays,N_samples]
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)   # 在这些间隔点中分层采样点

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand
    # 得到每个采样点的3D坐标:原点o+光线方向d*采样点深度
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    # ******************* 5.2.1 coarse网络推断 *******************
    # 送进粗糙网络进行预测
    raw = network_query_fn(pts, viewdirs, network_fn)   #[ray_batch,N_samples,4]  eg.[1024,64,4]
    # ******************* 5.2.2 coarse raw数据转换为rgb,视差，不透明度，权重和深度 *******************
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    # 精细网络采样
    if N_importance > 0:
        
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map # 利用上面coarse网络计算的各个值

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        # ******************* 5.2.3 精细网络Pdf采样 *******************
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        
        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        # ******************* 5.2.4 精细网络推断  *******************
        # network_query_fn()定义于3.3
        raw = network_query_fn(pts, viewdirs, run_fn)
        # ******************* 5.2.5 finw raw数据转换为rgb,视差，不透明度，权重和深度   *******************
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')    # 参数配置文件config.txt，见configs文件夹，不同数据集不同的训练参数
    parser.add_argument("--expname", type=str, 
                        help='experiment name')     # 实验名称
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')   # 训练模型和渲染结果保存路径
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')# 输入数据路径一定要给对

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')   #粗糙网络深度（层数）
    parser.add_argument("--netwidth", type=int, default=256,    
                        help='channels per layer')  #粗糙网络每层通道数
    parser.add_argument("--netdepth_fine", type=int, default=8,    
                        help='layers in fine network')#fine网络层数
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')#fine网络每层通道数
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')#batch size（一次处理多少组数据）这里就是光束数量
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')  #学习率
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')#指数学习衰减
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')# 并行处理多少光线
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')#一次只从一张图像中获取随机数量光线
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')#不从保存的ckpt(checkpoint)加载权重
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')#为粗糙网络加载指定权重

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')#粗糙网络每条光线采样的点数
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')#fine网络每条光线额外的采样点
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')#是否有抖动
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')#默认使用5D，即包含方向信息，而不是3D（x，y，z）
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')#是否使用位置编码
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')#位置编码的参数：位置编码最大频率为log2
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')#位置编码的参数：最大分辨率为log2
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')#噪声方差，训练网络时可加

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')#仅渲染，加载权重和渲染pose路径
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')#渲染测试集（有gt）而非pose路径，用于和gt比较，检测网络精度。
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')#下采样因子

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels') #数据类型，默认式llff
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    ##下面是不同数据类型的参数
    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')#图像下采样率（万一原图像素太高）
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')#默认不使用标准化坐标系
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')#默认在视差上均匀采样而非深度图
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')#360度场景
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')#取1/N测试数据来测试

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')#每隔多少次终端输出下标准
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')#每个多少次保存下图片
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')#每隔多少次保存下训练的权重
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')#每隔多少次测试下数据，并保存下来
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')#每个多少次渲染个视频，并保存下来

    return parser


def train():
    # ******************* 1.参数设置 *******************
    parser = config_parser()
    args = parser.parse_args()

    # ******************* 2.加载数据 *******************
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]    # hwf是pose[n,3,5]最后一列即第五列的那3个值hight,width,focal
        poses = poses[:,:3,:4]  # 位姿就是3x3的R和第四列的t
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        # 用load_llff_data（）传回来的i_test来生成测试的数据id
        if not isinstance(i_test, list):
            i_test = [i_test]
        # 或者用llfhold作为间隔，每隔load_llff_data取一个照片作为测试集
        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        # 剩下的都是训练集，给出这些照片的id
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])  

        # 论文里公式4(1) 的t_n,t_f
        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    # 将内参参数转为正确的类型内参矩阵
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        #主点坐标默认是图片中心，所以取w,h的一半
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])# fx,fy,cx,cy

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    # 创建log路径，保存训练用的所有参数到args.txt,复制config参数并保存
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # ******************* 3.网络构建 *******************
    # 模型构建：得到训练参数，测试参数，起始step和优化器
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    # 添加最近最远深度
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    # 仅渲染：就只调用渲染函数
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # ******************* 4.构建raybatch tensor,生成光线数据 *******************
    # Prepare raybatch tensor if batching random rays
    # 如果批量处理光线ray，就准备raybatch tensor
    N_rand = args.N_rand    # 如果有批处理就从一批照片中随机取imageN*N_rand个光束送入网络进行训练，否则就只从一张图片中选N_rand个光束
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3] eg.[20,3,378,504,3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only取出训练集 eg.[20,378,504,3,3]
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        # 打乱所有光线排布，或称洗牌 使其不按原来的顺序存储，这样训练更有鲁棒性
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)   # 训练图像id
    print('TEST views are', i_test)     # 测试图像id
    print('VAL views are', i_val)       # 验证图像id

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    # 迭代计算20万次
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # 每次从所有图像的Ray中抽取N_rand个ray，每遍历一边就打乱顺序，然后作为训练数据
            # Random over all images
            # 这里的3x3指的是：rgb值，光线起始点，光线方向
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)    #[3,B,3]
            # 光线起始点和方向：batch_rays =[2,B,3] 光线rgb值：target_s=[B,3]
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        #不推荐使用不使用Batch来训练
        else:
            # Random from one image
            # 每次随机抽取一张图像，抽取一个batch的ray作为训练数据
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)


        #####  Core optimization loop  #####
        # ******************* 5. 渲染过程*******************
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        # ******************* 6. 计算loss *******************
        img_loss = img2mse(rgb, target_s)   # 计算L2 loss
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)   # j计算性能指标psnr

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward() # 反向传播
        optimizer.step()# 更新参数

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        # 调整学习率
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        # 保存log
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            # 视屏渲染
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
