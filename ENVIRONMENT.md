# RuiHeadStudio Server Environment Guide

这份文档的目标不是解释“理论上需要哪些包”，而是让后来的人能在一台新的 Linux 服务器上把这个仓库实际跑起来。

当前仓库里和环境相关的文件分工如下：

- [`requirements.txt`](/home/huangqirui/Projects/ruiHeadStudio/requirements.txt)
  这是从本机 `ruiheadstudio` 环境直接 `pip freeze` 出来的快照，包含本机路径和历史包，不适合直接在新机器上安装。
- [`requirements.server.txt`](/home/huangqirui/Projects/ruiHeadStudio/requirements.server.txt)
  这是按当前可运行环境整理出来的、适合服务器安装的 pip 依赖清单。
- [`environment.yml`](/home/huangqirui/Projects/ruiHeadStudio/environment.yml)
  这是服务器重建主环境时应该优先使用的 conda 环境文件。

## 1. 服务器前提

推荐前提：

- Ubuntu 20.04 / 22.04
- NVIDIA GPU
- 可用的 NVIDIA 驱动
- 已安装 Conda 或 Miniconda

这个项目当前建议固定在下面这组版本上：

- Python 3.9
- CUDA 11.8
- PyTorch 2.0.1
- torchvision 0.15.2
- torchaudio 2.0.2

如果后面有人想升级 Python、CUDA 或 PyTorch，先默认会破 `pytorch3d`、`xformers`、`tiny-cuda-nn`、`simple-knn`、`diff-gaussian-rasterization` 这些二进制/编译依赖。

## 2. 主环境安装顺序

以下命令都在仓库根目录执行。

### 2.1 创建基础 conda 环境

```bash
conda env create -f environment.yml
conda activate ruiheadstudio
```

`environment.yml` 会先装好：

- Python 3.9
- CUDA 11.8 toolkit
- PyTorch 2.0.1
- 编译工具链
- 大部分 Python 运行依赖

### 2.2 安装必须的二进制包

`pytorch3d` 和 `xformers` 这两类包对 PyTorch/CUDA 组合比较敏感，按当前环境验证过的版本安装最稳：

```bash
conda install -y https://anaconda.org/pytorch3d/pytorch3d/0.7.7/download/linux-64/pytorch3d-0.7.7-py39_cu118_pyt201.tar.bz2
conda install -y https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py39_cu11.8.0_pyt2.0.1.tar.bz2
```

如果你想把日志/界面相关包也完全对齐当前环境，可以再执行：

```bash
conda install -y https://anaconda.org/conda-forge/pytorch-lightning/2.3.3/download/noarch/pytorch-lightning-2.3.3-pyhd8ed1ab_0.conda
conda install -y https://anaconda.org/conda-forge/lightning/2.3.3/download/noarch/lightning-2.3.3-pyhd8ed1ab_0.conda
```

### 2.3 安装 diff-gaussian-rasterization

这个包不在仓库里，必须单独拉下来装：

```bash
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
```

如果你后面遇到 rasterization 的非法内存访问问题，参考仓库 README 里的说明，用 `-fno-gnu-unique` 重新编译。

### 2.4 验证环境

至少执行下面几条：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import pytorch3d, xformers, mediapipe, diffusers; print('imports ok')"
python launch.py --help
```

如果第二条失败，优先排查：

- `pytorch3d`
- `xformers`
- `tinycudann`
- `simple_knn`
- `diff-gaussian-rasterization`

## 3. 需要额外下载的模型和数据

这个项目不是“装完环境就能跑”，下面这些资源必须手动准备。

### 3.1 FLAME 模型

来源：

- 官网: `https://flame.is.tue.mpg.de`

操作：

1. 去官网注册并同意许可
2. 下载 `FLAME 2020`
3. 把文件放到 `ckpts/FLAME-2000/`

当前代码和 `smplx` 的读取方式决定了至少要有：

- `ckpts/FLAME-2000/FLAME_GENERIC.pkl`

如果后面把配置里的 `gender` 改成 `male` 或 `female`，还需要：

- `ckpts/FLAME-2000/FLAME_MALE.pkl`
- `ckpts/FLAME-2000/FLAME_FEMALE.pkl`

注意：

- 当前仓库里已经有 `female_model.pkl`、`male_model.pkl` 这类文件名，但 `smplx.FLAME` 实际按 `FLAME_{GENDER}.pkl` 取文件。
- 当前默认配置 [`configs/headstudio.yaml`](/home/huangqirui/Projects/ruiHeadStudio/configs/headstudio.yaml) 里 `gender: 'generic'`，所以服务器最少先保证 `FLAME_GENERIC.pkl` 正常。

### 3.2 HeadStudio 自带 ckpts 和 TalkSHOW 数据

来源：

- 仓库 [`README.md`](/home/huangqirui/Projects/ruiHeadStudio/README.md) 里的百度网盘链接
- 链接文字：`https://pan.baidu.com/s/1BdFmOMNT4gWhqUKFuZWx9A?pwd=pkwj`
- 提取码：`pkwj`

这份网盘包里至少要把下面几类内容解压出来。

#### A. `ckpts/ControlNet-Mediapipe/`

至少需要：

- `ckpts/ControlNet-Mediapipe/flame2facemesh.npy`
- `ckpts/ControlNet-Mediapipe/mediapipe_landmark_embedding.npz`

这两个文件会被代码直接读取：

- [`threestudio/utils/head_v2.py:185`](/home/huangqirui/Projects/ruiHeadStudio/threestudio/utils/head_v2.py#L185)
- [`threestudio/utils/head_v2.py:194`](/home/huangqirui/Projects/ruiHeadStudio/threestudio/utils/head_v2.py#L194)

README 里写的是 `flame2facemsh.npy`，这里有拼写差异。当前仓库实际在用的是：

- `flame2facemesh.npy`

按代码里的名字放，不要按 README 里的错拼写放。

#### B. `ckpts/FLAME-2000/`

除 FLAME 官方 pkl 外，网盘包里还需要这些辅助文件：

- `ckpts/FLAME-2000/flame2facemesh.npy`
- `ckpts/FLAME-2000/flame_static_embedding.pkl`
- `ckpts/FLAME-2000/flame_dynamic_embedding.npy`
- `ckpts/FLAME-2000/mediapipe_landmark_embedding.npz`
- `ckpts/FLAME-2000/mica_mediapipe_landmark_embedding.npz`

注意这里的文件名是当前仓库实际目录里的名字：

- `flame_static_embedding.pkl`
- `flame_dynamic_embedding.npy`

不是 README 里写的复数版 `*_embeddings.*`。

#### C. TalkSHOW 训练/验证数据

当前配置文件里直接写死了两个路径：

- [`configs/headstudio.yaml:41`](/home/huangqirui/Projects/ruiHeadStudio/configs/headstudio.yaml#L41)
- [`configs/headstudio.yaml:42`](/home/huangqirui/Projects/ruiHeadStudio/configs/headstudio.yaml#L42)

也就是：

- `./collection/ruiheadstudio/flame_collections/talkshow/project_converted_exp.npy`
- `./talkshow/ExpressiveWholeBodyDatasetReleaseV1.0/chemistry/2nd_Order_Rate_Laws-6BZb96mqmbg.mp4/68891-00_01_40-00_01_46/68891-00_01_40-00_01_46.pkl`

所以你从百度网盘或 TalkSHOW 数据包解压后，至少要把这两个目标路径补齐。

如果数据放在别的位置，就修改：

- [`configs/headstudio.yaml`](/home/huangqirui/Projects/ruiHeadStudio/configs/headstudio.yaml)

中的：

- `talkshow_train_path`
- `talkshow_val_path`
- `flame_path`

### 3.3 HuggingFace 在线模型

首次训练时，代码还会自动拉取下面这些 HuggingFace 模型：

- `stablediffusionapi/realistic-vision-51`
- `lllyasviel/control_v11p_sd15_openpose`
- `lllyasviel/control_v11f1p_sd15_depth`

相关代码位置：

- [`configs/headstudio.yaml:71`](/home/huangqirui/Projects/ruiHeadStudio/configs/headstudio.yaml#L71)
- [`configs/headstudio.yaml:82`](/home/huangqirui/Projects/ruiHeadStudio/configs/headstudio.yaml#L82)
- [`threestudio/models/guidance/controlnet_guidance.py:76`](/home/huangqirui/Projects/ruiHeadStudio/threestudio/models/guidance/controlnet_guidance.py#L76)
- [`threestudio/models/guidance/controlnet_guidance.py:81`](/home/huangqirui/Projects/ruiHeadStudio/threestudio/models/guidance/controlnet_guidance.py#L81)

服务器如果不能直接访问 HuggingFace，需要提前处理：

- 配代理
- 配镜像
- 提前把模型缓存到本机

否则环境装好了，第一次训练仍然会卡在下载模型阶段。

## 4. 建议的目录结果

最少整理成下面这样：

```text
ruiHeadStudio/
├── ckpts/
│   ├── ControlNet-Mediapipe/
│   │   ├── flame2facemesh.npy
│   │   └── mediapipe_landmark_embedding.npz
│   └── FLAME-2000/
│       ├── FLAME_GENERIC.pkl
│       ├── FLAME_MALE.pkl
│       ├── FLAME_FEMALE.pkl
│       ├── flame2facemesh.npy
│       ├── flame_static_embedding.pkl
│       ├── flame_dynamic_embedding.npy
│       ├── mediapipe_landmark_embedding.npz
│       └── mica_mediapipe_landmark_embedding.npz
├── talkshow/
│   ├── collection/
│   │   └── chemistry_exp.npy
│   └── ExpressiveWholeBodyDatasetReleaseV1.0/
│       └── ...
├── environment.yml
├── requirements.server.txt
└── launch.py
```

## 5. 训练前最后检查

在正式训练前，建议按下面顺序确认。

1. `conda activate ruiheadstudio`
2. `python -c "import torch; print(torch.cuda.is_available())"`
3. `ls ckpts/ControlNet-Mediapipe`
4. `ls ckpts/FLAME-2000`
5. 确认 [`configs/headstudio.yaml`](/home/huangqirui/Projects/ruiHeadStudio/configs/headstudio.yaml) 里的 `flame_path`、`talkshow_train_path`、`talkshow_val_path` 都指向真实存在的文件
6. `python launch.py --help`

## 6. 启动命令

最常用的是：

```bash
python launch.py \
  --config configs/headstudio.yaml \
  --train \
  system.prompt_processor.prompt='a DSLR portrait of Joker in DC, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
  system.guidance.use_nfsd=True \
  system.max_grad=0.001 \
  system.area_relax=True
```

更多样例命令见：

- [`scripts/headstudio.sh`](/home/huangqirui/Projects/ruiHeadStudio/scripts/headstudio.sh)

## 7. TalkSHOW 第二套环境

如果只是跑 HeadStudio 主训练，这一段可以先不配。

如果你要跑：

- 音频驱动动画
- TalkSHOW 生成 FLAME 序列

建议单独创建第二套环境，不要和 `ruiheadstudio` 混在一起：

```bash
conda create -n env_SHOW python=3.7
conda activate env_SHOW
pip install "torch~=1.13.1" "torchaudio~=0.13.1" "torchvision~=0.14.1"
```

然后再按 TalkSHOW 官方仓库安装它自己的剩余依赖。

## 8. 交接时该看什么

如果后面的人只看一个文件，就看：

- [`ENVIRONMENT.md`](/home/huangqirui/Projects/ruiHeadStudio/ENVIRONMENT.md)

如果后面的人只执行一个环境文件，就用：

- [`environment.yml`](/home/huangqirui/Projects/ruiHeadStudio/environment.yml)

如果后面的人看到 [`requirements.txt`](/home/huangqirui/Projects/ruiHeadStudio/requirements.txt)，要知道那是本机快照，不是服务器安装入口。
