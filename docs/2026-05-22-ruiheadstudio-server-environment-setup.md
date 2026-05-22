# RuiHeadStudio Server Environment Setup

这份文档的目标是：让没有配过这个项目的人，在一台新的 Linux 服务器上，尽量按步骤复现当前能训练的环境。

当前已验证能跑通训练的本机环境是：

```text
conda env: /home/rui/miniconda3/envs/ruiheadstudio-bnbfix
python:    3.9.25
torch:     2.0.1+cu118
CUDA:      11.8 as seen by PyTorch
GPU:       NVIDIA GeForce RTX 3090
```

不要直接照抄 `pip freeze`。当前环境是多次修复后形成的，里面有一些本机路径和辅助工具。新机器请按下面步骤来。

## 1. 服务器前提

推荐系统和驱动：

```text
Ubuntu 20.04 / 22.04
NVIDIA driver >= 520
Miniconda / Anaconda
git
git-lfs
```

先确认服务器能看到 GPU：

```bash
nvidia-smi
```

如果 `nvidia-smi` 都不能正常显示 GPU，先修驱动，不要开始装 Python 环境。

## 2. 克隆仓库

```bash
git clone <your-ruiHeadStudio-repo-url> ruiHeadStudio
cd ruiHeadStudio
git lfs install
git lfs pull
```

如果仓库不是从远端重新 clone，而是手动拷贝，也要保证这些目录存在：

```text
gaussiansplatting/submodules/simple-knn/
configs/
threestudio/
gaussiansplatting/
```

## 3. 创建 conda 环境

仓库根目录已有：

- `environment.yml`
- `requirements.txt`
- `requirements.server.txt`

执行：

```bash
conda env create -f environment.yml
conda activate ruiheadstudio-bnbfix
```

如果下载太慢，可以先配置 conda/pip 镜像。不要随意升级 Python、Torch 或 CUDA 版本；这个项目的二进制依赖对版本很敏感。

当前目标组合是：

```text
Python 3.9
PyTorch 2.0.1
torchvision 0.15.2
torchaudio 2.0.2
CUDA 11.8
```

## 4. 安装 PyTorch3D 和 xformers

这两个包要和 PyTorch/CUDA 精确匹配。环境创建后执行：

```bash
conda activate ruiheadstudio-bnbfix

conda install -y \
  https://anaconda.org/pytorch3d/pytorch3d/0.7.7/download/linux-64/pytorch3d-0.7.7-py39_cu118_pyt201.tar.bz2

conda install -y \
  https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py39_cu11.8.0_pyt2.0.1.tar.bz2
```

如果服务器访问 Anaconda 慢，可以用等价镜像，但不要换版本。

## 5. 安装 diff-gaussian-rasterization

当前训练需要 `diff_gaussian_rasterization`。它不是主仓库 tracked 文件，需要在仓库根目录单独准备：

```bash
cd /path/to/ruiHeadStudio
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization.git
pip install ./diff-gaussian-rasterization
```

如果编译找不到 CUDA，先设置：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
```

然后重新执行 `pip install ./diff-gaussian-rasterization`。

## 6. 确认 simple-knn 已安装

`environment.yml` 会安装：

```text
./gaussiansplatting/submodules/simple-knn
```

如果后面 import 失败，手动补一次：

```bash
pip install ./gaussiansplatting/submodules/simple-knn
```

## 7. 准备模型和数据

环境装好不等于能训练。下面这些文件也必须存在。

### 7.1 FLAME

当前配置：

```yaml
data:
  flame_path: "./ckpts/FLAME-2000"
```

至少需要：

```text
ckpts/FLAME-2000/FLAME_GENERIC.pkl
```

当前本机是 symlink：

```text
ckpts/FLAME-2000/FLAME_GENERIC.pkl -> external/flame-head-tracker/models/FLAME2020/generic_model.pkl
```

新服务器可以用 symlink，也可以直接复制真实文件。FLAME 官方文件需要去 FLAME 官网按许可下载。

还需要这些辅助文件：

```text
ckpts/flame2facemesh.npy
ckpts/flame_dynamic_embedding.npy
ckpts/flame_static_embedding.pkl
ckpts/mediapipe_landmark_embedding.npz
ckpts/mica_mediapipe_landmark_embedding.npz
ckpts/FLAME-2000/flame2facemesh.npy
ckpts/FLAME-2000/flame_dynamic_embedding.npy
ckpts/FLAME-2000/flame_static_embedding.pkl
ckpts/FLAME-2000/mediapipe_landmark_embedding.npz
ckpts/FLAME-2000/mica_mediapipe_landmark_embedding.npz
```

`ckpts/FLAME-2000/` 里的文件可以是指向 `ckpts/` 的 symlink。

### 7.2 TalkSHOW / TalkVid pose 数据

当前 `configs/headstudio.yaml` 使用这些路径：

```text
talkshow/collection/chemistry/2nd_Order_Rate_Laws-6BZb96mqmbg__68891-00_01_40-00_01_46_exp.npy
talkshow/collection/talkvid/talkvid_tracker_exp.npy
talkshow/ExpressiveWholeBodyDatasetReleaseV1.0/chemistry/2nd_Order_Rate_Laws-6BZb96mqmbg.mp4/68891-00_01_40-00_01_46/68891-00_01_40-00_01_46.pkl
```

缺一个都会在训练初始化或验证阶段失败。

### 7.3 Hugging Face 模型

首次训练会用到：

```text
stablediffusionapi/realistic-vision-51
lllyasviel/control_v11p_sd15_openpose
lllyasviel/control_v11f1p_sd15_depth
```

服务器能访问 Hugging Face 时，第一次训练会自动下载。

如果服务器不能访问 Hugging Face，要提前把模型放进 Hugging Face cache，或者设置代理/镜像。我们本机 Thor 训练就是靠本地 cache 跑通的，并设置了：

```bash
export HF_HUB_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

只有在本地 cache 已经完整时，才打开这三个离线变量。

## 8. 环境验证

先做基础验证：

```bash
conda activate ruiheadstudio-bnbfix
export MPLCONFIGDIR="$PWD/.cache/matplotlib"

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

期望类似：

```text
2.0.1+cu118 11.8 True
NVIDIA GeForce RTX 3090
```

再验证关键包：

```bash
python -c "import diffusers, transformers, mediapipe, cv2, pytorch3d, xformers, nerfacc, nvdiffrast.torch, tinycudann, diff_gaussian_rasterization, simple_knn._C, smplx, controlnet_aux, bitsandbytes; print('imports ok')"
```

如果这里失败，按报错优先查：

```text
pytorch3d / xformers: 版本必须匹配 torch 2.0.1 + cu118
tinycudann / nvdiffrast / diff_gaussian_rasterization / simple_knn: 多半是 CUDA_HOME 或编译工具链问题
mediapipe / opencv: 多半是系统 OpenGL/GLib 依赖问题
```

最后验证项目入口：

```bash
python launch.py --help
```

## 9. 最小训练命令

在仓库根目录执行：

```bash
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR="$PWD/.cache/matplotlib"

python launch.py \
  --config configs/headstudio.yaml --train \
  system.prompt_processor.prompt="a DSLR portrait of Thor in Marvel, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
  system.guidance.use_nfsd=True \
  system.max_grad=0.001 \
  trainer.max_steps=10000 \
  system.area_relax=True
```

如果 Hugging Face cache 已经提前准备好，并且服务器不能联网，再加：

```bash
export HF_HUB_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 10. 常见坑

### 10.1 不要用 base Python

必须确认：

```bash
which python
```

应该指向：

```text
.../miniconda3/envs/ruiheadstudio-bnbfix/bin/python
```

### 10.2 不要照旧 requirements.txt 的 pip freeze 装

旧 `requirements.txt` 曾经是 `pip freeze`，包含本机绝对路径和无关工具。现在已经改成 curated core requirements。

### 10.3 sandbox 里可能看不到 CUDA

我们在 Codex sandbox 里跑 Python 时，`torch.cuda.is_available()` 可能是 `False`。真实训练需要在普通 shell、tmux 或非 sandbox 环境里跑。

### 10.4 Matplotlib cache

服务器上建议固定设置：

```bash
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
```

否则多进程或无权限 home 目录下可能反复报 cache warning。

### 10.5 ControlNet 下载失败

如果报：

```text
Can't load config for 'lllyasviel/control_v11p_sd15_openpose'
```

先确认是网络问题还是 cache 缺文件。cache 完整时可以开离线变量；cache 不完整时开离线变量只会更早失败。
