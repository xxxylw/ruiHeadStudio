# RuiHeadStudio Environment

当前可训练环境以 `ruiheadstudio-bnbfix` 为准。

新服务器请按这份文档配置：

- [docs/2026-05-22-ruiheadstudio-server-environment-setup.md](docs/2026-05-22-ruiheadstudio-server-environment-setup.md)

相关文件分工：

- `environment.yml`: conda 环境入口，创建 `ruiheadstudio-bnbfix`
- `requirements.txt`: curated pip 核心依赖，不再是本机 `pip freeze`
- `requirements.server.txt`: server 安装入口，目前委托到 `requirements.txt`
- `environment.lock.txt`: 当前能跑通环境的关键版本快照

已验证的关键组合：

```text
Python 3.9.25
PyTorch 2.0.1+cu118
CUDA 11.8 as seen by PyTorch
pytorch3d 0.7.7
xformers 0.0.22.post7
bitsandbytes 0.48.2
```
