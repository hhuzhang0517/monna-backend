# Monna AI模型目录

这个目录包含Monna应用使用的AI模型。模型文件体积较大，不包含在代码仓库中，而是在运行时自动下载。

## 目录结构

```
models/
├── facechain/     # FaceChain项目代码和模型（由python-worker自动下载）
└── other-models/  # 其他可能需要的模型
```

## FaceChain模型

[FaceChain](https://github.com/modelscope/facechain)是一个用于生成AI人像的项目，它在第一次运行时会自动下载和配置。主要模型包括：

- 人脸识别模型
- 扩散模型
- 风格模型

这些模型会在`python-worker`首次运行时自动下载到`models/facechain`目录下。

## 注意事项

- 首次下载模型可能需要较长时间，请保持网络连接
- 模型文件较大，请确保有足够的磁盘空间
- 如果下载失败，可以手动克隆FaceChain仓库到`models/facechain`目录 