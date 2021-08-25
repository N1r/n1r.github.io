---
layout: post
title: Summary of pre-semester
date: 2021-08-23 11:12:00-0400
description: write-up.
---

# writeup of my vacation
:+1:
## Things to do.
### Learning
- Brain-related knowledge and tools.
-- *eeglab / python-mne*

Resources：

[Python-mne offical documentation](https://mne.tools)

[中文学习资料](https://zhuanlan.zhihu.com/p/128667251) 知乎收集。

[Pybrain 2020: M/EEG analysis with MNE-Python(newest!)](https://github.com/hoechenberger/pybrain_mne/) with video and jupyter notebook! awesome.

eeglab 和 python mne 均是在脑电处理领域较为广泛使用的包，功能丰富，拓展性强。因为自身对于python的熟悉度，便开始试图掌握此工具，为未来的研究做好准备。

>MNE - MEG + EEG ANALYSIS & VISUALIZATION is a Open-source Python package for exploring, visualizing, and analyzing human neurophysiological data: MEG, EEG, sEEG, ECoG, NIRS, and more.

A simple example of time-frequency analysis in mne

使用morlet小波变换
``` python
import os
import numpy as np
import mne
frequencies = np.arange(7, 30, 3)
power = mne.time_frequency.tfr_morlet(aud_epochs, n_cycles=2, return_itc=False,
                                      freqs=frequencies, decim=3)
power.plot(['MEG 1332'])
```
---

# Learning
- Machine learning - related tools and knowledge
-- tensorflow & keras (hard!!)
-- pytroch

Resources：

[Pytorch Deep learning](https://github.com/Atcold/pytorch-Deep-Learning) the greatest course from NYU, with detailed video explaination, slide, and code example.

[Pytorch-tabular](https://github.com/manujosephv/pytorch_tabular) quickly hands on, cause many datas in our research are just sheets.

[Fastai](https://github.com/fastai/fastai) pytorch-based lib for deep learning, easy to use and replicate.

快速掌握基础的机器学习相关理论和方法，对于进一步将其应用于语音研究很有必要，故而在假期中，通过上述的学习资源，补足和完善了自己的理论知识，同时提高了动手实践能力。

Troubleshooting：

深度学习框架的主流使用 包含 Keras/tensorflow-backend 和pytorch,其中 torch的使用较为便捷，只需要对应的CUDA版本安装即可。tensorflow的研究历史较长，大量的更新迭代后，多种公式，调用都有变化，复现前人的代码常常容易出现bug

### bug1 “Could not interpret optimizer identifier” error in Keras
原因是模型(model)和层(layers)使用tensorflow.python.keras(或者tensorflow.keras) API，优化器optimizer（SGD, Adam等）使用keras.optimizers，或者反之。

这是两个不同的keras版本，放在一起无法工作，需要把他们统一到同一版本。优化器和模型必须来自相同的层，相同的keras。

调用过程中 需要使用完全一致的包环境

如
```python
from tensorflow.keras import optimizers
from keras import optimizers
# 不同的调用需要统一

model.complie('') # 此时可以使用名称 而非调用。
model.complie(optimizers = 'Adam')# etc
```

### bug2 "tensorflow很容易出现内存不足的bug"
- [ ] 解决方法
限制内存和内存动态使用都需要tf的1版本才行，使用如下公式可解决

```python
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
```

### bug3 CUDA/cuDnn 在3060下的配置

1. tensorflow 2.6.0
2. CUDA 11.1
3. CuDnn 8.05

保证三者完美兼容才可运行程序。
! 其他可能的原因 maybe 遇到

1. You have cache issues
I regularly work around this error by shutting down my python process, removing the ~/.nv directory (on linux, rm -rf ~/.nv), and restarting the Python process. I don't exactly know why this works. It's probably at least partly related to the second option:

2. You're out of memory
The error can also show up if you run out of graphics card RAM. With an nvidia GPU you can check graphics card memory usage with nvidia-smi. This will give you a readout of how much GPU RAM you have in use (something like 6025MiB /  6086MiB if you're almost at the limit) as well as a list of what processes are using GPU RAM.

reducing your batch size
using a simpler model
using less data
limit TensorFlow GPU memory fraction: For example, the following will make sure TensorFlow uses <= 90% of your RAM:

import keras
import tensorflow as tf
```python
# 配合版本
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
```
3. You have incompatible versions of CUDA, TensorFlow, NVIDIA drivers, etc.
If you've never had similar models working, you're not running out of VRAM and your cache is clean, I'd go back and set up CUDA + TensorFlow using the best available installation guide - I have had the most success with following the instructions at https://www.tensorflow.org/install/gpu rather than those on the NVIDIA / CUDA site. Lambda Stack is also a good way to go.

有用的代码
```python
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```

!TODO
tensorflow 和 pytorch 的代码互相转换。
:+1:
:rocket:




:sparkle:
# Experiment

TODO -> Tone-related experiment by using Deep learning.

Experiment One:

Mandarin tone classfication by using CNN, similar to [**ToneNet** - interspeech2019](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1483.pdf)

Experment Two :

Re-implimentation of [FastPitchFormant -interspeech2021](https://arxiv.org/abs/2106.15123)
**FastPitchFormant: Source-filter based Decomposed Modeling for Speech Synthesis**

>Synthesized speech with a large
pitch-shift scale suffers from audio quality degradation, and
speaker characteristics deformation.

语音合成中，合成的语音往往存在较大的pitch-shift，从而导致音质的下降和话者信息的变现。作者提出了一种feed-forward Transformer based TTS model的方法，用来生成鲁棒且准确的音高韵律特征，提升语音合成的自然度。

To be countinued.... See in next post.
