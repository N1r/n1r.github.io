---
layout: post
title: post about tensorflow and tone classficiation
date: 2021-08-22 11:12:00-0400
description: some notes.
---


## 今日总结

> tensorflow is *hard* as hell

今日试图 复现 deep learning method for tone classficiation [github](https://github.com/saber5433/ToneNet)
https://github.com/alicex2020/Mandarin-Tone-Classification

## 现阶段 单音节的 声调识别已经相当完善了，主要有论文

1. Tonenet 发表于 interspeech 2019 @Gaoqing
  在单音节的标准发音上，可以达到百分之99以上的识别率
主要思路便是，使用多种参数，例如pitch-related, spectragom, mel-spectragom etc..

作者使用了CASS语料库，即中文标准音节发音语料库，因为商业性暂时搁置，此类语料库也有Tone Perfect dataset from Michigan State University (https://tone.lib.msu.edu/).
初步复现的方法采取类似的模式，提取使用BLCU-sait的小部分语料库中的单音节，总计约10000条不同的语音片段，来自7男3女，

```python
import librosa
# 提取 频谱特征；
mel1 = librosa.feature.melspectrogram(audio1, sr=sample_rate1, n_fft=1024, hop_length=512, n_mels=80, fmin=75, fmax=3700)
plt.imshow(np.log10(mel1 + 1e-10), aspect='auto', cmap=cm.plasma)
plt.show()
print(mel1.shape)

```

```python
import librosa
# 提取 mfcc
mfcc = librosa.feature.mfcc(y=audio1, sr=sample_rate1, n_mfcc=60)
plt.imshow(mfcc, aspect='auto', cmap=cm.viridis)
plt.show()
```

````python
# 使用librosa自带的框架提取 音高 TODO 可变化
# e.g. pyworld / cnn pitch
pitch, mag = librosa.core.piptrack(audio1, sr=sample_rate1, n_fft=512)
plt.imshow(pitch, aspect='auto')
plt.ylim([20,100])
plt.show()
````
部分过程如下。

![mel](/pic/mel.png)
![mfcc](/pic/mfcc.png)
![pitch](/pic/pitch.png)

如图可见为不同方法是声调表征，从直观视觉，mfcc更贴合实际的声调轮廓。

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
