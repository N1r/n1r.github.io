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
:sparkle:
