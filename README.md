# AudioStory: Generating Long-Form Narrative Audio with Large Language Models

**[Yuxin Guo<sup>1,2</sup>](https://scholar.google.com/citations?user=x_0spxgAAAAJ&hl=en), 
[Teng Wang<sup>2,&#9993;</sup>](http://ttengwang.com/), 
[Yuying Ge<sup>2</sup>](https://geyuying.github.io/), 
[Shijie Ma<sup>1,2</sup>](https://mashijie1028.github.io/), 
[Yixiao Ge<sup>2</sup>](https://geyixiao.com/), 
[Wei Zou<sup>1</sup>](https://people.ucas.ac.cn/~zouwei),
[Ying Shan<sup>2</sup>](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)**
<br>
<sup>1</sup>Institute of Automation, CAS
<sup>2</sup>ARC Lab, Tencent PCG
<br>



## 📖 Release

[2025/8/28] 🔥🔥 We release the inference code!

[2025/8/28] 🔥🔥 We release our demo videos!



## 🔎 Introduction

![audiostory](audiostory.png)

✨ **TL; DR: We propose a model for long-form narrative audio generation built upon a unified understanding–generation framework, capable of handling video dubbing, audio continuation, and long-form narrative audio synthesis.**

Recent advances in text-to-audio (TTA) generation excel at synthesizing short audio clips but struggle with long-form narrative audio, which requires temporal coherence and compositional reasoning. To address this gap, we propose AudioStory, a unified framework that integrates large language models (LLMs) with TTA systems to generate structured, long-form audio narratives. AudioStory possesses strong instruction-following reasoning generation capabilities. It employs LLMs to decompose complex narrative queries into temporally ordered sub-tasks with contextual cues, enabling coherent scene transitions and emotional tone consistency. AudioStory has two appealing features: 

1) Decoupled bridging mechanism: AudioStory disentangles LLM-diffuser collaboration into two specialized components—a bridging query for intra-event semantic alignment and a consistency query for cross-event coherence preservation.
2) End-to-end training: By unifying instruction comprehension and audio generation within a single end-to-end framework, AudioStory eliminates the need for modular training pipelines while enhancing synergy between components. 
    Furthermore, we establish a benchmark AudioStory-10K, encompassing diverse domains such as animated soundscapes and natural sound narratives.

Extensive experiments show the superiority of AudioStory on both single-audio generation and narrative audio generation, surpassing prior TTA baselines in both instruction-following ability and audio fidelity.



## ⭐ Demos

### 1. Video Dubbing (Tom & Jerry style)
> Dubbing is achieved using AudioStory (trained on Tom & Jerry) with visual captions extracted from videos.

<table class="center">
  <td><video src="https://github.com/user-attachments/assets/f06b5999-6649-44d3-af38-63fdcecd833c"></video></td>
  <td><video src="https://github.com/user-attachments/assets/17727c2a-bfea-4252-9aa8-48fc9ac33500"></video></td>
  <td><video src="https://github.com/user-attachments/assets/09589d82-62c9-47a6-838a-5a62319f35e2"></video></td>
  <tr>
</table >


### 2. Cross-domain Video Dubbing (Tom & Jerry style)

<table class="center">
    <td><video src="https://github.com/user-attachments/assets/e62d0c09-cdf0-4e51-b550-0a2c23f8d68d"></video></td>
    <td><video src="https://github.com/user-attachments/assets/736d22ca-6636-4ef0-99f3-768e4dfb112a"></video></td>
    <td><video src="https://github.com/user-attachments/assets/f2f7c94c-7f72-4cc0-8edc-290910980b04"></video></td>
  <tr>
  <td><video src="https://github.com/user-attachments/assets/d3e58dd4-31ae-4e32-aef1-03f1e649cb0c"></video></td>
  <td><video src="https://github.com/user-attachments/assets/4f68199f-e48a-4be7-b6dc-1acb8d377a6e"></video></td>
  <td><video src="https://github.com/user-attachments/assets/062236c3-1d26-4622-b843-cc0cd0c58053"></video></td>
	<tr>
  <td><video src="https://github.com/user-attachments/assets/8931f428-dd4d-430f-9927-068f2912dd36"></video></td>
  <td><video src="https://github.com/user-attachments/assets/ab7e46d5-f42c-472e-b66e-df786b658210"></video></td>      
  <td><video src="https://github.com/user-attachments/assets/9a0998ad-b5a4-42ac-bdaf-ceaf796fc586"></video></td>
  <tr>
</table >



### 3. Text-to-Long Audio (Natural sound)

<table class="center">
  <td style="text-align:center;" width="480">Instruction: "Develop a comprehensive audio that fully represents jake shimabukuro performs a complex ukulele piece in a studio, receives applause, and discusses his career in an interview. The total duration is 49.9 seconds."</td>
  <td><video src="https://github.com/user-attachments/assets/461e8a34-4217-454e-87b3-e4285f36ec43"></video></td>
	<tr>
  <td style="text-align:center;" width="480">Instruction: "Develop a comprehensive audio that fully represents a fire truck leaves the station with sirens blaring, signaling an emergency response, and drives away. The total duration is 35.1 seconds."</td>
  <td><video src="https://github.com/user-attachments/assets/aac0243f-5d12-480e-9850-a7f6720e4f9c"></video></td>
	<tr>
     <td style="text-align:center;" width="480">Instruction: "Understand the input audio, infer the subsequent events, and generate the continued audio of the coach giving basketball lessons to the players. The total duration is 36.6 seconds."</td>    
    <td><video src="https://github.com/user-attachments/assets/c4ed306a-651e-43d6-aeea-ee159542418a"></video></td>
	<tr>
</table >




## 🔎 Methods

![audiostory_framework](audiostory_framework.png)

To achieve effective instruction-following audio generation, the ability to understand the input instruction or audio stream and reason about relevant audio sub-events is essential. To this end,  AudioStory adopts a unified understanding-generation framework (Fig.). Specifically, given textual instruction or audio input, the LLM analyzes and decomposes it into structured audio sub-events with context. Based on the inferred sub-events, the LLM performs **interleaved reasoning generation**, sequentially producing captions, semantic tokens, and residual tokens for each audio clip. These two types of tokens are fused and passed to the DiT, effectively bridging the LLM with the audio generator. Through progressive training, AudioStory ultimately achieves both strong instruction comprehension and high-quality audio generation.



## 🔩 Installation

### Dependencies

* Python >= 3.10 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
* [PyTorch >=2.1.0](https://pytorch.org/)
* NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Installation

```
git clone https://github.com/TencentARC/AudioStory.git
cd AudioStory
conda create -n audiostory python=3.10 -y
conda activate audiostory
bash install_audiostory.sh
```



## 📊 Evaluation

### inference

```
python evaluate/inference.py --model_path /path/to/ckpt --guidance 4.0 --save_folder_name audiostory --total_duration 50
```



## 🔋 Acknowledgement

When building the codebase of continuous denosiers, we refer to [SEED-X](https://github.com/AILab-CVC/SEED-X) and [TangoFlux](https://github.com/declare-lab/TangoFlux). Thanks for their wonderful projects.



## 📆 TO DO

- [ ] Release our gradio demo.
- [ ] Release checkpoints of AudioStory.
- [ ] Release training codes of all three stages.



## 📜 License

This repository is under the [Apache 2 License](https://github.com/mashijie1028/Gen4Rep/blob/main/LICENSE).



## 📚 BibTeX

```
@misc{guo2025audiostory,
      title={AudioStory: Generating Long-Form Narrative Audio with Large Language Models}, 
      author={Yuxin Guo and Teng Wang and Yuying Ge and Shijie Ma and Yixiao Ge and Wei Zou and Ying Shan},
      year={2025},
      eprint={2508.20088},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.20088}, 
}
```



## 📧 Contact

If you have further questions, feel free to contact me: guoyuxin2021@ia.ac.cn

Discussions and potential collaborations are also welcome.
