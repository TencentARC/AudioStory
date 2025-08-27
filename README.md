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
<sup>2</sup>ARC Lab, Tencent PCG, 
<br>



## 📖 Release

[8/17] 🔥We release our code of evaluation and checkpoint!



## 🔖 Contents

[toc]

## 🔎 Introduction

![audiostory](audiostory.png)

Recent advances in text-to-audio (TTA) generation excel at synthesizing short audio clips but struggle with long-form narrative audio, which requires temporal coherence and compositional reasoning. To address this gap, we propose AudioStory, a unified framework that integrates large language models (LLMs) with TTA systems to generate structured, long-form audio narratives. AudioStory possesses strong instruction-following reasoning generation capabilities. It employs LLMs to decompose complex narrative queries into temporally ordered sub-tasks with contextual cues, enabling coherent scene transitions and emotional tone consistency. AudioStory has two appealing features: 

1) Decoupled bridging mechanism: AudioStory disentangles LLM-diffuser collaboration into two specialized components—a bridging query for intra-event semantic alignment and a consistency query for cross-event coherence preservation.
2) End-to-end training: By unifying instruction comprehension and audio generation within a single end-to-end framework, AudioStory eliminates the need for modular training pipelines while enhancing synergy between components. 
    Furthermore, we establish a benchmark AudioStory-10K, encompassing diverse domains such as animated soundscapes and natural sound narratives.

Extensive experiments show the superiority of AudioStory on both single-audio generation and narrative audio generation, surpassing prior TTA baselines in both instruction-following ability and audio fidelity.



## ⭐ Demos

### 1. Video Dubbing (Tom & Jerry)

<table class="center">
  <td><video src="https://github.com/user-attachments/assets/f06b5999-6649-44d3-af38-63fdcecd833c"></video></td>
  <td><video src="https://github.com/user-attachments/assets/17727c2a-bfea-4252-9aa8-48fc9ac33500"></video></td>
  <td><video src="https://github.com/user-attachments/assets/09589d82-62c9-47a6-838a-5a62319f35e2"></video></td>
  <tr>
  <td style="text-align:center;" width="320">"Tom Cruise's face reflects focus, his eyes filled with purpose and drive."</td>
  <td style="text-align:center;" width="320">"A child excitedly swings on a rusty swing set, laughter filling the air."</td>
  <td style="text-align:center;" width="320">"A young woman with glasses is jogging in the park wearing a pink headband."</td>
  <tr>
</table >





### 2. Video Dubbing (Other Videos to Tom & Jerry)

<table class="center">
  <td><video src="https://github.com/user-attachments/assets/34e19f06-3b30-4438-a817-9e131af410f3"></video></td>
  <td><video src="https://github.com/user-attachments/assets/4a6de0c6-ef50-4cc3-b31b-d873af6fdf79"></video></td>
  <td><video src="https://github.com/user-attachments/assets/76f7f5de-42c6-475a-853c-5e2ba11ab7b2"></video></td>
  <tr>
  <td style="text-align:center;" width="320">"Snoopy."</td>
  <td style="text-align:center;" width="320">"Nezha."</td>
  <td style="text-align:center;" width="320">"Nezha."</td>
  <tr>
  <td><video src="https://github.com/user-attachments/assets/74415b54-0432-4b0f-9afb-9f2ecf0a80f2"></video></td>
  <td><video src="https://github.com/user-attachments/assets/5141f15b-f2a9-413b-bac1-3c89d61c75dc"></video></td>
  <td><video src="https://github.com/user-attachments/assets/d0cfa875-4637-461c-a8e8-416407a7640c"></video></td>
  <tr>
  <td style="text-align:center;" width="320">"We Bare Bears."</td>
  <td style="text-align:center;" width="320">"Miffy."</td>
  <td style="text-align:center;" width="320">"Donald Duck."</td>
  <tr>
  <td><video src="https://github.com/user-attachments/assets/5c801b5e-ce74-42a2-b8cf-3325ab0d7c4a"></video></td>
  <td><video src="https://github.com/user-attachments/assets/5c9ed7e9-527e-4163-a19b-ffa56ab034dc"></video></td>
  <td><video src="https://github.com/user-attachments/assets/5d603a4a-bf45-4ce9-81a3-62950ea89e99"></video></td>
  <tr>
  <td style="text-align:center;" width="320">"Sora Videos from Official Website."</td>
  <td style="text-align:center;" width="320">"Sora Videos from Official Website."</td>
  <td style="text-align:center;" width="320">"pets."</td>
  <tr>
</table >




### 3. Text-to-Audio (Long Narrative)

<table class="center">
  <td><audio src="https://github.com/user-attachments/files/22002278/201_concated_clips_5.wav"></audio></td>
  <td><audio src="https://github.com/user-attachments/files/22002279/217_concated_clips_5.wav"></audio></td>
  <td><audio src="https://github.com/user-attachments/files/22002280/229_concated_clips_5.wav"></audio></td>
  <tr>
  <td style="text-align:center;" width="320">"Tom Cruise's face reflects focus, his eyes filled with purpose and drive."</td>
  <td style="text-align:center;" width="320">"A child excitedly swings on a rusty swing set, laughter filling the air."</td>
  <td style="text-align:center;" width="320">"A young woman with glasses is jogging in the park wearing a pink headband."</td>
  <tr>
</table >






## 🔎 Methods

![audiostory_framework](audiostory_framework.png)

To achieve effective instruction-following audio generation, the ability to understand the input instruction or audio stream and reason about relevant audio sub-events is essential. To this end,  AudioStory adopts a unified understanding-generation framework (Fig.). Specifically, given textual instruction or audio input, the LLM analyzes and decomposes it into structured audio sub-events with context. Based on the inferred sub-events, the LLM performs **interleaved reasoning generation**, sequentially producing captions, semantic tokens, and residual tokens for each audio clip. These two types of tokens are fused and passed to the DiT, effectively bridging the LLM with the audio generator. Through progressive training, AudioStory ultimately achieves both strong instruction comprehension and high-quality audio generation.



## 📊 Evaluation





## 🔋 Acknowledgement





## 📆 TO DO





## 📂 Related Projects

