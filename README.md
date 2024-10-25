<p align="center">
    <img src="figs/11.png" width="900" style="margin-bottom: 0.2;"/>
<p>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub.  </h2>



[![arXiv PDF](https://img.shields.io/badge/Arxiv-406.11194-ff5733?logo=arXiv)]()[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MMKE--Bench--Dataset-lightgrey)]()[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-MMKE--Bench--Model-3357ff)](https://huggingface.co/datasets/MMMU/MMMU)[![GitHub Code](https://img.shields.io/badge/GitHub-Code-ffd700?logo=github)]()[![Slides PDF](https://img.shields.io/badge/Slides-PDF-ff1493?logo=slideshare)](static/Slides/MMKE-Bench.pdf)



## Table of Contents

- [Table of Contents](#table-of-contents)
- [🔔 News](#-news)
- [🌟Overview](#overview)
- [🤗 Dataset](#-dataset)
- [🛠️ Requirements and Installation](#️-requirements-and-installation)
- [💥Training](#training)
- [✏️ Citation](#️-citation)
- [⭐ Star History](#star-history)
- [🎉Contributors](#contributors)





## 🔔 News

* **[2024.10.25]**  **Code** is available now!

* **[2024.10.25]**  We release the **MMKE-Bench dataset** at 🤗 [Huggingface Dataset](https://huggingface.co/datasets/Yofuria/ICE).

* **[2023.10.25]**  We hung the **paper** on 🤗 [Huggingface Papers](https://huggingface.co/papers/2406.11194).

  

## 🌟Overview

**TL;DR:** We propose <span style="color:brown">**MMKE-Bench**</span>, a challenging benchmark for evaluating diverse semantic editing in real-world scenarios.

<img src="figs\fig1.jpg" width="900px">

<p align="justify">
    <strong><span style="color:black">Overview</span> of the <span style="color:brown">MMKE-Bench</span> dataset. Our contribution can be summarized as follows:</strong>
</p>


<p align="justify" style="margin-left: 30px; text-indent: -30px;">
    <strong><span style="color:black">1) Overview of MMKE-Bench</span></strong>: 
    MMKE-Bench is introduced as a benchmark designed to test semantic editing capabilities in realistic scenarios. It utilizes natural language for knowledge representation and includes three editing types aligned with practical contexts.
</p>


<p align="justify" style="margin-left: 30px; text-indent: -30px;">
    <strong><span style="color:black">2) Development of the Benchmark Pipeline</span></strong>: 
    Describes the novel pipeline used to develop the benchmark, which includes collecting original knowledge, generating editable knowledge, and crafting evaluation questions based on specific principles.
</p>


<p align="justify" style="margin-left: 30px; text-indent: -30px;">
    <strong><span style="color:black">3) Experimental Analysis and Challenges</span></strong>: 
    Details extensive experiments with various standard methods and large language models, highlighting several limitations in the existing approaches to knowledge editing in both single and multiple edit scenarios.
</p>



## 🤗 Dataset

<p align="justify">
We introduce <strong><span style="color:brown">MMKE-Bench</span></strong>, a benchmark designed to evaluate the ability of LMMs to edit visual knowledge in real-world scenarios. <strong><span style="color:brown">MMKE-Bench</span></strong> incorporates three editing tasks: <strong><span style="color:brown">visual entity editing</span></strong>, <strong><span style="color:brown">visual semantic editing</span></strong>, and <strong><span style="color:brown">user-specific editing</span></strong>. Additionally, it uses free-form natural language to represent and edit knowledge, offering more flexibility. The benchmark includes <strong><span style="color:brown">2,940</span></strong> pieces of knowledge and <strong><span style="color:brown">7,229</span></strong> images across 110 fine-grained types, with automatically generated, human-verified evaluation questions.
</p>



You can download **MMKE-Bench data** 🤗 [Huggingface Dataset](https://huggingface.co/datasets/Yofuria/ICE). And the expected structure of files is:

```text
MMKE-Bench
|-- data_json
|   |-- entity
|   |   |-- train.json
|   |   |-- eval.json
|   |-- visual
|   |   |-- train.json
|   |   |-- eval.json
|   |-- user
|   |   |-- train.json
|   |   |-- eval.json
|-- data_image
|   |-- entity
|   |   |-- image.....
|   |-- visual
|   |   |-- image.....
|   |-- user
|   |   |-- image.....
```



## 🛠️ Requirements and Installation

```text
# clone ICE
git clone https://github.com/Yofuria/ICE.git
cd ICE

# create conda env
conda create -n ICE python=3.10
conda activate ICE

# install package
pip install -r requirements.txt
```



## 💥Training

**For FT-LLM:**

```shell
python multimodal_edit.py --function_name=test_FT_LLaVA --hop=1 --data_type=entity

function_name in ['test_FT_LLaVA','test_FT_MiniGPT4','test_FT_Blip2OPT']
data_type in ['entity','visual','user']
```

**For FT-Alignment:**

```shell
python multimodal_edit.py --function_name=test_FT_LLaVA_mmproj --hop=1 --data_type=entity

function_name in ['test_FT_LLaVA_mmproj','test_FT_MiniGPT4_Qformer','test_FT_Blip2OPT_QFormer']
data_type in ['entity','visual','user']
```

**For SERAC:**

```shell
python multimodal_edit.py --function_name=train_SERAC_LLaVA --hop=1 --data_type=entity

function_name in ['train_SERAC_LLaVA','train_SERAC_MiniGPT4','train_SERAC_Blip2OPT']
data_type in ['entity','visual','user']
```

**For MEND:**

```shell
python multimodal_edit.py --function_name=train_MEND_LLaVA --hop=1 --data_type=entity

function_name in ['train_MEND_LLaVA','train_MEND_MiniGPT4','train_MEND_Blip2OPT']
data_type in ['entity','visual','user']
```

**For IKE:**

```shell
python multimodal_edit.py --function_name=test_IKE_LLaVA --hop=1 --data_type=entity

function_name in ['test_IKE_LLaVA','test_IKE_MiniGPT4','test_IKE_Blip2OPT']
data_type in ['entity','visual','user']
```

**For KE:**

```shell
bash KE/train_ke.sh 0 llava entity
bash KE/train_ke.sh 0 minigpt4 entity
bash KE/train_ke.sh 0 blip2 entity

model_name in ['llava','minigpt4','blip2']
data_type in ['entity','visual','user']
```



## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```text
@article{du2024mmke_bench,
            title = {MMKE-Bench: A Multimodal Editing Benchmark for Diverse Visual Knowledge},
            author = {Yuntao Du and Kailin Jiang and Zhi Gao and Chenrui Shi and Zilong Zheng and Siyuan Qi and Qing Li},
            year = {2024}
          }
```



## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mmke-bench-bigai/mmke-bench&type=Date)](https://star-history.com/#mmke-bench-bigai/mmke-bench&Date)





## 🎉Contributors

<a href="https://github.com/mmke-bench-bigai/mmke-bench/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mmke-bench-bigai/mmke-bench" />
</a>





