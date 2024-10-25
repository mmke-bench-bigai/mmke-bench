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
- [Star History](#star-history)
- [🎉Contributors](#contributors)





## 🔔 News

* **[2024.07.02]**  **Code** is available now!

* **[2024.07.02]**  We release the **ICE dataset** at 🤗 [Huggingface Dataset](https://huggingface.co/datasets/Yofuria/ICE).

* **[2023.06.18]**  We hung the **paper** on 🤗 [Huggingface Papers](https://huggingface.co/papers/2406.11194).

  

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

In **lines 32 and 33** of **` examples/run_knowedit_llama2.py`**, you need to download the **` punkt `** package.

- If your Internet **speed is fast** enough, you can **run the code directly** from the command line.

```text
if __name__ == "__main__":
    # If you have a slow Internet connection and can't download nltk quickly, comment these two lines and use the second method of Requirements and Installation in README.md
    import nltk
    nltk.download('punkt')
```

- If your Internet **speed is slow**, **comment lines 32 and 33** and **download punkt manually**🤗 [punkt]([kailinjiang/punkt · Datasets at Hugging Face](https://huggingface.co/datasets/kailinjiang/punkt)). And place it in the ICE environment directory you created, create a **nltk_data/tokenizers** folder, and **unpack punkt** into this directory.

<div align="center">   <img src="assets/punkt.png" width="650px"> </div>



## 💥Training

We provide the training hyperparameters for five methods in `./hparams`. 

For ICE, we update **GPT2-xl** using **layers 13 to 17** and **Llama2-7b-chat** using **layers 4 to 8**. 

Both FT-L and FT-M use the same hparams located in `./hparams/FT`. 

For FT-L, replace `objective_optimization` with `prompt_last`, and for FT-M, replace it with `target_new`. For details on other methods, please refer to [EasyEdit](https://github.com/zjunlp/EasyEdit). You can execute the following commands to obtain results:

**For ICE:**

```shell
python examples/run_knowedit_llama2.py \
	--editing_method=ICE \
	--hparams_dir=./hparams/ICE/gpt2-xl.yaml \
    --data_dir=./data/zsre.json \  
    --datatype='zsre' \  
    --metrics_save_dir=./results/gpt2-xl/ICE
```

**For FT-L:**

```shell
python examples/run_knowedit_llama2.py \
	--editing_method=FT-L \
	--hparams_dir=./hparams/ICE/gpt2-xl.yaml \
    --data_dir=./data/zsre.json \  
    --datatype='zsre' \  
    --metrics_save_dir=./results/gpt2-xl/ICE
```

**For FT-M:**

```shell
python examples/run_knowedit_llama2.py \
	--editing_method=FT-M \
	--hparams_dir=./hparams/ICE/gpt2-xl.yaml \
    --data_dir=./data/zsre.json \  
    --datatype='zsre' \  
    --metrics_save_dir=./results/gpt2-xl/ICE
```

**For MEMIT:**

```shell
python examples/run_knowedit_llama2.py \
	--editing_method=MEMIT \
	--hparams_dir=./hparams/ICE/gpt2-xl.yaml \
    --data_dir=./data/zsre.json \  
    --datatype='zsre' \  
    --metrics_save_dir=./results/gpt2-xl/ICE
```

**For ROME:**

```shell
python examples/run_knowedit_llama2.py \
	--editing_method=ROME \
	--hparams_dir=./hparams/ICE/gpt2-xl.yaml \
    --data_dir=./data/zsre.json \  
    --datatype='zsre' \  
    --metrics_save_dir=./results/gpt2-xl/ICE
```

The optional range of `datatype` is `['zsre','recent','counterfact','wikibio']` 





## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```text
@article{qi2024ice,
      title={In-Context Editing: Learning Knowledge from Self-Induced Distributions}, 
      author={Siyuan Qi and Bangcheng Yang and Kailin Jiang and Xiaobo Wang and Jiaqi Li and Yifan Zhong and Yaodong Yang and Zilong Zheng},
      year={2024},
      eprint={2406.11194},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.11194}, 
}
```



## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mmke-bench-bigai/mmke-bench&type=Date)](https://star-history.com/#mmke-bench-bigai/mmke-bench&Date)





## 🎉Contributors

<a href="https://github.com/mmke-bench-bigai/mmke-bench/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mmke-bench-bigai/mmke-bench" />
</a>





