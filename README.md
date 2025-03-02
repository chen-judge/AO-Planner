# AO-Planner

The official implementation of AO-Planner. [[Paper]](https://arxiv.org/abs/2407.05890) [[Project]](https://chen-judge.github.io/AO-Planner/)

**Affordances-Oriented Planning using Foundation Models for Continuous Vision-Language Navigation.**

Jiaqi Chen, Bingqian Lin, Xinmin Liu, Lin Ma, Xiaodan Liang, Kwan-Yee K. Wong.

AAAI Conference on Artificial Intelligence (**AAAI 2025**).

## Setup
Our framework requires two different environments. The low-level agent (including Grounded SAM and Gemini) needs to run in Python 3.10, while the high-level agent (including Habitat simulator and GPT) runs in Python 3.7.

Therefore, we implement a simple file-based communication mechanism between the two environments, which also facilitates saving the outputs of low-level and high-level agents.

You need to run the code in these two environments simultaneously.

### Low-Level Agent

1. Create a virtual environment. We use Python 3.10 for the low-level agent.
```bash
conda create -n py310 python=3.10
conda activate py310
```

2. Please follow this repo:
[https://github.com/IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to install the environment for Grounded SAM.

3. Install Gemini api
```bash
pip install -U google-generativeai
```

### High-Level Agent

We follow [[https://github.com/YicongHong/Discrete-Continuous-VLN]](https://github.com/YicongHong/Discrete-Continuous-VLN?tab=readme-ov-file) to install Habitat Simulator but use Python 3.7 instead.

1. Create a virtual environment. We use Python 3.7 for the high-level agent.
```bash
conda create -n py37 python=3.7
conda activate py37
```

2. Install habitat-sim.
```bash
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
```

3. Install all requirements.
```bash
git clone https://github.com/chen-judge/AO-Planner.git
cd AO-Planner
python -m pip install -r requirements.txt
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

4. Clone a stable habitat-lab version from the github repository and install. The command below will install the core of Habitat Lab as well as the habitat_baselines.
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
python setup.py develop --all # install habitat and habitat_baselines

5. Others.
Our code is based on [ETPNav](https://github.com/MarSaKi/ETPNav), which requires the use of CLIP. Although CLIP is not actually necessary for our project, we haven't had the time to clean up and organize all the code yet. You may need to install this package in order to run this project. We feel sorry about this inconvenience and will reorganize the code in time.
```bash
pip install git+https://github.com/openai/CLIP.git
```


## Running

You need to run these two scripts simultaneously.

```bash
bash run_r2r/zero_shot.bash
```

```bash
bash llm/run_grounded_sam.sh
```

## Citation

```bash
@inproceedings{chen2024affordances,
  title={Affordances-Oriented Planning using Foundation Models for Continuous Vision-Language Navigation},
  author={Chen, Jiaqi and Lin, Bingqian and Liu, Xinmin and Ma, Lin and Liang, Xiaodan and Wong, Kwan-Yee~K.},
  booktitle = "Proceedings of the AAAI Conference on Artificial Intelligence",
  year={2025}
}
```

```bash
@inproceedings{chen2024mapgpt,
  title={MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation},
  author={Chen, Jiaqi and Lin, Bingqian and Xu, Ran and Chai, Zhenhua and Liang, Xiaodan and Wong, Kwan-Yee~K.},
  booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics",
  year={2024}
}
```

