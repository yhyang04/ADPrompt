## Introduction

This is our implementation of the paper *Adaptive Dual Prompting: Hierarchical Debiasing for Fairness-aware Graph Neural Networks*.

**Abstract**:
In recent years, pre-training Graph Neural Networks (GNNs) through self-supervised learning on unlabeled graph data has emerged as a widely adopted paradigm in graph learning. Although the paradigm is effective for pre-training powerful GNN models, the objective gap often exists between pre-training and downstream tasks. To bridge this gap, graph prompting adapts pre-trained GNN models to specific downstream tasks with extra learnable prompts while keeping the pre-trained GNN models frozen. As recent graph prompting methods largely focus on enhancing model utility on downstream tasks, they often overlook fairness concerns when designing prompts for adaptation. In fact, pre-trained GNN models will produces discriminative node representations across demographic subgroups because the downstream graph data itself contains inherent biases in both node attributes and graph structures. To address this issue, we propose a Adaptive Dual Prompting (ADPrompt) framework that enhances fairness for adapting pre-trained GNN models to downstream tasks. To mitigate attribute bias in graph prompting, we design an adaptive feature rectification module that learns customized attribute prompts to suppress sensitive information via a self-gating mechanism at the input layer, thereby reducing biased inputs at the source. Afterward, we propose an adaptive message calibration module that generates structure prompts at each GNN layer, which adjust the messages from neighboring nodes to enable dynamic and soft calibration of the information flow. In the end, ADPrompt optimizes these two prompting modules using a joint optimization objective for adapting the pre-trained GNN model while enhancing model fairness. We conduct extensive experiments on four datasets with four pre-training strategies to evaluate the performance of ADPrompt. The results demonstrate that our proposed ADPrompt outperforms seven baseline methods on node classification tasks.

---

**Note:** This repository currently contains the core implementation details and essential code to reproduce the main results. We are actively working on refining and expanding the codebase, and the full code repository will be made available in a subsequent update. Thank you for your patience and understanding.

---

## Dependencies

- python==3.9
- torch==2.2.1 (CUDA 12.1)
- torch-geometric==2.5.2
- torch-scatter==2.1.2
- torch-sparse==0.6.18
- torch-cluster==1.6.3
- numpy==1.26.1

---

## Usage

### 1. Install dependencies

```bash
conda create --name DFPrompt -y python=3.9
conda activate DFPrompt

pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.1 torch-geometric==2.5.2

pip install torch-cluster torch-sparse torch-scatter \
  -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
