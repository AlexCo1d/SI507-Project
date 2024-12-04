# SI507-Project: Med-VQA Evaluation Project
This repository contains the code and resources for evaluating Vision-Language Pre-trained (VLP) models on Medical Visual Question Answering (Med-VQA) tasks. The project focuses on comparing the performance of general-purpose and domain-specific models, including CLIP, PubMedCLIP, and BioMedCLIP, on two prominent datasets: VQA-RAD and SLAKE.

## Overview
Medical Visual Question Answering (Med-VQA) is a challenging task that requires a deep understanding of both medical images and corresponding textual questions. This project evaluates how well state-of-the-art VLP models, pre-trained with or without domain-specific knowledge, perform on Med-VQA tasks.

## Datasets
The following datasets are used in this project:

- [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad): A dataset consisting of radiology images paired with clinical questions. 
- [SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE): A bilingual dataset for medical visual question answering with knowledge-based and vision-only questions.
## Models
The project benchmarks the following models:

- [CLIP](https://huggingface.co/openai/clip-vit-base-patch32): A general-purpose Vision-Language model pre-trained on natural images and texts.
- [PubMedCLIP](https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32): A CLIP-based model pre-trained on medical data to capture domain-specific knowledge.
- [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224): A state-of-the-art Vision-Language model specifically pre-trained on large-scale biomedical datasets.

## Features
Evaluate Med-VQA models on Open, Closed, and Overall question types.
Compare performance on domain-specific datasets with multilingual capabilities (EN & ZH).
Analyze the impact of domain-specific pretraining on accuracy and generalizability.

## Usage
Download the datasets and model weights, add to specific root for evaluation using each model file and dataset.
