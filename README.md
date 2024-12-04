# SI507-Project: Med-VQA Evaluation Project
This repository contains the code and resources for evaluating Vision-Language Pretrained (VLP) models on Medical Visual Question Answering (Med-VQA) tasks. The project focuses on comparing the performance of general-purpose and domain-specific models, including CLIP, PubMedCLIP, and BioMedCLIP, on two prominent datasets: VQA-RAD and SLAKE.

## Overview
Medical Visual Question Answering (Med-VQA) is a challenging task that requires a deep understanding of both medical images and corresponding textual questions. This project evaluates how well state-of-the-art VLP models, pre-trained with or without domain-specific knowledge, perform on Med-VQA tasks.

## Datasets
The following datasets are used in this project:

- VQA-RAD: A dataset consisting of radiology images paired with clinical questions.
- SLAKE: A bilingual dataset for medical visual question answering with knowledge-based and vision-only questions.
## Models
The project benchmarks the following models:

- CLIP: A general-purpose Vision-Language model pre-trained on natural images and texts.
- PubMedCLIP: A CLIP-based model pre-trained on medical data to capture domain-specific knowledge.
- BioMedCLIP: A state-of-the-art Vision-Language model specifically pre-trained on large-scale biomedical datasets.
## Features
Evaluate Med-VQA models on Open, Closed, and Overall question types.
Compare performance on domain-specific datasets with multilingual capabilities (EN & ZH).
Analyze the impact of domain-specific pretraining on accuracy and generalizability.
