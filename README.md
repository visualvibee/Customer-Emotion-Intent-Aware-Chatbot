# Emotion-Aware Customer Support AI

> A transformer-based customer support system that jointly classifies **user intent** and **emotional state** to generate dynamic, empathetic responses — going beyond traditional static chatbot replies.

---

## Overview

Most customer support chatbots only detect *what* a user wants (intent), ignoring *how* they feel. This project addresses that gap by running two fine-tuned **RoBERTa-base** models in parallel — one for intent classification, one for emotion detection — and fusing their outputs to produce context-aware, emotionally appropriate responses in real time.

---

## Architecture

The system is composed of three layers:

```
User Utterance
      │
      ▼
┌─────────────────────────────────────────┐
│           Classification Layer           │
│  ┌──────────────┐   ┌─────────────────┐ │
│  │ Intent Model │   │  Emotion Model  │ │
│  │ (RoBERTa)    │   │  (RoBERTa)      │ │
│  │  7 classes   │   │  6 classes      │ │
│  └──────┬───────┘   └────────┬────────┘ │
└─────────│───────────────────│──────────┘
          │                   │
          ▼                   ▼
┌─────────────────────────────────────────┐
│              Fusion Layer                │
│      ⟨intent_label, emotion_label⟩      │
│         + confidence scores             │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│          Response Generation Layer       │
│   Emotion Prefix Bank + Intent Response │
│   Bank → Final Empathetic Response      │
└─────────────────────────────────────────┘
```

---

## Models

| Component | Details |
|---|---|
| **Intent Classifier** | RoBERTa-base, fine-tuned on Banking77 + CLINC150 + SNIPS + Custom CSVs; 7 classes |
| **Emotion Detector** | RoBERTa-base, fine-tuned on GoEmotions (filtered); 6 classes |
| **Fusion Layer** | Builds `⟨intent, emotion⟩` context tuple with per-class confidence scores |
| **Response Generator** | Emotion prefix bank + intent response bank; assembled via string concatenation |
| **Fallback** | Triggered when either model confidence < 0.50; escalates to human agent |

### Intent Classes
`complaint` · `inquiry` · `support_request` · `refund_replace` · `feedback` · `appreciation` · `cancel_request`

### Emotion Classes
`angry` · `happy` · `sad` · `neutral` · `confused` · `frustrated`

---

## Datasets

| Dataset | Used For |
|---|---|
| [Banking77](https://huggingface.co/datasets/PolyAI/banking77) | Intent classification |
| [CLINC150](https://huggingface.co/datasets/clinc_oos) | Intent classification |
| [SNIPS](https://github.com/snipsco/nlu-benchmark) | Intent classification |
| Custom CSVs (appreciation, feedback) | Intent classification |
| [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Emotion detection |

All intent labels are consolidated into 7 unified classes. GoEmotions is filtered to single-label samples and remapped to 6 application-relevant emotions.

---

## Results

### Intent Classification (Test set: 500 samples)

| Metric | Score |
|---|---|
| Accuracy | **98%** |
| Weighted F1 | **0.98** |

Classes `appreciation`, `cancel_request`, and `feedback` achieved **perfect precision and recall (1.00)**.

### Emotion Detection (Test set: 2,117 samples)

| Metric | Score |
|---|---|
| Accuracy | **84%** |
| Weighted F1 | **0.83** |

`neutral` class achieved F1 of 0.91. Boundary emotions (`frustrated`, `confused`) are harder to distinguish due to linguistic overlap — a known challenge in affective computing.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Base Model | `roberta-base` (HuggingFace) |
| Max Token Length | 64 |
| Batch Size | 16 |
| Optimizer | AdamW |
| Learning Rate | 2 × 10⁻⁵ |
| LR Scheduler | Linear decay with 10% warmup |
| Max Epochs | 5 |
| Early Stopping Patience | 2 (validation loss) |
| Gradient Clipping | `max_norm = 1.0` |
| Intent Loss | Weighted CrossEntropy |
| Emotion Loss | Standard CrossEntropy |
| Train/Val/Test Split | 70% / 15% / 15% (stratified) |
| Random State | 42 |

---

## Installation

```bash
git clone https://github.com/<your-username>/emotion-aware-customer-support-ai.git
cd emotion-aware-customer-support-ai
pip install -r requirements.txt
```

### Requirements

```
torch
transformers
scikit-learn
pandas
numpy
datasets
```

---

## Usage

```python
from model import customer_support_ai

response = customer_support_ai("My card was declined and no one is helping me!")

print(response)
# {
#   "intent": "complaint",
#   "intent_confidence": 0.97,
#   "emotion": "frustrated",
#   "emotion_confidence": 0.82,
#   "response": "I'm really sorry you're going through this. Let me help resolve your complaint right away."
# }
```


---

## How Response Generation Works

At inference time:

1. The user utterance is tokenized (max 64 tokens) and fed to **both models in parallel**.
2. Each model outputs a predicted label and a confidence score via softmax + argmax.
3. The **Fusion Layer** combines outputs into `⟨intent, emotion⟩`.
4. If either confidence < 0.50, a **fallback response** is returned for human escalation.
5. Otherwise, an **emotion prefix** (e.g., *"I understand you're frustrated..."*) is prepended to an **intent-specific response** from the response bank.

---

## Authors

- **Tarni Khatri** — Manipal University Jaipur
- **Satvik Ahuja** — Manipal University Jaipur
- **Aditi Shrivastava** — Manipal University Jaipur

Department of Computer Science, Manipal University Jaipur, Jaipur, India

---



This project is for academic purposes. Please cite the associated paper if you use this work.
