# 🧠 LLM Agent for Image Analysis Pipelines

This repository explores how **LLM agent** can enhance and orchestrate **image analysis pipelines**, combining intelligent decision-making with deterministic image processing tools.

The work is inspired by the Hugging Face course:
👉 https://huggingface.co/learn/agents-course

---

## 🚀 Overview

Classical image analysis pipelines are often rigid and require manual orchestration of multiple steps.
This project demonstrates how **LLM agents can act as a control layer**, enabling:

* Dynamic tool selection
* Flexible pipeline execution
* Natural language interaction with image analysis workflows

---

## 🤖 Agents Explored

Based on the Hugging Face Agents Course, this repository explores:

* **SmolAgents**
* **LangGraph**
* **Transformers Agents (Hugging Face)**
* **ReAct-style tool-using agents**
* **Code Agents**

The goal is to evaluate how each framework can support **modular and extensible image analysis pipelines**.

---

## 🧪 Image Analysis Tools

All image processing capabilities rely on **deterministic, publicly available libraries**, such as:

* `scikit-image (skimage)`
* `OpenCV`
* `NumPy`

The LLM does **not process images directly**. Instead, it:

* Chooses which tools to use
* Decides execution order
* Interprets user instructions

---

## 🧩 Key Concept

> The LLM acts as an **orchestrator**, not the computation engine.

This ensures:

* Reproducibility
* Transparency
* Separation between reasoning and execution

---

## 📁 Repository Structure

```bash
.
├── agent.py            # Core agent logic (LLM + tool orchestration)
├── app.py              # FastAPI backend
├── tools/              # Image processing tools (e.g., colorize, filters)
├── static/             # Frontend assets
├── index.html          # Simple UI (chat interface)
├── env.yml             # Environment configuration
├── __pycache__/        # Python cache (ignored)
├── .gitignore
└── README.md
```

---

## 🌐 System Architecture

```
User (Browser UI)
        ↓
FastAPI (app.py)
        ↓
LLM Agent (agent.py)
        ↓
Tool Execution (tools/)
        ↓
Image Processing Output
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create environment

Using conda:

```bash
conda env create -f env.yml
conda activate smolagent
```

Or with pip (if preferred):

```bash
pip install -r requirements.txt
```

---

### 3. Start the FastAPI server

```bash
uvicorn app:app --reload
```

---

### 4. Open the interface

Go to your browser:

```
http://127.0.0.1:8000
```

---

## 💡 Example Usage

You can interact with the system using natural language:

```
"Load the image and apply color normalization"
"Detect edges and highlight structures"
```

The agent will:

1. Interpret the request
2. Select appropriate tools
3. Execute the pipeline

---

## 🔬 Motivation

This project explores how LLM agents can be applied to domains like:

* Microscopy image analysis
* Biomedical imaging pipelines

Where workflows are often:

* Multi-step
* Tool-heavy
* Hard to generalize

---

## 📚 Reference

Hugging Face Agents Course:
👉 https://huggingface.co/learn/agents-course

---

## 📌 Disclaimer

This repository is for **research and experimentation purposes only**.
It demonstrates how LLM agents can assist in building flexible image analysis pipelines, but does not replace validated production workflows.

---
