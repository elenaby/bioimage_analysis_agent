# 🧠 LLM Agents for Image Analysis Pipelines

This repository explores how **Large Language Model (LLM) agents** can enhance and orchestrate **image analysis pipelines**, with a focus on modularity, automation, and interpretability.

The project is inspired by and follows concepts from the Hugging Face course:
👉 https://huggingface.co/learn/agents-course

---

## 🚀 Overview

Traditional image analysis pipelines are often:

* Rigid
* Hard to extend
* Difficult to automate across multiple steps

This project investigates how **LLM-powered agents** can:

* Dynamically select tools
* Chain together image processing operations
* Interpret user instructions
* Enable flexible and interactive workflows

---

## 🤖 Agents Explored

This repository experiments with multiple agent frameworks introduced in the course, including:

* **SmolAgents**
* **LangGraph**
* **Transformers Agents (Hugging Face)**
* **Tool-using LLM agents (ReAct-style)**
* **Code Agents**

Each framework is evaluated for:

* Ease of integration
* Control over tool execution
* Suitability for image analysis workflows

---

## 🧪 Image Analysis Capabilities

All image processing functionality is built using **publicly available Python libraries**, including:

* `scikit-image (skimage)`
* `OpenCV`
* `NumPy`

The LLM agents **do not perform image processing directly** — instead, they:

* Decide **which tool to use**
* Determine **execution order**
* Interpret **user intent**

This separation ensures:

* Deterministic, reproducible image processing
* Flexible, intelligent pipeline orchestration

---

## 🧩 Key Idea

> The LLM acts as a **decision-making layer**, not a computation engine.

It translates natural language instructions into:

* Tool selection
* Parameter configuration
* Multi-step workflows

---

## 📁 Repository Structure

```
.
├── agents/              # Agent implementations (SmolAgents, LangGraph, etc.)
├── tools/               # Image processing tools (wrapping skimage, OpenCV)
├── pipelines/           # Example workflows
├── api/                 # (Optional) FastAPI interface
├── examples/            # Demo scripts
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create environment

```bash
conda create -n llm-agents python=3.10
conda activate llm-agents
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run an example

```bash
python examples/run_agent.py
```

---

## 💡 Example Use Case

You can provide a prompt like:

```
"Load the image, detect tissue regions, and apply edge detection"
```

The agent will:

1. Select appropriate tools
2. Execute them in sequence
3. Return results

---

## 📚 Learning Resource

This project is based on concepts from the Hugging Face Agents Course:
👉 https://huggingface.co/learn/agents-course

---

## 📌 Disclaimer

This repository is for **research and experimentation purposes**.
It demonstrates how LLM agents can assist—not replace—traditional image analysis pipelines.

---
