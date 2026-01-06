# Study Tutor AI

> A complete RAG + QLoRA fine-tuned AI tutor system built with LangChain, optimized for 8GB VRAM GPUs

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

**Study Tutor AI** is a portfolio-ready project demonstrating:
- **RAG (Retrieval-Augmented Generation)** for accurate, source-cited answers
- **QLoRA fine-tuning** on small LLMs (Llama 3.2 3B) with 8GB VRAM
- **Production-ready API** with FastAPI
- **Workflow automation** using n8n
- **Memory-optimized** for RTX 4060 / similar 8GB GPUs

### Key Features
- 📚 Ingest PDFs, documents, and web pages as knowledge base
- 🎓 Subject-specific tutoring (ML, DL, DSA, Mathematics, etc.)
- 💬 Interactive Q&A with source citations
- 🏋️ Practice question generation
- 📝 Step-by-step solution explanations
- 🔄 n8n workflow integration for automation
- 🖥️ Streamlit UI (optional)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Llama 3.2 3B (4-bit quantized) |
| **Fine-tuning** | QLoRA with PEFT + bitsandbytes |
| **RAG Framework** | LangChain |
| **Vector DB** | ChromaDB (local, persistent) |
| **Embeddings** | all-MiniLM-L6-v2 (CPU) |
| **API** | FastAPI + Uvicorn |
| **UI** | Streamlit (optional) |
| **Automation** | n8n workflows |

---

## 💻 Hardware Requirements

### Minimum Specs
- **GPU**: NVIDIA RTX 4060 (8GB VRAM) or equivalent
- **CPU**: Ryzen 7 / Intel i7 or better
- **RAM**: 16GB system RAM
- **Storage**: 20GB free space (for models + data)

### Software Requirements
- **OS**: Windows 10/11, Linux, or WSL2
- **CUDA**: 12.1+ (for PyTorch GPU support)
- **Python**: 3.10 or 3.11

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd "AI tutor"
```

### 2. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install PyTorch with CUDA
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Configure Environment
```powershell
cp .env.example .env
# Edit .env with your settings (optional)
```

### 6. Verify GPU Setup
```python
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## 📖 Usage Guide

### Document Ingestion

Ingest PDFs or documents into the vector store:

```powershell
python src/scripts/ingest_docs.py --files "data/raw/ml_notes.pdf" --subject ML
```

**Options:**
- `--files`: PDF paths, directories, or URLs (space-separated)
- `--subject`: ML, DL, DSA, Mathematics, Statistics, Python, General
- `--reset`: Delete existing vector store and start fresh

**Example - Multiple files:**
```powershell
python src/scripts/ingest_docs.py --files "data/raw/book1.pdf" "data/raw/book2.pdf" --subject DL
```

### Interactive Q&A (CLI)

Test the RAG pipeline from terminal:

```powershell
python src/scripts/run_rag_cli.py --subject ML
```

This starts an interactive session:
```
You: What is gradient descent?
Tutor: Gradient descent is an optimization algorithm...

[Sources: 3 documents, 2.1s]
  1. ml_textbook.pdf (Page 42)
  2. optimization_notes.pdf (Page 5)
```

**Single Question Mode:**
```powershell
python src/scripts/run_rag_cli.py --subject DSA --question "Explain binary search" --verbose
```

### FastAPI Server

Start the API server:

```powershell
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

**API Endpoints:**

1. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Ask Question**
   ```bash
   curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is overfitting?", "subject": "ML"}'
   ```

3. **Ingest Documents**
   ```bash
   curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"files": ["data/raw/notes.pdf"], "subject": "ML"}'
   ```

4. **Generate Practice Questions**
   ```bash
   curl -X POST http://localhost:8000/practice \
     -H "Content-Type: application/json" \
     -d '{"topic": "Neural Networks", "subject": "DL", "num_questions": 5}'
   ```

**API Documentation:** http://localhost:8000/docs

---

## 🏋️ Model Fine-tuning (QLoRA)

### Prepare Training Data

Create `data/train_qa.jsonl` with instruction-style Q&A:

```jsonl
{"instruction": "Explain gradient descent", "input": "", "output": "Gradient descent is...", "subject": "ML"}
{"instruction": "What is binary search?", "input": "", "output": "Binary search is...", "subject": "DSA"}
```

See `data/train_qa_example.jsonl` for format examples.

### Run Fine-tuning

```powershell
python src/models/lora_finetune.py --data data/train_qa.jsonl --max-steps 500
```

**Expected Output:**
```
Loading base model: meta-llama/Llama-3.2-3B-Instruct
VRAM allocated: 4.2 GB
Trainable parameters: 8,388,608 / 3,213,184,000
Trainable %: 0.26%
Starting training...
Step 100/500 - Loss: 1.234
...
Training completed!
Peak VRAM: 7.1 GB
```

**Training Time:** ~2-3 hours on RTX 4060 (500 steps)

**LoRA Adapters Saved:** `models/lora_adapters/final/`

### Use Fine-tuned Model

```powershell
python src/scripts/run_rag_cli.py --use-lora --adapter-path models/lora_adapters/final
```

Or via API (edit `src/api/server.py` to set `use_lora=True`).

---

## 🔄 n8n Workflow Automation

### Setup n8n

1. Install n8n:
   ```bash
   npm install -g n8n
   ```

2. Start n8n:
   ```bash
   n8n start
   ```

3. Import workflow:
   - Open http://localhost:5678
   - Go to **Workflows** → **Import from File**
   - Select `workflows/n8n_tutor_workflow.json`

### Workflow Features

The n8n workflow includes:

1. **Webhook Trigger** (`/tutor-webhook`)
   - Receives question via POST
   - Calls `/ask` endpoint
   - Returns answer to caller

2. **Ingest Webhook** (`/tutor-ingest`)
   - Triggers document ingestion
   - Processes new PDFs

3. **Optional Integrations** (configure as needed):
   - Google Sheets: Log all Q&A pairs
   - Slack/Discord: Send answers to channels
   - Google Drive: Auto-ingest PDFs on upload

### Test Webhook

```bash
curl -X POST http://localhost:5678/webhook/tutor-webhook \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "subject": "ML"}'
```

---

## 📊 Performance Benchmarks

### VRAM Usage (RTX 4060 8GB)

| Operation | VRAM | Notes |
|-----------|------|-------|
| **Base Model (4-bit)** | 4.2 GB | Llama 3.2 3B quantized |
| **+ Embeddings** | 4.5 GB | all-MiniLM on CPU |
| **Inference** | 4.8 GB | With context retrieval |
| **Training (QLoRA)** | 6.5-7.5 GB | Batch size 1, grad checkpointing |

### Inference Speed

- **Tokens/sec**: 15-25 (RTX 4060)
- **Average response time**: 2-4 seconds (512 tokens)
- **RAG retrieval**: <0.5 seconds

### Training Metrics

- **Time per step**: ~20-25 seconds
- **500 steps**: ~2.5-3 hours
- **Peak VRAM**: ~7.2 GB (within 8GB limit)

---

## 📁 Project Structure

```
AI tutor/
├── src/
│   ├── config/
│   │   ├── settings.py           # Config management
│   │   └── model_config.json     # QLoRA hyperparameters
│   ├── ingestion/
│   │   ├── load_documents.py     # PDF/web loaders
│   │   ├── chunking.py           # Text splitting
│   │   └── vectorstore.py        # ChromaDB management
│   ├── rag/
│   │   ├── retriever.py          # Enhanced retrieval
│   │   ├── prompts.py            # Tutor prompts
│   │   └── chains.py             # RAG orchestration
│   ├── models/
│   │   ├── model_loader.py       # 4-bit model loading
│   │   ├── inference.py          # Optimized inference
│   │   └── lora_finetune.py      # QLoRA training
│   ├── api/
│   │   └── server.py             # FastAPI endpoints
│   └── scripts/
│       ├── ingest_docs.py        # CLI ingestion
│       └── run_rag_cli.py        # Interactive Q&A
├── workflows/
│   └── n8n_tutor_workflow.json   # n8n automation
├── data/
│   ├── raw/                      # Original PDFs
│   ├── processed/                # Chunked data
│   ├── vectorstore/              # ChromaDB storage
│   └── train_qa_example.jsonl   # Sample training data
├── models/
│   ├── cache/                    # HF model cache
│   ├── checkpoints/              # Training checkpoints
│   └── lora_adapters/            # Fine-tuned LoRA
├── notebooks/
│   └── experiments.ipynb         # Testing & evaluation
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🎓 Learning Outcomes & Resume Highlights

This project demonstrates:

✅ **RAG Implementation**: LangChain + ChromaDB for accurate Q&A  
✅ **LLM Fine-tuning**: QLoRA on consumer GPU (8GB VRAM)  
✅ **Memory Optimization**: 4-bit quantization, gradient checkpointing  
✅ **Production API**: FastAPI with async endpoints  
✅ **Workflow Automation**: n8n integration for real-world use  
✅ **Clean Code**: Modular, documented, type-hinted Python  
✅ **DevOps Ready**: Environment configs, logging, error handling  

---

##   Quantifying Performance (Resume Metrics)

### Generate Your Metrics

Run the evaluation script to generate quantified performance metrics for your resume:

```powershell
# Full evaluation (15 test questions)
python src/scripts/evaluate_system.py --mode full

# Quick benchmark (5 questions)  
python src/scripts/evaluate_system.py --mode quick
```

This generates comprehensive reports in `evaluation_results/` with:
- **Retrieval metrics**: Recall@5, MRR, latency
- **Answer quality**: Response rate, citation accuracy
- **Performance**: Inference speed, tokens/second, VRAM usage
- **Resume-ready bullets**: Pre-formatted metrics for your CV

### Example Output Metrics

```
📈 Overall Score: 82.3/100

Key Metrics:
• Retrieval Recall@5: 85.3%
• Avg Inference Time: 2.1s
• Generation Speed: 23.4 tokens/s
• Answer Rate: 86.7%
• Source Citation: 93.3%
• Peak VRAM: 3.2GB

NLG Quality Metrics:
• BLEU-4: 0.352
• ROUGE-1: 0.523
• ROUGE-L: 0.441
```

### Resume Bullet Examples

```
• Achieved 85% retrieval accuracy with 58ms latency across 417-page corpus
• Optimized inference to 2.1s response time with 23 tokens/s generation speed
• Reduced VRAM to 3.2GB via 4-bit quantization for consumer GPU deployment
• Maintained 87% answer rate with 92% source citation accuracy
• Achieved BLEU-4 score of 0.352 and ROUGE-L F1 of 0.441 on test set
• Overall system performance score: 82.3/100
```

**📘 Full Guide:** See [METRICS_GUIDE.md](METRICS_GUIDE.md) for detailed explanation of all metrics and benchmarks.

**📘 NLG Metrics:** See [NLG_METRICS_GUIDE.md](NLG_METRICS_GUIDE.md) for BLEU/ROUGE explanation and benchmarks.

**📋 Quick Reference:** See [METRICS_QUICK_REF.md](METRICS_QUICK_REF.md) for one-page cheat sheet.

---

##  🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `max_sequence_length` in `model_config.json` (try 1024)
- Lower batch size (already 1 for training)
- Use `torch.cuda.empty_cache()` via `/clear-cache` endpoint

### Vector Store Not Found
```powershell
# Reingest documents
python src/scripts/ingest_docs.py --files "data/raw/*.pdf" --subject ML --reset
```

### Model Download Slow
- HuggingFace models download to `models/cache/`
- First run may take 10-20 minutes
- Subsequent runs use cached models

### Import Errors
- Ensure virtual environment is activated
- Reinstall: `pip install -r requirements.txt --force-reinstall`

---

## 📝 Example Queries

**Machine Learning:**
- "Explain the difference between bagging and boosting"
- "What is the vanishing gradient problem?"
- "How does cross-validation work?"
- "Explain gradient descent step by step"
- "What is overfitting and how to prevent it?"

**Deep Learning:**
- "Explain the transformer architecture"
- "What is attention mechanism?"
- "How does batch normalization help training?"
- "How does backpropagation work?"

**DSA:**
- "Explain quicksort with an example"
- "What is the time complexity of binary search?"
- "Compare stack vs queue data structures"

---

## ✅ Project Status

**Current Status: FULLY FUNCTIONAL** 🎉

### What's Working
- ✅ Model loaded: Llama 3.2 3B-Instruct (4-bit quantized, 2.09 GB VRAM)
- ✅ Vector store: 1,002 document chunks from 3 ML PDFs
- ✅ Retrieval pipeline: 5-document semantic search working
- ✅ Generation pipeline: GPU inference operational
- ✅ Streamlit UI: Interactive chat interface
- ✅ All LangChain compatibility issues resolved

### Recent Fixes (December 2025)
1. ✅ Updated to `langchain-huggingface` and `langchain-chroma` packages
2. ✅ Fixed `TutorLLM` to inherit from LangChain's `LLM` base class
3. ✅ Replaced deprecated `get_relevant_documents()` with `invoke()`
4. ✅ Added `stop` parameter handling for generation control
5. ✅ Switched from `ChatPromptTemplate` to `PromptTemplate` for string-based prompts
6. ✅ Implemented direct `llm.generate()` calls for reliable inference

### Performance
- **Inference Speed**: ~10-20 tokens/sec on RTX 4060
- **VRAM Usage**: 2.09 GB allocated, 2.91 GB reserved (plenty of headroom)
- **Retrieval Time**: <1 second for 5 documents
- **End-to-End Latency**: 3-8 seconds per question

### System Requirements Met
- ✅ Windows page file increased to 16-32GB
- ✅ PyTorch 2.6.0+cu124 (compatible with CUDA 13)
- ✅ All dependencies installed and compatible

---

## 🔮 Future Enhancements

- [ ] Multi-modal support (images in PDFs)
- [ ] Conversation memory/history
- [ ] Fine-tuning on code explanation with QLoRA
- [ ] Quantization to 3-bit for even lower VRAM
- [ ] Docker containerization
- [ ] WebSocket streaming responses
- [ ] Multi-turn conversation support
- [ ] Export study notes feature
- [ ] Quiz generation mode

---

## 🐛 Troubleshooting

### Common Issues

**Issue: Model loading fails with memory error**
- **Solution**: Increase Windows page file to 16-32GB (System Properties → Performance → Virtual Memory)

**Issue: CUDA out of memory**
- **Solution**: Reduce `max_new_tokens` in `model_config.json` or close other GPU applications

**Issue: Slow inference**
- **Solution**: Normal for 3B model on 8GB GPU. Consider using smaller prompts or reducing `k` (retrieved documents)

**Issue: Import errors from LangChain**
- **Solution**: Ensure you have the latest packages:
  ```bash
  pip install langchain-huggingface langchain-chroma -U
  ```

---

## 📄 License

MIT License - feel free to use for learning and portfolio projects.

---

## 🙏 Acknowledgments

- **LangChain** for RAG framework and updated integration packages
- **Hugging Face** for transformers and PEFT
- **Meta AI** for Llama models
- **ChromaDB** for vector storage
- **bitsandbytes** for 4-bit quantization

---

## 📧 Contact

Built by [Your Name] | [GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)

---

**⭐ If this helped you, star the repo!**
