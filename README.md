# ⚡ APTOS - Auto-Parallelization and Task-Offloading System

**APTOS** is a lightweight Python framework that automatically distributes computational workloads across CPU-bound tasks, simulated GPU operations, and deep learning models using intelligent offloading and parallel execution strategies.

It simulates a heterogeneous computing environment without requiring real GPUs or cluster infrastructure, making it ideal for prototyping, benchmarking, and learning parallel computing concepts.

---

## 📌 What It Is

APTOS is a **self-contained compute orchestrator**, built with:

- ✅ CPU-bound task execution using NumPy
- ✅ Simulated GPU task routing (fallback to CPU)
- ✅ Deep learning support via TensorFlow (CPU-based)
- ✅ Multi-threaded and multi-process execution using `ThreadPoolExecutor` and `ProcessPoolExecutor`
- ✅ Intelligent task type detection and offloading logic

---

## 📊 What It Does

- 🧠 Classifies and routes tasks (CPU / GPU / DL) intelligently
- 🔄 Executes them in parallel based on task type
- 💡 Simulates real GPU offloading using CPU (optional CuPy support can be added)
- 📈 Measures execution time and performance
- 🔧 Provides a clean base to extend into cloud offloading, real GPU tasks, or edge computing

---

### 📈 Sample Result Output (from `app.py`)

```bash
Using CPU for the task
CPU Task Completed in 0.0115 seconds.

Simulated GPU task using CPU (no GPU available)
Simulated GPU Task Completed in 11.3342 seconds.

Using TensorFlow for deep learning task
Deep Learning Task Completed in 2.7757 seconds.

---

## 📉 Performance Metrics Tracked
- 🕒 Encoding / Decoding Time
- 🔢 Task-specific execution durations
- 🧮 Reconstruction quality (for ML tasks, if extended)
- 📊 Optional: CPU vs Simulated GPU vs DL benchmarking

---

## 🚀 Future Enhancements
- Real GPU task support (e.g., CuPy / CUDA integration)
- Cloud or edge offloading via sockets or APIs
- Real-time profiling and adaptive task scheduling
- Task prioritization and queuing system
- Web-based dashboard for monitoring

---

## 💻 How to Clone and Run

git clone https://github.com/<your-username>/APTOS-Auto-Parallel-Offloading-System.git

cd APTOS-Auto-Parallel-Offloading-System

pip install -r requirements.txt

python app.py

```
