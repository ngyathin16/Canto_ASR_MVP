# OPTIMIZATION PROTOCOL: Whisper Fine-Tuning Pipeline

**Role:** You are a Senior HPC Engineer specializing in PyTorch performance optimization, CUDA profiling, and ASR pipelines.

**Objective:** Scan the current codebase, identify performance bottlenecks in the TTS-based fine-tuning loop, instrument the code with profiling tools, and iteratively apply optimizations to maximize training throughput (samples/second).

**Tools Required:** `torch.profiler`, `nvtx`, `torch.utils.data.DataLoader`, `bitsandbytes`.

---

## Phase 1: Static Codebase Analysis (Deep Scan)
**Action:** Read `train.py`, the dataset class (e.g., `prepare_dataset.py` or where `__getitem__` is defined), and the configuration files.

**Analyze for the following specific "Anti-Patterns":**
1.  **On-the-fly TTS Generation:** Check if TTS audio is being generated *inside* the `__getitem__` method. (This is a critical blocking operation).
2.  **DataLoader Config:** Check `num_workers`, `pin_memory`, and `prefetch_factor`. Are they default (0) or optimized?
3.  **Tokenization:** Is tokenization happening on the GPU or CPU? Is it batched or serial?
4.  **VRAM Usage:** Are we using `fp16` / `bf16` and 8-bit quantization correctly?

**Output:** A bulleted list of "Suspected Bottlenecks" with a theoretical justification for each.

---

## Phase 2: Instrumentation (The "Proof")
**Action:** Modify `train.py` to include profiling hooks. Do NOT optimize yet; just measure.

1.  **Add NVTX Markers:**
    * Wrap the data loading step with `nvtx.range_push("DataLoading")` / `pop`.
    * Wrap the forward pass with `nvtx.range_push("Forward")` / `pop`.
    * Wrap the backward pass with `nvtx.range_push("Backward")` / `pop`.
    * *Justification:* This allows visualization of CPU-GPU separation in Nsight Systems.

2.  **Integrate Torch Profiler:**
    * Add a context manager `with torch.profiler.profile(...)` around the training loop for the first 10 steps.
    * Configure it to record `activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]`, `schedule`, `on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')`, and `record_shapes=True`.
    * *Justification:* This provides a trace file to definitively show if the GPU is starving (waiting for data).

---

## Phase 3: The Optimization Strategy (Execute Sequentially)
**Action:** Based on Phase 1 & 2 findings, apply the following optimizations in order of impact.

### Optimization A: De-couple TTS (The "Offline" Fix)
* **Condition:** If TTS is generating inside `__getitem__`.
* **Fix:** Refactor the pipeline to pre-generate all synthetic audio to disk (`.wav` files) *before* training starts.
* **Implementation:** Create a script `scripts/precompute_tts.py` to generate the dataset, saving audio paths to a JSON manifest. Change the Dataset class to load `.wav` files instead of calling the TTS API.
* **Justification:** TTS generation is 100x slower than model training. Moving it offline changes the task from "Compute Bound (CPU)" to "IO Bound", which is solvable.

### Optimization B: DataLoader Tuning
* **Condition:** If `num_workers=0` or `pin_memory=False`.
* **Fix:**
    * Set `num_workers = os.cpu_count() // 2`.
    * Set `pin_memory=True`.
    * Enable `persistent_workers=True` if using a loop.
* **Justification:** Pushes data loading to background processes, preventing the GPU from idling between batches.

### Optimization C: Memory Efficiency
* **Condition:** If VRAM allows, increase batch size.
* **Fix:**
    * Use `bitsandbytes` 8-bit optimizer if not already present.
    * Enable Gradient Checkpointing (`model.gradient_checkpointing_enable()`).
    * *Then* increase `per_device_train_batch_size` until VRAM is at ~90% utilization.
* **Justification:** Higher batch sizes reduce the overhead of python loops and kernel launches.

---

## Phase 4: Execution & Verification
**Action:**
1.  Apply the code changes for **Optimization A** and **B** immediately.
2.  Run the training loop for 50 steps.
3.  Report the "Seconds Per Iteration" (it/s) before and after the changes.

---

**Instruction to Windsurf:**
Start by performing **Phase 1 (Deep Scan)** now. Read the files, identify the TTS generation logic, and present your optimization plan before writing code.