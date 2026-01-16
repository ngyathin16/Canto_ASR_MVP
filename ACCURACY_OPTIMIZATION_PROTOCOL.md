# ACCURACY_OPTIMIZATION_PROTOCOL.md

**Role:** You are a Senior ASR Research Engineer specializing in Low-Resource Language Adaptation and Whisper fine-tuning.

**Context:**
The user is fine-tuning a Cantonese Whisper model using **TTS (Text-to-Speech) generated audio**.
* **Current Status:** CER improved from ~18.38% (Base) to ~14.59% (Fine-Tuned).
* **Goal:** Drastically lower CER (<10%) and fix specific "hallucination/repetition" errors observed in the logs (e.g., repeated phrases like "安全主任...").
* **Observed Issues:** High token-count jargon (High Risk terms) is being mis-transcribed; Model is hallucinating during silence; Potential mismatch in base models.

**Task:** Perform a comprehensive audit of the code and apply the following 4-Phase Optimization Strategy.

---

## Phase 1: The "Model Identity" Audit (CRITICAL)
**Logic:** The provided evaluation log uses `khleeloo/whisper-large-v3-cantonese`, but the user's strategy document specifies `JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english`. Training on one and evaluating on another invalidates the CER metrics.

**Action Steps for Windsurf:**
1.  **Check `train.py` vs `evaluate_model.py`:** Confirm the `model_id` is identical in both files.
2.  **Recommendation:** Switch strictly to `JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english`. It is a distilled model specifically optimized for Hong Kong Cantonese (Yue), whereas `khleeloo` is likely general Cantonese.
3.  **Tokenizer Check:** Ensure `tokenizer.language` is explicitly forced to `"yue"` (not `"zh"`) in the training loop to prevent Mandarin token leakage.

---

## Phase 2: Data Engineering (Fixing TTS Artifacts)
**Logic:** TTS audio is "too clean." Whisper overfits to the perfect digital silence and robotic prosody, causing it to fail or hallucinate (repeat phrases) when it encounters slight variations.

**Action Steps for Windsurf:**
1.  **Implement "Digital Dirt" (Augmentation):**
    * Modify the dataset loading script to apply **Gaussian Noise** or **Reverb** to the TTS audio on-the-fly.
    * *Target SNR:* Mix TTS with background noise at 20dB–30dB SNR. This forces the model to focus on the *phonemes*, not the silence.
2.  **Silence Trimming:**
    * TTS engines often leave 1-2 seconds of absolute silence at the start/end. Whisper treats long silence as a trigger to hallucinate (repeat previous text).
    * *Code Action:* Apply `librosa.effects.trim` to all TTS clips before tokenization.
3.  **Text Normalization (The Glossary Enforcement):**
    * The evaluation shows "芝士巴拿" (Cheese Spanner) vs "士Banne...".
    * *Code Action:* Ensure the training labels for "Spanner" are standardized to "士巴拿" (or the preferred English spelling) in the training CSV. Inconsistent labels confuse the model.

---

## Phase 3: LoRA Hyperparameter Tuning (Capacity Injection)
**Logic:** The "High Risk" terms (e.g., "工程公司術語" = 7 tokens) require more model capacity to memorize than standard vocabulary. Default LoRA settings are often too shallow.

**Action Steps for Windsurf:**
1.  **Expand Target Modules:**
    * *Current Guess:* You are likely targeting only `["q_proj", "v_proj"]`.
    * *Optimization:* Change `LoraConfig` to target **all linear layers**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
    * *Why:* This allows the model to modify the Feed-Forward Networks (FFN), where factual knowledge (vocabulary) is stored.
2.  **Increase Rank (r):**
    * Increase LoRA `r` from 16/32 to **64**.
    * Increase `lora_alpha` to **128**.
3.  **Disable "Condition on Previous Text":**
    * In `training_args`, ensure `condition_on_prev_tokens=False`. This prevents the "hallucination loop" where the model repeats the same sentence twice.

---

## Phase 4: Inference Optimization (The "Free Lunch")
**Logic:** You can improve CER without retraining by guiding the model during generation.

**Action Steps for Windsurf:**
1.  **Inject `initial_prompt`:**
    * Modify `evaluate_model.py` to accept a string of your top 20 jargon terms.
    * Pass this string to the `model.generate(..., initial_prompt="狗臂架, 趷雞陣, 孖葉...")` function.
    * *Impact:* This primes the model's context window, significantly lowering CER for rare words.
2.  **Beam Search:**
    * Update evaluation generation config to use `num_beams=5`. This trades speed for accuracy.

---

**Execution Order:**
1.  **Audit:** specific model IDs.
2.  **Code:** Implement Silence Trimming & Noise Augmentation (Phase 2).
3.  **Config:** Update LoRA target modules (Phase 3).
4.  **Eval:** Update evaluation script to use `initial_prompt` (Phase 4).

**Start by analyzing `train.py` and `evaluate_model.py` for Phase 1.**