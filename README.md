# SpatioChat

Official repository for **ACM MM 2026 Dataset Track** paper:  
**SpatioChat: A 3D Audio Dialogue Benchmark for Evaluating Spatial-Semantic Joint Reasoning**.

**[Notice for reviewers]** The full dataset, code generation pipeline, and supplementary materials are being organized and will be available by the supplemental materials deadline. This repository contains the **end-to-end data generation pipeline** (LLM script → multi-speaker TTS → 8-channel spatialization → QA generation → optional evaluation).

- **Code** in this repo: **MIT License** (see [`LICENSE`](LICENSE)).
- **SpatioChat dataset** (audio + annotations): **CC BY 4.0** (see dataset card on Hugging Face).

---

## Repository overview

| Item | Description |
|------|-------------|
| [`demo.py`](demo.py) | Single entrypoint: `baseline`, `enhanced`, `reflection` modes |
| [`config.example.json`](config.example.json) | Template configuration (copy to `config.json`) |
| [`DA/`](DA/) | LLM client, prompts, TTS adapters (VoxCPM / CosyVoice2 / FishAudio), spatializer, evaluator, utils |
| [`data/`](data/) | Minimal speaker metadata, scene list, IR classification index (add your own `data/IR/*.wav` for environmental reverb) |
| **Dataset** | Not shipped in GitHub (large size). Host on Hugging Face or similar; see **Dataset** below. |

---

## Prerequisites

- **OS**: Linux recommended (GPU + CUDA).
- **Python**: 3.10+ (tested with conda environments).
- **GPU**: NVIDIA GPU with CUDA for TTS and `gpuRIR` spatialization.
- **API access**: An OpenAI-compatible HTTP API (`base_url` + `api_key`) for the **chat model** and, for `enhanced` / `reflection`, an **audio-capable judge model** (`judge_model`).
- **Models** (local paths or Hugging Face IDs where supported):
  - **VoxCPM** — required for **all** modes as the baseline dry-speech engine.
  - **CosyVoice2** — required for `enhanced` / `reflection` multi-engine paths.
  - **Fish-Speech** — optional third engine in the same modes.

---

## Quick start (full pipeline)

### 1. Clone

```bash
git clone https://github.com/Nefllbata/SpatioChat.git
cd SpatioChat
```

### 2. Create environment

```bash
conda create -n spatiochat python=3.10 -y
conda activate spatiochat
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` includes core deps (`torch`, `voxcpm`, `gpuRIR`, `openai-whisper`, etc.).  
You may need **extra** packages for CosyVoice2 / Fish-Speech following their upstream install guides once weights are placed on disk.

### 3. CosyVoice2 source tree (for `enhanced` / `reflection`)

The adapter [`DA/tts/cosyTTS.py`](DA/tts/cosyTTS.py) expects a **CosyVoice** checkout under:

`SpatioChat/DA/CosyVoice/`

Place the upstream CosyVoice repository (or symlink) there, download **CosyVoice2** weights, and set `models.cosyvoice` in config to that checkpoint directory.

### 4. Fish-Speech (optional)

Install the `fish-speech` stack as required by [`DA/tts/fishaudioTTS.py`](DA/tts/fishaudioTTS.py) and point `models.fishaudio` to your checkpoint.

### 5. Configuration

```bash
cp config.example.json config.json
# Edit config.json — do NOT commit real API keys
```

Fill in at minimum:

| Key | Purpose |
|-----|---------|
| `openai.api_key` | API token |
| `openai.base_url` | OpenAI-compatible endpoint (e.g. `https://api.openai.com/v1`) |
| `openai.model` | Text LLM for script + QA |
| `openai.judge_model` | Required for `enhanced` / `reflection` (audio evaluation) |
| `models.voxcpm` | VoxCPM path or HF id (e.g. `openbmb/VoxCPM1.5`) |
| `models.cosyvoice` / `models.fishaudio` | Paths to checkpoints when using those engines |
| `tts_queue` | Ordered list of engines for battles; or `null` to auto-build from `models` |

Paths under `data.*` are relative to the project root unless absolute.

### 6. Run

**Baseline** (VoxCPM only — smallest footprint):

```bash
python demo.py --mode baseline --output_dir ./outputs --num_samples 1
```

**Enhanced** (multi-TTS + evaluation):

```bash
python demo.py --mode enhanced --output_dir ./outputs --num_samples 1
```

**Reflection** (iterative refinement with LLM + judges):

```bash
python demo.py --mode reflection --output_dir ./outputs --num_samples 1 --max_loop 3
```

Custom config path:

```bash
python demo.py --mode baseline --output_dir ./outputs --config /path/to/my_config.json
```

### 7. Outputs (per generated sample id)

Under `output_dir/<sample_id>/` you typically get:

- `<id>.json` — dialogue + spatial metadata  
- `<id>.wav` — full dry or mixed audio (mode-dependent)  
- `segments/` — per-utterance dry WAVs  
- `segments_8ch_direct/` (and optionally mixed paths) — spatialized 8-ch audio  
- `RIR_8CH/` — room impulse related artifacts  
- `<id>_QA.json` — generated QA  
- `<id>_layout.png` — spatial layout plot  

Exact files depend on mode and configuration.

---

## Dataset (Hugging Face)

The **SpatioChat** corpus (multi-channel audio, scripts, QA, etc.) is distributed separately from this GitHub repo, e.g. via Hugging Face Datasets.  
**Placeholder:** add your public dataset URL here after publication, for example:

`https://huggingface.co/datasets/<YOUR_ORG>/SpatioChat`

A small toy sample may be linked here for quick sanity checks.

---

## Evaluation notes

- **Subjective scores** (naturalness / emotiveness): from the configured **audio judge** (`judge_model`).
- **WER / CER**: local **Whisper** transcription vs. reference text (default Whisper size in code: `base`); no extra key in `config`.

---

## IR / reverberation

- `data/IR/classifications.txt` maps scene IDs to labels.  
- Place matching **`{scene_id}_*_img.wav`** files under `data/IR/` to enable **environmental IR** mixing.  
- If no file matches, the pipeline continues with **directional RIR only** (no environmental reverb).

---

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| `Config not found` | `cp config.example.json config.json` |
| VoxCPM cannot load path | `models.voxcpm` must be a **VoxCPM** checkpoint, not CosyVoice |
| CosyVoice import errors | Ensure `DA/CosyVoice` exists and matches upstream layout |
| CUDA OOM | Reduce batch sizes / use smaller models / one engine at a time |
| HF upload SSL / slow | Use `hf_transfer`, stable proxy, or `upload_large_folder` with fewer workers |

---

## Citation

If you use this benchmark or dataset, please cite the ACM MM 2026 Dataset Track paper (bibtex will be added upon publication).

---

## Acknowledgments

Pipeline evolved from internal **DialogueAgents** experiments; released here in minimal form for reproducibility.
