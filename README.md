# SpatioChat

Official repository for our **ACM MM 2026 Dataset Track** paper:  
**SpatioChat: A 3D Audio Dialogue Benchmark for Evaluating Spatial-Semantic Joint Reasoning**.

---

## Dataset (Hugging Face)

The SpatioChat corpus (multi-speaker dry speech, 8-channel spatial audio, scripts, QA, layout figures, RIR artifacts, etc.) is released separately on Hugging Face:

**[https://huggingface.co/datasets/NefIibata/SpatioChat](https://huggingface.co/datasets/NefIibata/SpatioChat)**

---

## Quick start: data generation pipeline

### 1. Prerequisites

- **OS**: Linux + **NVIDIA GPU** (CUDA) strongly recommended.
- **Python**: 3.10+.
- **API**: An OpenAI-compatible HTTP API (`base_url` + `api_key`) for the dialogue/QA LLM; for `enhanced` / `reflection`, an **audio-capable judge** model (`judge_model`).
- **Weights**: At minimum **VoxCPM** (all modes). **CosyVoice2** and **Fish-Speech** are required for multi-engine modes — install their extra dependencies per upstream docs.

### 2. Clone and environment

```bash
git clone https://github.com/Nefllbata/SpatioChat.git
cd SpatioChat

conda create -n spatiochat python=3.10 -y
conda activate spatiochat
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. CosyVoice2 (for `enhanced` / `reflection`)

The adapter [`DA/tts/cosyTTS.py`](DA/tts/cosyTTS.py) expects the CosyVoice source tree under:

`SpatioChat/DA/CosyVoice/`

Clone or symlink the official CosyVoice repo there, download **CosyVoice2** weights, and set `models.cosyvoice` in `config.json` to the checkpoint directory.

### 4. Fish-Speech (optional third engine)

Follow Fish-Speech install instructions, then set `models.fishaudio` in `config.json`.

### 5. Configure `config.json`

The repo ships a **template** [`config.json`](config.json) with placeholders. Edit it in place:

| Field | Meaning |
|-------|---------|
| `openai.api_key` | Your API key |
| `openai.base_url` | e.g. `https://api.openai.com/v1` or your gateway |
| `openai.model` | Text model for scripts + QA |
| `openai.judge_model` | **Required** for `enhanced` / `reflection` (audio scoring) |
| `models.voxcpm` | VoxCPM path or HF id (e.g. `openbmb/VoxCPM1.5`) |
| `models.cosyvoice` / `models.fishaudio` | Local paths to checkpoints |
| `tts_queue` | Ordered engines for battles, or `null` to auto-build from `models` |
| `data.*` | Paths to speaker JSON, scene list, IR directory (relative to project root) |

### 6. Run the pipeline

**Baseline** (VoxCPM only — smallest setup):

```bash
python demo.py --mode baseline --output_dir ./outputs --num_samples 1
```

**Enhanced** (multi-TTS + evaluation):

```bash
python demo.py --mode enhanced --output_dir ./outputs --num_samples 1
```

**Reflection** (iterative LLM + judge refinement):

```bash
python demo.py --mode reflection --output_dir ./outputs --num_samples 1 --max_loop 3
```

### 7. Typical outputs (per sample id under `output_dir/<id>/`)

- `<id>.json` — script  
- `<id>.wav` — rendered audio (mode-dependent)  
- `segments/` — per-utterance dry WAVs  
- `segments_8ch_direct/` — 8-channel spatialized segments  
- `RIR_8CH/` — RIR-related files  
- `<id>_QA.json` — generated QA  
- `<id>_layout.png` — spatial layout visualization  

---

## License

- **Code**: [MIT License](LICENSE).
- **Dataset**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
