import os
import sys
import time
import torch
import numpy as np
import scipy.io.wavfile
from pathlib import Path
from .base import BaseTTS
from fish_speech.models.text2semantic.inference import init_model, load_codec_model, encode_audio, decode_to_audio, generate_long

class fishaudio_agent(BaseTTS):

    def __init__(self, model_id_or_path, output_dir='outputs', device='cuda'):
        print('Initializing FishAudio Agent ...')
        self.device = device
        self.output_dir = output_dir
        self.model_path = Path(model_id_or_path)
        self.precision = torch.half
        try:
            print(' -> Loading Text2Semantic LLM ...')
            self.llm_model, self.decode_one_token = init_model(checkpoint_path=self.model_path, device=self.device, precision=self.precision, compile=True)
            with torch.device(self.device):
                self.llm_model.setup_caches(max_batch_size=1, max_seq_len=self.llm_model.config.max_seq_len, dtype=next(self.llm_model.parameters()).dtype)
            print(' -> Loading audio codec decoder ...')
            codec_checkpoint = self.model_path / 'codec.pth'
            self.codec_model = load_codec_model(codec_checkpoint, self.device, self.precision)
            self.sample_rate = self.codec_model.sample_rate
            self.speaker_cache = {}
            print(f"FishAudio Agent '{model_id_or_path}' initialized successfully.")
        except Exception as e:
            print(f'Failed to initialize FishAudio model: {e}')
            raise e

    def run_batch(self, texts: list[str], speaker_sequence: list[str], prompts: list[str], wavs: list[str], output_filename: str):
        print(f'\n--- Batch synthesis: {output_filename} ---')
        if not len(texts) == len(speaker_sequence):
            raise ValueError('texts and speaker_sequence must have the same length.')
        unique_speakers = sorted(list(set(speaker_sequence)))
        speaker_to_prompt_map = {name: {'prompt_text': p, 'wav_path': w} for name, p, w in zip(unique_speakers, prompts, wavs)}
        segments_dir = os.path.join(self.output_dir, 'segments')
        os.makedirs(segments_dir, exist_ok=True)
        all_waveforms = []
        all_filenames = []
        for i, text in enumerate(texts):
            speaker_name = speaker_sequence[i]
            segment_filename = f'{speaker_name}_text{i}_output.wav'
            segment_path = os.path.join(segments_dir, segment_filename)
            if speaker_name not in speaker_to_prompt_map:
                print(f"Error: No reference info for speaker '{speaker_name}'. Skipping this line.")
                continue
            prompt_info = speaker_to_prompt_map[speaker_name]
            prompt_text = prompt_info['prompt_text']
            prompt_wav_path = prompt_info['wav_path']
            print(f'[{i + 1}/{len(texts)}] Synthesizing | speaker: {speaker_name} | text: {text[:30]}...')
            if i == 0:
                print(' [Note] torch.compile is enabled: the first utterance may take 1–2 minutes to compile CUDA kernels; later lines are much faster.')
            try:
                if speaker_name not in self.speaker_cache:
                    print(f'   -> First use of {speaker_name}: extracting voice embedding ...')
                    prompt_tokens = encode_audio(prompt_wav_path, self.codec_model, self.device).cpu()
                    self.speaker_cache[speaker_name] = prompt_tokens
                else:
                    prompt_tokens = self.speaker_cache[speaker_name]
                generator = generate_long(model=self.llm_model, device=self.device, decode_one_token=self.decode_one_token, text=text, num_samples=1, max_new_tokens=0, top_p=0.9, top_k=30, temperature=1.0, compile=True, iterative_prompt=True, chunk_length=300, prompt_text=[prompt_text], prompt_tokens=[prompt_tokens])
                codes = []
                final_waveform = None
                for response in generator:
                    if response.action == 'sample':
                        codes.append(response.codes)
                    elif response.action == 'next':
                        if codes:
                            merged_codes = torch.cat(codes, dim=1)
                            audio_tensor = decode_to_audio(merged_codes.to(self.device), self.codec_model)
                            final_waveform = audio_tensor.cpu().float().numpy()
                        codes = []
                if final_waveform is None:
                    raise RuntimeError('Generation produced no valid audio codes.')
                final_waveform_clipped = np.clip(final_waveform, -1.0, 1.0)
                segment_waveform_int16 = (final_waveform_clipped * 32767).astype(np.int16)
                scipy.io.wavfile.write(segment_path, rate=self.sample_rate, data=segment_waveform_int16)
                print(f'   -> Segment saved: {segment_path}')
                all_waveforms.append(final_waveform)
                all_filenames.append(segment_path)
            except Exception as e:
                print(f'Error synthesizing audio (speaker {speaker_name}): {e}')
                estimated_duration_ms = len(text) * 150
                silence = np.zeros(int(self.sample_rate * estimated_duration_ms / 1000), dtype=np.float32)
                all_waveforms.append(silence)
                silence_int16 = (silence * 32767).astype(np.int16)
                scipy.io.wavfile.write(segment_path, rate=self.sample_rate, data=silence_int16)
                all_filenames.append(segment_path)
                print(f'   -> [Warning] Wrote silence placeholder: {segment_path}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not all_waveforms or all((w.size == 0 for w in all_waveforms)):
            print('Warning: No valid waveforms produced; skipping concatenation.')
            return
        concat_list = []
        for w in all_waveforms:
            concat_list.append(w)
            concat_list.append(np.zeros(int(self.sample_rate * 0.2), dtype=np.float32))
        full_conversation_waveform = np.concatenate(concat_list)
        output_path = os.path.join(self.output_dir, output_filename)
        full_waveform_clipped = np.clip(full_conversation_waveform, -1.0, 1.0)
        full_waveform_int16 = (full_waveform_clipped * 32767).astype(np.int16)
        scipy.io.wavfile.write(output_path, rate=self.sample_rate, data=full_waveform_int16)
        print(f'Batch synthesis done. Full dialogue saved to: {output_path}\n')
