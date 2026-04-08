import os
import sys
import torch
import numpy as np
import scipy.io.wavfile
from .base import BaseTTS
from voxcpm import VoxCPM

class voxcpm_agent(BaseTTS):

    def __init__(self, model_id_or_path, output_dir='outputs', device='cuda'):
        print('Initializing VoxCPM Agent...')
        self.device = device
        self.output_dir = output_dir
        self.all_speakers_info = {}
        try:
            self.tts_pipeline = VoxCPM.from_pretrained(model_id_or_path)
            if hasattr(self.tts_pipeline, 'tts_model'):
                self.tts_pipeline.tts_model = self.tts_pipeline.tts_model.to(self.device)
                print(f"VoxCPM internal model moved to '{self.device}'.")
            else:
                print("Warning: No 'tts_model' attribute on VoxCPM object; device placement skipped.")
            print(f"VoxCPM Agent '{model_id_or_path}' initialized successfully.")
        except Exception as e:
            print(f'Failed to initialize VoxCPM model: {e}')
            raise e

    def set_speaker_info(self, speaker_info_data):
        self.all_speakers_info = speaker_info_data

    def run_batch(self, texts: list[str], speaker_sequence: list[str], prompts: list[str], wavs: list[str], output_filename: str):
        print(f'--- Starting batch generation for {output_filename} ---')
        if not len(texts) == len(speaker_sequence):
            raise ValueError(f'The length of texts and speaker_sequence must be the same. Got {len(texts)} texts and {len(speaker_sequence)} speakers.')
        unique_speakers = sorted(list(set(speaker_sequence)))
        if len(unique_speakers) != len(prompts) or len(unique_speakers) != len(wavs):
            print(f'Warning: The number of unique speakers ({len(unique_speakers)}) does not match the number of prompts ({len(prompts)}) or wavs ({len(wavs)}). Ensure your main script prepares these correctly.')
        speaker_to_prompt_map = {name: {'prompt_text': p, 'wav_path': w} for name, p, w in zip(unique_speakers, prompts, wavs)}
        sample_rate = getattr(self.tts_pipeline.tts_model, 'sample_rate', 16000)
        segments_dir = os.path.join(self.output_dir, 'segments')
        os.makedirs(segments_dir, exist_ok=True)
        all_waveforms = []
        for i, text in enumerate(texts):
            speaker_name = speaker_sequence[i]
            if speaker_name not in speaker_to_prompt_map:
                print(f"Error: Speaker '{speaker_name}' not found in the provided prompt materials. Skipping this utterance.")
                continue
            prompt_info = speaker_to_prompt_map[speaker_name]
            prompt_text = prompt_info['prompt_text']
            prompt_wav_path = prompt_info['wav_path']
            print(f"Generating utterance {i + 1}/{len(texts)}: Speaker='{speaker_name}', Text='{text[:30]}...'")
            try:
                waveform = self.tts_pipeline.generate(text=text, prompt_wav_path=prompt_wav_path, prompt_text=prompt_text, cfg_value=1.75, inference_timesteps=10, normalize=True, denoise=False, retry_badcase=True, retry_badcase_max_times=3, retry_badcase_ratio_threshold=6.0)
                try:
                    segment_filename = f'{speaker_name}_text{i}_output.wav'
                    segment_path = os.path.join(segments_dir, segment_filename)
                    segment_waveform_int16 = (waveform * 32767).astype(np.int16)
                    scipy.io.wavfile.write(segment_path, rate=sample_rate, data=segment_waveform_int16)
                    print(f'  -> Segment saved to: {segment_path}')
                except Exception as save_e:
                    print(f'  -> Error saving segment for utterance {i + 1}: {save_e}')
                all_waveforms.append(waveform)
            except Exception as e:
                print(f"Error generating audio for speaker {speaker_name} with text '{text}': {e}")
                estimated_duration_ms = len(text) * 150
                silence = np.zeros(int(sample_rate * estimated_duration_ms / 1000), dtype=np.float32)
                all_waveforms.append(silence)
        if not all_waveforms or all((w.size == 0 for w in all_waveforms)):
            print('Warning: No valid waveforms were generated for this batch. Skipping concatenation.')
            return
        final_waveform = np.concatenate(all_waveforms)
        output_path = os.path.join(self.output_dir, output_filename)
        final_waveform_int16 = (final_waveform * 32767).astype(np.int16)
        scipy.io.wavfile.write(output_path, rate=sample_rate, data=final_waveform_int16)
        print(f'Batch generation complete. Final audio saved to: {output_path}')
