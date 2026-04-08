import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cosyvoice_root_path = os.path.join(PROJECT_ROOT, 'DA', 'CosyVoice')
if cosyvoice_root_path not in sys.path:
    sys.path.insert(0, cosyvoice_root_path)
matcha_tts_path = os.path.join(cosyvoice_root_path, 'third_party', 'Matcha-TTS')
if matcha_tts_path not in sys.path:
    sys.path.insert(0, matcha_tts_path)
from .base import BaseTTS
import torchaudio
import subprocess
import torch
from cosyvoice.cli.cosyvoice import AutoModel

def _get_audio(wav_obj):
    if hasattr(wav_obj, '__iter__') and (not isinstance(wav_obj, dict)):
        wav_obj = next(iter(wav_obj))
    if not isinstance(wav_obj, dict) or 'tts_speech' not in wav_obj:
        raise TypeError("Unexpected CosyVoice output, expect dict with key 'tts_speech'")
    audio = wav_obj['tts_speech']
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    return audio

class cosyvoice_agent(BaseTTS):

    def __init__(self, model_path=None, output_dir='outputs', device='cuda'):
        if model_path is None:
            raise ValueError('model_path cannot be None')
        try:
            print(f"Loading CosyVoice model via AutoModel from '{model_path}'...")
            self.chat = AutoModel(model_dir=model_path)
            print('Model loaded successfully.')
        except Exception as e:
            print(f"Failed to initialize CosyVoice model '{model_path}': {e}")
            sys.exit(1)
        self.output_dir = output_dir

    def save_wav(self, wav, prefix):
        filename = f'{prefix}_output.wav'
        filepath = os.path.join(self.temp_dir, filename)
        audio = _get_audio(wav)
        torchaudio.save(filepath, audio.to(torch.float32).cpu(), 22050)
        print(f'Saved {filepath}')
        return filepath

    def concatenate_wavs(self, wav_files, output_filename):
        with open(self.concat_file, 'w', encoding='utf-8') as f:
            for i in wav_files:
                f.write(f"file '{os.path.abspath(i)}'\n")
        output_path = os.path.join(self.help_dir, output_filename)
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', self.concat_file, '-c', 'copy', output_path], check=True, capture_output=True)
        print(f'Concatenated audio saved at {output_path}')

    def run_batch(self, text_batches, speakers, prompts, wavs, output_filename, emotion_instructs=None):
        self.help_dir = os.path.join(self.output_dir, output_filename.split('.')[0])
        self.temp_dir = os.path.join(self.help_dir, 'segments')
        self.concat_file = os.path.join(self.help_dir, 'concat.txt')
        os.makedirs(self.temp_dir, exist_ok=True)
        all_filenames = []
        if emotion_instructs is None:
            emotion_instructs = [None] * len(text_batches)
        args = zip(text_batches, speakers, prompts, wavs, emotion_instructs)
        for i, (text, speaker_label, ref_text, ref_wav, emotion_inst) in enumerate(args):
            print(f'Synthesizing segment {i} for speaker: {speaker_label}...')
            try:
                if emotion_inst and str(emotion_inst).strip():
                    instruct_text = str(emotion_inst).strip()
                    if not instruct_text.endswith('<|endofprompt|>'):
                        instruct_text += '<|endofprompt|>'
                    print(f'  -> Using Instruct Mode with prompt: {instruct_text}')
                    wav_generator = self.chat.inference_instruct2(text, instruct_text, ref_wav, stream=False)
                else:
                    safe_ref_text = str(ref_text).strip()
                    print(f'  -> Using Zero-Shot Mode Reference Text: {safe_ref_text}')
                    wav_generator = self.chat.inference_zero_shot(text, safe_ref_text, ref_wav, stream=False)
                wav_output = next(wav_generator) if hasattr(wav_generator, '__iter__') else wav_generator[0]
                filename = self.save_wav(wav_output, f'{speaker_label}_text{i}')
                all_filenames.append(filename)
            except Exception as e:
                print(f'[ERROR] TTS model failed to produce output for segment {i}. Error: {e}')
                continue
        if all_filenames:
            self.concatenate_wavs(all_filenames, output_filename)
