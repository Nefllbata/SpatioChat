import os
import re
import json
import torch
import jiwer
import whisper
import numpy as np
import soundfile as sf
import librosa
from DA.utils import save_json
import random
from DA.prompt import *

class Evaluator:

    def __init__(self, llm_bot, audio_judges=None, device='cuda' if torch.cuda.is_available() else 'cpu', whisper_model='base'):
        self.llm = llm_bot
        if audio_judges is None:
            self.audio_judges = []
        elif not isinstance(audio_judges, list):
            self.audio_judges = [audio_judges]
        else:
            self.audio_judges = audio_judges
        self.device = device
        self.whisper_model_name = whisper_model or 'base'
        self.whisper_model = None
        self.utmos_model = None

    def _load_whisper(self):
        if self.whisper_model is None:
            print(f'Loading Whisper model for WER/CER calculation ({self.whisper_model_name})...')
            self.whisper_model = whisper.load_model(self.whisper_model_name, device=self.device)

    def _load_utmos(self):
        if self.utmos_model is None:
            print('Loading UTMOS model for Naturalness Prediction...')
            self.utmos_model = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)
            self.utmos_model.to(self.device)
            self.utmos_model.eval()

    def _extract_reference_text_with_prompts(self, script_content):
        turns = script_content.get('conversation', [])
        ref_text_parts = []
        for turn in turns:
            prompt_str = f"[{turn['prompt']}] " if 'prompt' in turn and turn['prompt'] else ''
            ref_text_parts.append(f"{prompt_str}{turn['text']}")
        return ' '.join(ref_text_parts)

    def compute_wer_cer(self, reference_text, audio_path):
        if not os.path.exists(audio_path):
            return {'wer': 1.0, 'cer': 1.0, 'transcription': ''}
        self._load_whisper()
        result = self.whisper_model.transcribe(audio_path)
        hypothesis_text = result['text']
        transformation = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
        clean_ref_text = re.sub('\\[.*?\\]\\s*', '', reference_text)
        ref_processed = transformation(clean_ref_text)
        hyp_processed = transformation(hypothesis_text)
        if not ref_processed:
            wer, cer = (0.0, 0.0)
        else:
            wer = jiwer.wer(ref_processed, hyp_processed)
            cer = jiwer.cer(ref_processed, hyp_processed)
        return {'wer': round(wer, 4), 'cer': round(cer, 4), 'transcription': hypothesis_text}

    def compute_utmos(self, target_path):
        if not os.path.exists(target_path):
            return 0.0
        try:
            self._load_utmos()
            if os.path.isdir(target_path):
                wav_files = [os.path.join(target_path, f) for f in os.listdir(target_path) if f.endswith('.wav')]
                if not wav_files:
                    return 0.0
                scores = []
                for wav_file in wav_files:
                    wave, sr = librosa.load(wav_file, sr=16000, mono=True)
                    wave_tensor = torch.from_numpy(wave).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        score = self.utmos_model(wave_tensor, sr)
                    scores.append(score.item())
                return round(float(np.mean(scores)), 4)
            else:
                wave, sr = librosa.load(target_path, sr=16000, mono=True)
                wave_tensor = torch.from_numpy(wave).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    score = self.utmos_model(wave_tensor, sr)
                return round(score.item(), 4)
        except Exception as e:
            print(f'Error computing UTMOS: {e}')
            return 0.0

    def eval_audio_scores_only(self, audio_path, script_text=''):
        if not self.audio_judges:
            return {'naturalness_score': 0.0, 'emotiveness_score': 0.0}
        nat_scores = []
        emo_scores = []
        for judge in self.audio_judges:
            judge_name = getattr(judge, 'model_name', 'Unknown-Audio-LLM')
            try:
                prompt = PROMPT_EXPERT_STEP1_SCORE.format(script_text=script_text)
                response = judge.evaluate_audio(audio_path, prompt)
                parsed = self._parse_llm_json(response)
                if 'naturalness_score' in parsed:
                    nat_scores.append(float(parsed['naturalness_score']))
                if 'emotiveness_score' in parsed:
                    emo_scores.append(float(parsed['emotiveness_score']))
            except Exception as e:
                print(f'Evaluation failed for {judge_name} (Scores only): {e}')
        return {'naturalness_score': round(np.mean(nat_scores), 2) if nat_scores else 0.0, 'emotiveness_score': round(np.mean(emo_scores), 2) if emo_scores else 0.0}

    def evaluate_dry_audio_for_battle(self, script_path, audio_path, segments_dir=None):
        results = {'utmos': 0.0, 'naturalness_score': 0.0, 'emotiveness_score': 0.0}
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_data = json.load(f)
            script_content = script_data.get('scripts', [script_data])[0]
        except Exception as e:
            print(f'Error loading script file: {e}')
            return results
        reference_text_with_prompts = self._extract_reference_text_with_prompts(script_content)
        utmos_target = segments_dir if segments_dir and os.path.exists(segments_dir) else audio_path
        results['utmos'] = self.compute_utmos(utmos_target)
        multi_scores = self.eval_audio_scores_only(audio_path, reference_text_with_prompts)
        results['naturalness_score'] = multi_scores.get('naturalness_score', 0.0)
        results['emotiveness_score'] = multi_scores.get('emotiveness_score', 0.0)
        return results

    def eval_audio_with_judges(self, audio_path, script_text=''):
        if not self.audio_judges:
            return {'naturalness_score': 0.0, 'emotiveness_score': 0.0, 'advice': '', 'text': ''}
        nat_scores = []
        emo_scores = []
        advices = []
        raw_texts = []
        for judge in self.audio_judges:
            judge_name = getattr(judge, 'model_name', 'Unknown-Audio-LLM')
            try:
                prompt = PROMPT_EXPERT_AUDIO_EVAL.format(script_text=script_text)
                response = judge.evaluate_audio(audio_path, prompt)
                raw_texts.append(f'--- {judge_name} RAW OUTPUT ---\n{response}')
                parsed = self._parse_llm_json(response)
                if 'naturalness_score' in parsed:
                    nat_scores.append(float(parsed['naturalness_score']))
                if 'emotiveness_score' in parsed:
                    emo_scores.append(float(parsed['emotiveness_score']))
                if 'advice' in parsed and parsed['advice']:
                    advices.append(parsed['advice'])
            except Exception as e:
                error_msg = f'Evaluation failed for {judge_name}: {e}'
                print(error_msg)
                raw_texts.append(error_msg)
        return {'naturalness_score': round(np.mean(nat_scores), 2) if nat_scores else 0.0, 'emotiveness_score': round(np.mean(emo_scores), 2) if emo_scores else 0.0, 'advice': random.choice(advices) if advices else 'No advice available.', 'evaluated_by_count': len(nat_scores), 'text': '\n\n'.join(raw_texts)}

    def _parse_llm_json(self, response_str):
        original_str = response_str
        response_str = re.sub('<think>.*?</think>', '', response_str, flags=re.DOTALL).strip()
        if '```json' in response_str:
            response_str = response_str.split('```json')[1].split('```')[0]
        elif '```vbnet' in response_str:
            response_str = response_str.split('```vbnet')[1].split('```')[0]
        elif '```' in response_str:
            response_str = response_str.split('```')[1].split('```')[0]
        try:
            return json.loads(response_str.strip())
        except json.JSONDecodeError:
            print(f'Failed to parse strict JSON from evaluation response. Attempting regex extraction...')
            fallback_dict = {'analysis': 'Failed to parse analysis properly.', 'naturalness_score': 0.0, 'emotiveness_score': 0.0, 'advice': 'Failed to extract advice.'}
            nat_match = re.search('naturalness[\\s_]*score[\\s:"]*([\\d\\.]+)', original_str, re.IGNORECASE)
            if nat_match:
                fallback_dict['naturalness_score'] = float(nat_match.group(1))
            emo_match = re.search('emotiveness[\\s_]*score[\\s:"]*([\\d\\.]+)', original_str, re.IGNORECASE)
            if emo_match:
                fallback_dict['emotiveness_score'] = float(emo_match.group(1))
            advice_match = re.search('(?:actionable\\s*)?advice[\\s:"]*(.*)', original_str, re.IGNORECASE | re.DOTALL)
            if advice_match:
                fallback_dict['advice'] = advice_match.group(1).strip(' \n\r"')
            analysis_match = re.split('naturalness[\\s_]*score', original_str, flags=re.IGNORECASE)
            if len(analysis_match) > 1:
                fallback_dict['analysis'] = analysis_match[0].replace('analysis:', '').replace('"analysis":', '').strip(' {\n\r"')
            print(f'Regex extracted: {fallback_dict}')
            return fallback_dict

    def run_comprehensive_evaluation(self, script_path, audio_path, qa_path, output_dir, segments_dir=None):
        print('\n--- Starting Comprehensive Evaluation ---')
        results = {}
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_data = json.load(f)
            script_content = script_data.get('scripts', [script_data])[0]
            qa_data = None
            if os.path.exists(qa_path):
                with open(qa_path, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
        except Exception as e:
            print(f'Error loading files: {e}')
            return results
        reference_text_with_prompts = self._extract_reference_text_with_prompts(script_content)
        results['objective_metrics'] = self.compute_wer_cer(reference_text_with_prompts, audio_path)
        utmos_target = segments_dir if segments_dir and os.path.exists(segments_dir) else audio_path
        results['utmos_score'] = self.compute_utmos(utmos_target)
        if self.audio_judges:
            print(f'Evaluating Audio with {len(self.audio_judges)} Audio LLM Judge(s)...')
            results['expert_eval'] = self.eval_audio_with_judges(audio_path, reference_text_with_prompts)
        save_path = os.path.join(output_dir, 'evaluation_report.json')
        save_json(results, save_path)
        print(f'Evaluation finished. Results saved to {save_path}')
        return results
