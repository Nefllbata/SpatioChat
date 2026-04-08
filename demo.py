import glob
import os
import gc
import argparse
import json
import random
import string
import time
import shutil
import torch
import logging
import re
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
from collections import defaultdict
from DA.llm.openai import OpenAILLM
from DA.prompt import *
from DA.utils import get_script, get_speaker, get_text_inside_tag, read_json_file, save_json, load_weighted_scenes
from DA.evaluator import Evaluator
import gpuRIR as gr
from DA.tts.voxcpmTTS import voxcpm_agent
from DA.audio.spatializer import spatialize_and_mix_voxcpm
from DA.visualization.plotter import plot_spatial_layout
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.json')

def resolve_engine_class(name):
    if name == 'VoxCPM':
        return voxcpm_agent
    if name == 'CosyVoice2':
        from DA.tts.cosyTTS import cosyvoice_agent
        return cosyvoice_agent
    if name == 'FishAudio':
        from DA.tts.fishaudioTTS import fishaudio_agent
        return fishaudio_agent
    raise ValueError(f'Unknown TTS engine: {name}')

def resolve_path(p):
    if not p:
        return p
    if os.path.isabs(p):
        return p
    candidate = os.path.join(PROJECT_ROOT, p)
    if os.path.exists(candidate):
        return candidate
    return p

def load_config(config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Config not found: {config_path}. Copy config.example.json to config.json and edit.')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def data_paths(cfg):
    d = cfg.get('data', {})
    return (resolve_path(d.get('speaker_json', 'data/speaker/myspeaker.json')), resolve_path(d.get('scene_file', 'data/IR/classifications.txt')), resolve_path(d.get('ir_dir', 'data/IR')))

def voxcpm_model_path(cfg):
    m = cfg.get('models', {})
    p = m.get('voxcpm')
    if not p:
        for item in cfg.get('tts_queue') or []:
            if item.get('name') == 'VoxCPM' and item.get('path'):
                p = item['path']
                break
    if not p:
        raise ValueError('Set config.models.voxcpm or a VoxCPM entry in config.tts_queue (baseline uses VoxCPM)')
    return resolve_path(p)

def build_tts_queue(cfg):
    spec = cfg.get('tts_queue')
    if spec:
        out = []
        for item in spec:
            name = item.get('name')
            path = item.get('path')
            if not path:
                continue
            cls = resolve_engine_class(name)
            out.append({'name': name, 'class': cls, 'path': resolve_path(path)})
        if out:
            return out
    models = cfg.get('models', {})
    out = []
    order = [('VoxCPM', 'voxcpm'), ('CosyVoice2', 'cosyvoice'), ('FishAudio', 'fishaudio')]
    for name, key in order:
        path = models.get(key)
        if path:
            out.append({'name': name, 'class': resolve_engine_class(name), 'path': resolve_path(path)})
    if not out:
        raise ValueError('No TTS engines configured: set config.models or config.tts_queue')
    return out

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='baseline | enhanced | reflection')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--max_loop', type=int, default=1, help='Reflection mode max loops')
    parser.add_argument('--num_samples', type=int, default=1, help='Batch samples')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='Path to config JSON')
    return parser.parse_args()

def initialize_tts(model_id, output_dir):
    return voxcpm_agent(model_id, output_dir)

def initialize_llm(cfg):
    o = cfg.get('openai', {})
    api_key = o.get('api_key')
    model = o.get('model')
    if not api_key or not model:
        raise ValueError('config.openai.api_key and config.openai.model are required')
    return OpenAILLM(model_name=model, api_key=api_key, base_url=o.get('base_url'), timeout=float(o.get('timeout', 120.0)), max_retries=int(o.get('max_retries', 3)))

def initialize_api_judges(cfg):
    o = cfg.get('openai', {})
    api_key = o.get('api_key')
    judge_model = o.get('judge_model')
    if not judge_model:
        raise ValueError('config.openai.judge_model is required for enhanced/reflection')
    print('Initializing API Audio Judges...')
    gemini_judge = OpenAILLM(model_name=judge_model, api_key=api_key, base_url=o.get('base_url'), timeout=float(o.get('judge_timeout', 180.0)), max_retries=int(o.get('max_retries', 3)))
    return [gemini_judge]

def process_baseline_mode(llm_bot, output_dir, cfg):
    print('\n=== Running in BASELINE Mode ===')
    random.seed(time.time())
    speaker_json_path, scene_file_path, ir_base_dir = data_paths(cfg)
    model_id = voxcpm_model_path(cfg)
    all_characters_data = read_json_file(speaker_json_path)
    all_speaker_names = list(all_characters_data.keys())
    num_speakers = random.choices([2, 3, 4], weights=[0.6, 0.2, 0.2], k=1)[0]
    selected_speaker_names = random.sample(all_speaker_names, num_speakers)
    scene_ids, scenes_chinese, scenes_english, weights = load_weighted_scenes(scene_file_path)
    if not scenes_english:
        print('Error: Could not load scenes from file. Using built-in fallback scene list and empty IR path.')
        scenes_list = ['Dining Room', 'Living Room', 'Kitchen', 'Bedroom', 'Balcony', 'Cafe', 'Park', 'Community Center', 'Office', 'Meeting Room']
        selected_scene = random.choice(scenes_list)
        env_ir_path = ''
    else:
        indices = list(range(len(scenes_english)))
        selected_index = random.choices(indices, weights=weights, k=1)[0]
        selected_scene_id = scene_ids[selected_index]
        selected_scene = scenes_english[selected_index]
        try:
            search_pattern = os.path.join(ir_base_dir, f'{selected_scene_id}_*_img.wav')
            matching_ir_files = glob.glob(search_pattern)
            if not matching_ir_files:
                print(f'Warning: Scene ID {selected_scene_id} found but no matching IR WAVs under {ir_base_dir}. Spatialization will run without environmental reverb.')
                env_ir_path = ''
            else:
                env_ir_path = random.choice(matching_ir_files)
                print(f'Selected scene ID {selected_scene_id}. Found {len(matching_ir_files)} IRs. Randomly selected: {env_ir_path}')
        except Exception as e:
            print(f'Error during IR file search: {e}')
            env_ir_path = ''
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    print(f'Generated unique ID for this run: {random_id}')
    run_dir = os.path.join(output_dir, random_id)
    os.makedirs(run_dir, exist_ok=True)
    character_cards = {name: {key: value for key, value in all_characters_data[name].items() if key != 'wav'} for name in selected_speaker_names}
    print(f"Selected {num_speakers} speakers: {', '.join(selected_speaker_names)}")
    print(f'Selected scene: {selected_scene}')
    character_cards_json_str = json.dumps(character_cards, indent=4, ensure_ascii=False)
    prompt = SCRIPT_MIXED.format(character_cards=character_cards_json_str, scene_key=selected_scene)
    llm_response_str = llm_bot.predict(prompt)
    if llm_response_str.startswith('```json'):
        llm_response_str = llm_response_str[7:]
    if llm_response_str.endswith('```'):
        llm_response_str = llm_response_str[:-3]
    try:
        generated_script_data = json.loads(llm_response_str)
        script_save_path = os.path.join(run_dir, f'{random_id}.json')
        save_json(generated_script_data, script_save_path)
        print(f'Script successfully generated and saved to {script_save_path}')
        plot_save_path = os.path.join(run_dir, f'{random_id}_layout.png')
        plot_spatial_layout(generated_script_data['scripts'][0], plot_save_path)
    except json.JSONDecodeError as e:
        print(f'Error: Failed to decode LLM response into JSON. Error: {e}')
        print(f'LLM Response:\n{llm_response_str}')
        return
    print('\n--- Starting TTS Synthesis ---')
    tts = initialize_tts(model_id, run_dir)
    tts.set_speaker_info(all_characters_data)
    text_batches, speaker_sequences = get_script(script_save_path)
    for i, text_batch in enumerate(text_batches):
        speaker_sequence = speaker_sequences[i]
        unique_speakers = sorted(list(set(speaker_sequence)))
        prompts_for_batch = []
        wavs_for_batch = []
        for speaker_name in unique_speakers:
            _, prompt_text, wav_path = get_speaker(speaker_name, all_characters_data)
            prompts_for_batch.append(prompt_text)
            wavs_for_batch.append(wav_path)
        processed_texts = [re.sub('\\[.*?\\]\\s*', '', t).strip() for t in text_batch]
        tts.run_batch(texts=processed_texts, speaker_sequence=speaker_sequence, prompts=prompts_for_batch, wavs=wavs_for_batch, output_filename=f'{random_id}_{i}.wav')
    speaker_info_path = speaker_json_path
    spatialize_and_mix_voxcpm(run_dir, random_id, speaker_info_path, env_ir_path)
    print('\n--- Starting QA Generation ---')
    try:
        script_data = generated_script_data['scripts'][0]
        clean_conversation = []
        for turn in script_data.get('conversation', []):
            clean_conversation.append({k: v for k, v in turn.items() if k != 'prompt'})
        conversation_script_json = json.dumps(clean_conversation, indent=4, ensure_ascii=False)
        spatial_annotation = {'roomsize': script_data.get('roomsize'), 'listener': script_data.get('listener') or script_data.get('listener_pos'), 'speaker': script_data.get('speaker')}
        spatial_annotation_json = json.dumps(spatial_annotation, indent=4, ensure_ascii=False)
        final_qa_prompt = QA_MIXED.format(conversation_script=conversation_script_json, spatial_annotation=spatial_annotation_json)
        qa_response_str = llm_bot.predict(final_qa_prompt)
        if qa_response_str.startswith('```json'):
            qa_response_str = qa_response_str[7:]
        if qa_response_str.endswith('```'):
            qa_response_str = qa_response_str[:-3]
        qa_data = json.loads(qa_response_str)
        qa_save_path = os.path.join(run_dir, f'{random_id}_QA.json')
        save_json(qa_data, qa_save_path)
        print(f'QA set successfully generated and saved to {qa_save_path}')
    except json.JSONDecodeError as e:
        print(f'Error: Failed to decode LLM response for QA into JSON. Error: {e}')
    except Exception as e:
        print(f'An error occurred during QA generation: {e}')
    print('\n--- Baseline Mode Finished ---')

def process_enhanced_mode(llm_bot, output_dir, cfg):
    print('\n=== Running in ENHANCED Mode (Dynamic Complete Script Battle) ===')
    random.seed(time.time())
    speaker_json_path, scene_file_path, ir_base_dir = data_paths(cfg)
    all_characters_data = read_json_file(speaker_json_path)
    all_speaker_names = list(all_characters_data.keys())
    num_speakers = random.choices([2, 3, 4], weights=[0.6, 0.2, 0.2], k=1)[0]
    selected_speaker_names = random.sample(all_speaker_names, num_speakers)
    scene_ids, scenes_chinese, scenes_english, weights = load_weighted_scenes(scene_file_path)
    if not scenes_english:
        print('Error: Could not load scenes from file. Using built-in fallback scene list and empty IR path.')
        scenes_list = ['Dining Room', 'Living Room', 'Kitchen', 'Bedroom', 'Balcony', 'Cafe', 'Park', 'Community Center', 'Office', 'Meeting Room']
        selected_scene = random.choice(scenes_list)
        env_ir_path = ''
    else:
        indices = list(range(len(scenes_english)))
        selected_index = random.choices(indices, weights=weights, k=1)[0]
        selected_scene_id = scene_ids[selected_index]
        selected_scene = scenes_english[selected_index]
        try:
            search_pattern = os.path.join(ir_base_dir, f'{selected_scene_id}_*_img.wav')
            matching_ir_files = glob.glob(search_pattern)
            if not matching_ir_files:
                print(f'Warning: Scene ID {selected_scene_id} found but no matching IR WAVs under {ir_base_dir}. Spatialization will run without environmental reverb.')
                env_ir_path = ''
            else:
                env_ir_path = random.choice(matching_ir_files)
                print(f'Selected scene ID {selected_scene_id}. Found {len(matching_ir_files)} IRs. Randomly selected: {env_ir_path}')
        except Exception as e:
            print(f'Error during IR file search: {e}')
            env_ir_path = ''
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    print(f'Generated unique ID for this run: {random_id}')
    run_dir = os.path.join(output_dir, random_id)
    os.makedirs(run_dir, exist_ok=True)
    character_cards = {name: {key: value for key, value in all_characters_data[name].items() if key != 'wav'} for name in selected_speaker_names}
    print(f"Selected {num_speakers} speakers: {', '.join(selected_speaker_names)}")
    print(f'Selected scene: {selected_scene}')
    character_cards_json_str = json.dumps(character_cards, indent=4, ensure_ascii=False)
    prompt = SCRIPT_MIXED.format(character_cards=character_cards_json_str, scene_key=selected_scene)
    llm_response_str = llm_bot.predict(prompt)
    if llm_response_str.startswith('```json'):
        llm_response_str = llm_response_str[7:]
    if llm_response_str.endswith('```'):
        llm_response_str = llm_response_str[:-3]
    try:
        generated_script_data = json.loads(llm_response_str)
        script_save_path = os.path.join(run_dir, f'{random_id}.json')
        save_json(generated_script_data, script_save_path)
        print(f'Script successfully generated and saved to {script_save_path}')
        plot_save_path = os.path.join(run_dir, f'{random_id}_layout.png')
        plot_spatial_layout(generated_script_data['scripts'][0], plot_save_path)
    except json.JSONDecodeError as e:
        print(f'Error: Failed to decode LLM response into JSON. Error: {e}')
        print(f'LLM Response:\n{llm_response_str}')
        return
    print('\n--- Starting Dynamic TTS Synthesis & Evaluation ---')
    tts_queue = build_tts_queue(cfg)
    text_batches, speaker_sequences = get_script(script_save_path)
    api_judges_list = initialize_api_judges(cfg)
    evaluator = Evaluator(llm_bot=llm_bot, audio_judges=api_judges_list)
    best_score = -1.0
    best_engine_name = ''
    best_battle_results = {}
    for tts_config in tts_queue:
        engine_name = tts_config['name']
        engine_class = tts_config['class']
        model_path = tts_config['path']
        print(f'\n[>>>] Testing engine: {engine_name}')
        print(f'[>>>] Loading and initializing engine: {engine_name} ...')
        text_to_instructs = defaultdict(list)
        for item in generated_script_data['scripts'][0].get('conversation', []):
            text_to_instructs[item['text']].append(item.get('prompt', None))
        if 'CosyVoice' in engine_name:
            engine_dir = run_dir
        else:
            engine_dir = os.path.join(run_dir, f'{random_id}_{engine_name}')
            os.makedirs(engine_dir, exist_ok=True)
        tts_instance = engine_class(model_path, engine_dir)
        if hasattr(tts_instance, 'set_speaker_info'):
            tts_instance.set_speaker_info(all_characters_data)
        for i, text_batch in enumerate(text_batches):
            speaker_sequence = speaker_sequences[i]
            unique_speakers = sorted(list(set(speaker_sequence)))
            v_prompts, v_wavs = ([], [])
            for speaker_name in unique_speakers:
                _, pt, wp = get_speaker(speaker_name, all_characters_data)
                v_prompts.append(pt)
                v_wavs.append(wp)
            c_prompts, c_wavs = ([], [])
            for speaker_name in speaker_sequence:
                _, pt, wp = get_speaker(speaker_name, all_characters_data)
                c_prompts.append(pt)
                c_wavs.append(wp)
            emotion_instructs = []
            for t in text_batch:
                if text_to_instructs[t]:
                    emotion_instructs.append(text_to_instructs[t].pop(0))
                else:
                    emotion_instructs.append(None)
            audio_name = f'{random_id}_{engine_name}_{i}.wav'
            print(f'[-] {engine_name} generating dry audio for batch {i}...')
            if engine_name == 'FishAudio':
                processed_texts = text_batch
            else:
                processed_texts = [re.sub('\\[.*?\\]\\s*', '', t).strip() for t in text_batch]
            if engine_name == 'VoxCPM':
                tts_instance.run_batch(texts=processed_texts, speaker_sequence=speaker_sequence, prompts=v_prompts, wavs=v_wavs, output_filename=audio_name)
            elif 'CosyVoice' in engine_name:
                tts_instance.run_batch(text_batches=processed_texts, speakers=speaker_sequence, prompts=c_prompts, wavs=c_wavs, output_filename=audio_name, emotion_instructs=emotion_instructs)
            elif engine_name == 'FishAudio':
                tts_instance.run_batch(texts=processed_texts, speaker_sequence=speaker_sequence, prompts=v_prompts, wavs=v_wavs, output_filename=audio_name)
            else:
                tts_instance.run_batch(texts=processed_texts, speakers=speaker_sequence, prompts=v_prompts, wavs=v_wavs, output_filename=audio_name)
        del tts_instance
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        audio_name_0 = f'{random_id}_{engine_name}_0.wav'
        if 'CosyVoice' in engine_name:
            score_save_dir = os.path.join(run_dir, audio_name_0.replace('.wav', ''))
            dry_audio_path = os.path.join(score_save_dir, audio_name_0)
            current_segments_dir = os.path.join(score_save_dir, 'segments')
        else:
            score_save_dir = engine_dir
            dry_audio_path = os.path.join(engine_dir, audio_name_0)
            current_segments_dir = os.path.join(engine_dir, 'segments')
        print(f'[-] Evaluating {engine_name} with API Judges (Scores Only)...')
        battle_results = evaluator.evaluate_dry_audio_for_battle(script_save_path, dry_audio_path, current_segments_dir)
        nat = battle_results['naturalness_score']
        emo = battle_results['emotiveness_score']
        utmos = battle_results['utmos']
        score = 0.35 * nat + 0.35 * emo + 0.3 * utmos
        os.makedirs(score_save_dir, exist_ok=True)
        score_log_path = os.path.join(score_save_dir, f'{random_id}_{engine_name}_battle_score.json')
        save_json({'engine': engine_name, 'total_score': score, 'metrics': battle_results}, score_log_path)
        print(f' [Final score] {engine_name} | Total: {score:.3f} (Nat: {nat}, Emo: {emo}, UTMOS: {utmos:.2f})')
        if score > best_score:
            best_score = score
            best_engine_name = engine_name
            best_battle_results = battle_results
    print(f'\n' + '★' * 50)
    print(f'Winner: {best_engine_name} | Best score: {best_score:.3f}')
    print('★' * 50)
    print(f'[-] Promoting full-dialogue audio and segment WAVs from {best_engine_name}...')
    dst_segments = os.path.join(run_dir, 'segments')
    os.makedirs(dst_segments, exist_ok=True)
    for i in range(len(text_batches)):
        audio_name = f'{random_id}_{best_engine_name}_{i}.wav'
        if 'CosyVoice' in best_engine_name:
            engine_sub_dir = os.path.join(run_dir, audio_name.replace('.wav', ''))
        else:
            engine_sub_dir = os.path.join(run_dir, f'{random_id}_{best_engine_name}')
        src_path = os.path.join(engine_sub_dir, audio_name)
        dst_path = os.path.join(run_dir, f'{random_id}_{i}.wav')
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        src_segments = os.path.join(engine_sub_dir, 'segments')
        if os.path.exists(src_segments):
            for file in os.listdir(src_segments):
                if file.endswith('.wav'):
                    shutil.copy(os.path.join(src_segments, file), os.path.join(dst_segments, file))
    print('\n--- Starting Spatialization (Using Winning Audios) ---')
    speaker_info_path = speaker_json_path
    spatialize_and_mix_voxcpm(run_dir, random_id, speaker_info_path, env_ir_path)
    print('\n--- Starting QA Generation ---')
    qa_save_path = os.path.join(run_dir, f'{random_id}_QA.json')
    try:
        script_data = generated_script_data['scripts'][0]
        clean_conversation = []
        for turn in script_data.get('conversation', []):
            clean_conversation.append({k: v for k, v in turn.items() if k != 'prompt'})
        conversation_script_json = json.dumps(clean_conversation, indent=4, ensure_ascii=False)
        spatial_annotation = {'roomsize': script_data.get('roomsize'), 'listener': script_data.get('listener') or script_data.get('listener_pos'), 'speaker': script_data.get('speaker')}
        spatial_annotation_json = json.dumps(spatial_annotation, indent=4, ensure_ascii=False)
        final_qa_prompt = QA_MIXED.format(conversation_script=conversation_script_json, spatial_annotation=spatial_annotation_json)
        qa_response_str = llm_bot.predict(final_qa_prompt)
        if qa_response_str.startswith('```json'):
            qa_response_str = qa_response_str[7:]
        if qa_response_str.endswith('```'):
            qa_response_str = qa_response_str[:-3]
        qa_data = json.loads(qa_response_str)
        save_json(qa_data, qa_save_path)
        print(f'QA set successfully generated and saved to {qa_save_path}')
    except json.JSONDecodeError as e:
        print(f'Error: Failed to decode LLM response for QA into JSON. Error: {e}')
    except Exception as e:
        print(f'An error occurred during QA generation: {e}')
    print('\n--- Generating Final Evaluation Report (Skipping Redundant API Call) ---')
    final_dry_audio_path = os.path.join(run_dir, f'{random_id}_0.wav')
    try:
        script_content = generated_script_data['scripts'][0]
        reference_text_with_prompts = evaluator._extract_reference_text_with_prompts(script_content)
        print('[-] Calculating WER and CER for the winning audio via local Whisper...')
        objective_metrics = evaluator.compute_wer_cer(reference_text_with_prompts, final_dry_audio_path)
        final_report = {'winning_engine': best_engine_name, 'winning_score': best_score, 'objective_metrics': objective_metrics, 'utmos_score': best_battle_results.get('utmos', 0.0), 'expert_eval': {'naturalness_score': best_battle_results.get('naturalness_score', 0.0), 'emotiveness_score': best_battle_results.get('emotiveness_score', 0.0), 'advice': f'Winner is {best_engine_name}. Detailed advice generation is skipped in Enhanced Mode to save API calls.', 'evaluated_by_count': len(api_judges_list)}}
        report_save_path = os.path.join(run_dir, 'evaluation_report.json')
        save_json(final_report, report_save_path)
        print(f'Final Evaluation Report successfully generated and saved to {report_save_path}')
    except Exception as e:
        print(f'Error generating final evaluation report: {e}')
    print('\n--- Enhanced Mode Finished ---')

def process_reflection_mode(llm_bot, output_dir, cfg, max_loop=3):
    print('\n=== Running in REFLECTION Mode (Multi-TTS Battle with Iterative Refinement) ===')
    random.seed(time.time())
    speaker_json_path, scene_file_path, ir_base_dir = data_paths(cfg)
    all_characters_data = read_json_file(speaker_json_path)
    all_speaker_names = list(all_characters_data.keys())
    num_speakers = random.choices([2, 3, 4], weights=[0.6, 0.2, 0.2], k=1)[0]
    selected_speaker_names = random.sample(all_speaker_names, num_speakers)
    scene_ids, scenes_chinese, scenes_english, weights = load_weighted_scenes(scene_file_path)
    if not scenes_english:
        print('Error: Could not load scenes from file. Using built-in fallback scene list and empty IR path.')
        scenes_list = ['Dining Room', 'Living Room', 'Kitchen', 'Bedroom', 'Balcony', 'Cafe', 'Park', 'Community Center', 'Office', 'Meeting Room']
        selected_scene = random.choice(scenes_list)
        env_ir_path = ''
    else:
        indices = list(range(len(scenes_english)))
        selected_index = random.choices(indices, weights=weights, k=1)[0]
        selected_scene_id = scene_ids[selected_index]
        selected_scene = scenes_english[selected_index]
        try:
            search_pattern = os.path.join(ir_base_dir, f'{selected_scene_id}_*_img.wav')
            matching_ir_files = glob.glob(search_pattern)
            if not matching_ir_files:
                print(f'Warning: Scene ID {selected_scene_id} found but no matching IR WAVs under {ir_base_dir}.')
                env_ir_path = ''
            else:
                env_ir_path = random.choice(matching_ir_files)
        except Exception as e:
            print(f'Error during IR file search: {e}')
            env_ir_path = ''
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    print(f'Reflection Mode Run ID: {random_id}')
    run_dir = os.path.join(output_dir, random_id)
    os.makedirs(run_dir, exist_ok=True)
    character_cards = {name: {key: value for key, value in all_characters_data[name].items() if key != 'wav'} for name in selected_speaker_names}
    print(f"Selected {num_speakers} speakers: {', '.join(selected_speaker_names)}")
    print(f'Selected scene: {selected_scene}')
    character_cards_json_str = json.dumps(character_cards, indent=4, ensure_ascii=False)
    llm_bot.clear_history()
    initial_prompt = SCRIPT_MIXED.format(character_cards=character_cards_json_str, scene_key=selected_scene)
    print('\n--- Generating Initial Script ---')
    llm_response_str = llm_bot.chat(initial_prompt)
    if llm_response_str.startswith('```json'):
        llm_response_str = llm_response_str[7:]
    if llm_response_str.endswith('```'):
        llm_response_str = llm_response_str[:-3]
    try:
        current_script_data = json.loads(llm_response_str)
    except json.JSONDecodeError as e:
        print(f'Error decoding initial script: {e}')
        return
    tts_queue = build_tts_queue(cfg)
    last_successful_iter_dir = ''
    best_engine_overall_name = ''
    best_overall_score = -1.0
    api_judges_list = initialize_api_judges(cfg)
    for loop_idx in range(max_loop + 1):
        print(f'\n' + '=' * 60)
        print(f'🔄 === Iteration {loop_idx} / {max_loop} ===')
        print('=' * 60)
        iter_dir = os.path.join(run_dir, f'{random_id}_iter_{loop_idx}')
        os.makedirs(iter_dir, exist_ok=True)
        last_successful_iter_dir = iter_dir
        script_save_path = os.path.join(iter_dir, f'{random_id}_script_iter{loop_idx}.json')
        save_json(current_script_data, script_save_path)
        print(f'[+] Script saved to: {script_save_path}')
        text_batches, speaker_sequences = get_script(script_save_path)
        if not text_batches:
            break
        evaluator = Evaluator(llm_bot=llm_bot, audio_judges=api_judges_list)
        best_iter_score = -1.0
        best_iter_engine = ''
        for tts_config in tts_queue:
            engine_name = tts_config['name']
            engine_class = tts_config['class']
            model_path = tts_config['path']
            print(f'\n[>>>] Loading TTS Engine: {engine_name} ...')
            text_to_instructs = defaultdict(list)
            for item in current_script_data['scripts'][0].get('conversation', []):
                text_to_instructs[item['text']].append(item.get('prompt', None))
            if 'CosyVoice' in engine_name:
                engine_dir = iter_dir
            else:
                engine_dir = os.path.join(iter_dir, f'{random_id}_iter{loop_idx}_{engine_name}')
                os.makedirs(engine_dir, exist_ok=True)
            tts_instance = engine_class(model_path, engine_dir)
            if hasattr(tts_instance, 'set_speaker_info'):
                tts_instance.set_speaker_info(all_characters_data)
            for i, text_batch in enumerate(text_batches):
                speaker_sequence = speaker_sequences[i]
                unique_speakers = sorted(list(set(speaker_sequence)))
                v_prompts, v_wavs = ([], [])
                for speaker_name in unique_speakers:
                    _, pt, wp = get_speaker(speaker_name, all_characters_data)
                    v_prompts.append(pt)
                    v_wavs.append(wp)
                c_prompts, c_wavs = ([], [])
                for speaker_name in speaker_sequence:
                    _, pt, wp = get_speaker(speaker_name, all_characters_data)
                    c_prompts.append(pt)
                    c_wavs.append(wp)
                emotion_instructs = []
                for t in text_batch:
                    if text_to_instructs[t]:
                        emotion_instructs.append(text_to_instructs[t].pop(0))
                    else:
                        emotion_instructs.append(None)
                audio_name = f'{random_id}_iter{loop_idx}_{engine_name}_{i}.wav'
                print(f'[-] {engine_name} generating Batch {i}...')
                if engine_name == 'FishAudio':
                    processed_texts = text_batch
                else:
                    processed_texts = [re.sub('\\[.*?\\]\\s*', '', t).strip() for t in text_batch]
                if engine_name == 'VoxCPM':
                    tts_instance.run_batch(texts=processed_texts, speaker_sequence=speaker_sequence, prompts=v_prompts, wavs=v_wavs, output_filename=audio_name)
                elif 'CosyVoice' in engine_name:
                    tts_instance.run_batch(text_batches=processed_texts, speakers=speaker_sequence, prompts=c_prompts, wavs=c_wavs, output_filename=audio_name, emotion_instructs=emotion_instructs)
                elif engine_name == 'FishAudio':
                    tts_instance.run_batch(texts=processed_texts, speaker_sequence=speaker_sequence, prompts=v_prompts, wavs=v_wavs, output_filename=audio_name)
                else:
                    tts_instance.run_batch(texts=processed_texts, speakers=speaker_sequence, prompts=v_prompts, wavs=v_wavs, output_filename=audio_name)
            del tts_instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            audio_name_0 = f'{random_id}_iter{loop_idx}_{engine_name}_0.wav'
            if 'CosyVoice' in engine_name:
                score_save_dir = os.path.join(iter_dir, audio_name_0.replace('.wav', ''))
                dry_audio_path = os.path.join(score_save_dir, audio_name_0)
                current_segments_dir = os.path.join(score_save_dir, 'segments')
            else:
                score_save_dir = engine_dir
                dry_audio_path = os.path.join(engine_dir, audio_name_0)
                current_segments_dir = os.path.join(engine_dir, 'segments')
            print(f'[-] Evaluating {engine_name} with API Judges (Scores Only)...')
            battle_results = evaluator.evaluate_dry_audio_for_battle(script_save_path, dry_audio_path, current_segments_dir)
            nat = battle_results['naturalness_score']
            emo = battle_results['emotiveness_score']
            utmos = battle_results['utmos']
            score = 0.35 * nat + 0.35 * emo + 0.3 * utmos
            os.makedirs(score_save_dir, exist_ok=True)
            score_log_path = os.path.join(score_save_dir, f'{random_id}_iter{loop_idx}_{engine_name}_battle_score.json')
            save_json({'engine': engine_name, 'total_score': score, 'metrics': battle_results}, score_log_path)
            print(f' [🏅 Score] {engine_name} | Total: {score:.3f} (Nat: {nat}, Emo: {emo}, UTMOS: {utmos:.2f})')
            if score > best_iter_score:
                best_iter_score = score
                best_iter_engine = engine_name
        print(f'\n🏆 Iteration {loop_idx} Winner: {best_iter_engine} (Score: {best_iter_score:.3f})')
        best_overall_score = best_iter_score
        best_engine_overall_name = best_iter_engine
        print(f"[-] Promoting {best_iter_engine}'s audios to iteration root...")
        iter_dst_segments = os.path.join(iter_dir, 'segments')
        os.makedirs(iter_dst_segments, exist_ok=True)
        current_audio_path_to_evaluate = ''
        for i in range(len(text_batches)):
            audio_name = f'{random_id}_iter{loop_idx}_{best_iter_engine}_{i}.wav'
            if 'CosyVoice' in best_iter_engine:
                engine_sub_dir = os.path.join(iter_dir, audio_name.replace('.wav', ''))
            else:
                engine_sub_dir = os.path.join(iter_dir, f'{random_id}_iter{loop_idx}_{best_iter_engine}')
            src_path = os.path.join(engine_sub_dir, audio_name)
            dst_name = f'{random_id}_iter{loop_idx}_{i}.wav'
            dst_path = os.path.join(iter_dir, dst_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                if i == 0:
                    current_audio_path_to_evaluate = dst_path
            src_segments = os.path.join(engine_sub_dir, 'segments')
            if os.path.exists(src_segments):
                for file in os.listdir(src_segments):
                    if file.endswith('.wav'):
                        shutil.copy(os.path.join(src_segments, file), os.path.join(iter_dst_segments, file))
        print(f'\n[>>>] Loading API Judges for Detailed Advice...')
        iter_results = evaluator.run_comprehensive_evaluation(script_path=script_save_path, audio_path=current_audio_path_to_evaluate, qa_path='', output_dir=iter_dir, segments_dir=iter_dst_segments)
        old_report_path = os.path.join(iter_dir, 'evaluation_report.json')
        new_report_path = os.path.join(iter_dir, f'{random_id}_eval_iter{loop_idx}.json')
        if os.path.exists(old_report_path):
            os.rename(old_report_path, new_report_path)
        expert_feedback = iter_results.get('expert_eval', {})
        advice = expert_feedback.get('advice', '')
        if isinstance(advice, (dict, list)):
            advice_str = json.dumps(advice, ensure_ascii=False, indent=2)
        else:
            advice_str = str(advice)
        print(f'  -> Audio Judges Advice:\n{advice_str}')
        if loop_idx == max_loop:
            print('Max loops reached. Proceeding to final spatialization.')
            break
        print('Refining script with LLM based on advice...')
        refine_prompt = UPDATE_REFINE_PROMPT.format(advice=advice_str)
        llm_response_str = llm_bot.chat(refine_prompt)
        if llm_response_str.startswith('```json'):
            llm_response_str = llm_response_str[7:]
        if llm_response_str.endswith('```'):
            llm_response_str = llm_response_str[:-3]
        try:
            current_script_data = json.loads(llm_response_str)
            print(f'Script updated for next iteration.')
        except json.JSONDecodeError:
            print('Error: LLM failed to return valid JSON during refinement. Stopping loops.')
            break
    print('\n--- Finalizing and Starting Spatialization ---')
    if last_successful_iter_dir:
        for i in range(len(text_batches)):
            dst_name_in_iter = f'{random_id}_iter{loop_idx}_{i}.wav'
            src_audio_path = os.path.join(last_successful_iter_dir, dst_name_in_iter)
            target_baseline_name = os.path.join(run_dir, f'{random_id}_{i}.wav')
            if os.path.exists(src_audio_path):
                shutil.copy(src_audio_path, target_baseline_name)
        src_segments = os.path.join(last_successful_iter_dir, 'segments')
        dst_segments = os.path.join(run_dir, 'segments')
        if os.path.exists(src_segments):
            os.makedirs(dst_segments, exist_ok=True)
            for file in os.listdir(src_segments):
                if file.endswith('.wav'):
                    shutil.copy(os.path.join(src_segments, file), os.path.join(dst_segments, file))
    target_script_name = os.path.join(run_dir, f'{random_id}.json')
    save_json(current_script_data, target_script_name)
    final_plot_save_path = os.path.join(run_dir, f'{random_id}_layout.png')
    plot_spatial_layout(current_script_data['scripts'][0], final_plot_save_path)
    speaker_info_path = speaker_json_path
    spatialize_and_mix_voxcpm(run_dir, random_id, speaker_info_path, env_ir_path)
    print('\n--- Starting QA Generation ---')
    qa_save_path = os.path.join(run_dir, f'{random_id}_QA.json')
    try:
        script_data = current_script_data['scripts'][0]
        clean_conversation = []
        for turn in script_data.get('conversation', []):
            clean_conversation.append({k: v for k, v in turn.items() if k != 'prompt'})
        conversation_script_json = json.dumps(clean_conversation, indent=4, ensure_ascii=False)
        spatial_annotation = {'roomsize': script_data.get('roomsize'), 'listener': script_data.get('listener') or script_data.get('listener_pos'), 'speaker': script_data.get('speaker')}
        spatial_annotation_json = json.dumps(spatial_annotation, indent=4, ensure_ascii=False)
        llm_bot.clear_history()
        final_qa_prompt = QA_MIXED.format(conversation_script=conversation_script_json, spatial_annotation=spatial_annotation_json)
        qa_response_str = llm_bot.chat(final_qa_prompt)
        if qa_response_str.startswith('```json'):
            qa_response_str = qa_response_str[7:]
        if qa_response_str.endswith('```'):
            qa_response_str = qa_response_str[:-3]
        qa_data = json.loads(qa_response_str)
        save_json(qa_data, qa_save_path)
        print(f'QA set successfully generated and saved to {qa_save_path}')
    except Exception as e:
        print(f'An error occurred during QA generation: {e}')
    print('\n--- Starting Final Comprehensive Evaluation ---')
    final_dry_audio_path = os.path.join(run_dir, f'{random_id}_0.wav')
    final_segments_dir = os.path.join(run_dir, 'segments')
    evaluator.run_comprehensive_evaluation(script_path=target_script_name, audio_path=final_dry_audio_path, qa_path=qa_save_path, output_dir=run_dir, segments_dir=final_segments_dir)
    print('--- Reflection Mode Finished ---')
if __name__ == '__main__':
    args = parse_arguments()
    mode = args.mode
    num_samples = args.num_samples
    cfg = load_config(args.config)
    gr.activateMixedPrecision(True)
    gr.activateLUT(True)
    print('Initializing LLM...')
    llm_bot = initialize_llm(cfg)
    for i in range(num_samples):
        print('\n' + '🚀' * 20)
        print(f'🚀 Starting Generation Loop: {i + 1} / {num_samples}')
        print('🚀' * 20)
        try:
            if mode == 'baseline':
                process_baseline_mode(llm_bot, args.output_dir, cfg)
            elif mode == 'enhanced':
                process_enhanced_mode(llm_bot, args.output_dir, cfg)
            elif mode == 'reflection':
                process_reflection_mode(llm_bot, args.output_dir, cfg, args.max_loop)
            else:
                raise ValueError(f'Unknown mode: {mode}')
        except Exception as e:
            print(f'\n❌ Error occurred in loop {i + 1}: {e}')
            logging.error(f'Error in loop {i + 1}', exc_info=True)
            continue
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)
    print(f'\nDone. Batch run finished (target samples: {num_samples}).')
