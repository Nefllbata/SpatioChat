import re
import json

def get_text_inside_tag(html_string: str, tag: str):
    pattern = f'<{tag}>(.*?)<\\/{tag}>'
    try:
        result = re.findall(pattern, html_string, re.DOTALL)
        return result
    except SyntaxError as e:
        raise 'Json Decode Error: {error}'.format(error=e)

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f'file not found: {file_path}')
    except json.JSONDecodeError:
        print(f'json edcodeerror: {file_path}')
    except Exception as e:
        print(f'error: {e}')

def get_script(file_path):
    data = read_json_file(file_path)
    text_batches = []
    speakers = []
    for i in data['scripts']:
        text = []
        speaker = []
        for j in i['conversation']:
            speaker.append(j['speaker'])
            text.append(j['text'])
        text_batches.append(text)
        speakers.append(speaker)
    return (text_batches, speakers)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def replace_(text):
    return re.sub('[?！？!]', '.', text)

def get_speaker(name):
    speakers = read_json_file('data/speaker/myspeaker.json')
    return (speakers[name]['label'], speakers[name]['description'], speakers[name]['wav'])

def get_speaker(name, speakers_data):
    if name in speakers_data:
        return (speakers_data[name]['label'], speakers_data[name]['text'], speakers_data[name]['wav'])
    else:
        raise KeyError(f"Speaker '{name}' not found in the provided speaker data.")

def load_weighted_scenes(file_path):
    scene_ids = []
    scenes_chinese = []
    scenes_english = []
    numerical_weights = []
    weight_map = {'H': 4, 'M': 3, 'ML': 2, 'L': 1}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f'Error: Scene file not found at {file_path}')
        return ([], [], [], [])
    except Exception as e:
        print(f'Error reading scene file: {e}')
        return ([], [], [], [])
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            print(f'Skipping malformed line: {line}')
            continue
        scene_id = parts[0]
        scene_name_chinese = parts[1]
        weight_str = parts[-1]
        if weight_str not in weight_map:
            print(f"Skipping line with invalid weight '{weight_str}': {line}")
            continue
        english_name_parts = parts[2:-1]
        scene_name_english = ' '.join(english_name_parts)
        scene_ids.append(scene_id)
        scenes_chinese.append(scene_name_chinese)
        scenes_english.append(scene_name_english)
        numerical_weights.append(weight_map[weight_str])
    return (scene_ids, scenes_chinese, scenes_english, numerical_weights)
