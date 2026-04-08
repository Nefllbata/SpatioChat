import os
import json
import numpy as np
import gpuRIR as gr
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly
from fractions import Fraction
BRIR_FS = 16000
T60 = 0.1
REVERB_TAIL_SECONDS = 0.3

def _read_wav_float_scipy(path):
    try:
        sr, x = wavfile.read(path)
    except FileNotFoundError:
        print(f'[Error] File not found: {path}')
        return (None, None)
    if x.dtype.kind in 'iu':
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32) / max(1, info.max)
    else:
        x = x.astype(np.float32)
    return (sr, x)

def _read_wav_float_sf(path):
    try:
        x, sr = sf.read(path, dtype='float32')
        return (sr, x)
    except Exception as e:
        print(f'Failed to read file {path}: {e}')
        return (None, None)

def _to_mono(x):
    if x.ndim == 2 and x.shape[1] > 1:
        return x.mean(axis=1)
    return x

def _resample_to(sr_from, sr_to, x):
    if sr_from == sr_to:
        return x
    frac = Fraction(sr_to, sr_from).limit_denominator()
    up, down = (frac.numerator, frac.denominator)
    if x.ndim == 1:
        y = resample_poly(x, up, down).astype(np.float32)
    else:
        y = np.stack([resample_poly(x[:, ch], up, down) for ch in range(x.shape[1])], axis=1).astype(np.float32)
    return y

def _generate_8ch_rir_for_speaker(room_sz, listener_pos, speaker_pos):
    print(f'  > Generating 8-ch RIR for speaker position {speaker_pos} ...')
    if listener_pos is None:
        print('    [Warning] No listener position in script; placing listener at room center.')
        listener_pos = [room_sz[0] / 2.0, room_sz[1] / 2.0, 1.0]
    room_sz = np.array(room_sz)
    beta = gr.beta_SabineEstimation(room_sz, T60)
    pos_src = np.array([speaker_pos])
    head_center = np.array(listener_pos)
    d = 0.1
    offsets = np.array([[d, d, d], [d, d, -d], [d, -d, d], [d, -d, -d], [-d, d, d], [-d, d, -d], [-d, -d, d], [-d, -d, -d]])
    pos_rcv = head_center + offsets
    Tmax = T60
    nb_img = gr.t2n(Tmax, room_sz)
    RIR = gr.simulateRIR(room_sz=room_sz, beta=beta, pos_src=pos_src, pos_rcv=pos_rcv, nb_img=nb_img, Tmax=Tmax, fs=BRIR_FS, mic_pattern='omni')
    return RIR[0].T

def _convolve_rir_with_env(directional_rir, env_ir, rir_sr, env_ir_sr):
    env_ir_mono = _to_mono(env_ir)
    env_ir_rs = _resample_to(env_ir_sr, rir_sr, env_ir_mono)
    env_rir_channels = []
    for ch in range(directional_rir.shape[1]):
        convolved_ch = fftconvolve(directional_rir[:, ch], env_ir_rs, mode='full')
        env_rir_channels.append(convolved_ch)
    env_rir = np.stack(env_rir_channels, axis=1).astype(np.float32)
    return env_rir

def spatialize_and_mix(run_dir, script_id, speaker_info_path, env_ir_path=None):
    pass

def spatialize_and_mix_voxcpm(run_dir, script_id, speaker_info_path, env_ir_path=None):
    print(f'\n--- 8-channel spatialization, ID: {script_id} ---')
    script_path = os.path.join(run_dir, f'{script_id}.json')
    brir_output_dir = os.path.join(run_dir, 'RIR_8CH')
    final_output_direct_path = os.path.join(run_dir, f'{script_id}.wav')
    final_output_mixed_path = os.path.join(run_dir, f'{script_id}_mixed.wav')
    segments_direct_dir = os.path.join(run_dir, 'segments_8ch_direct')
    segments_mixed_dir = os.path.join(run_dir, 'segments_8ch_mixed')
    normalize_path = os.path.join(brir_output_dir, 'normalize.txt')
    os.makedirs(brir_output_dir, exist_ok=True)
    os.makedirs(segments_direct_dir, exist_ok=True)
    if env_ir_path:
        os.makedirs(segments_mixed_dir, exist_ok=True)
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            generated_script_data = json.load(f)
        with open(speaker_info_path, 'r', encoding='utf-8') as f:
            all_speakers_info = json.load(f)
        env_ir, sr_env = (None, None)
        if env_ir_path:
            sr_env, env_ir = _read_wav_float_sf(env_ir_path)
            if env_ir is None:
                print(f'[Warning] Could not load environment IR: {env_ir_path}. Skipping mixed output.')
    except Exception as e:
        print(f'[Error] Failed to load input files: {e}')
        return
    for i, script_data_batch in enumerate(generated_script_data['scripts']):
        output_dir = os.path.dirname(run_dir)
        dry_audio_dir = os.path.join(output_dir, f'{script_id}', 'segments')
        print(f'\n[Batch {i}] Step 1/4: Convolving and staging 8-channel segments...')
        final_mix_direct = np.array([], dtype=np.float32).reshape(0, 8)
        final_mix_mixed = np.array([], dtype=np.float32).reshape(0, 8)
        target_sr = None
        batch_segments_direct = []
        batch_segments_mixed = []
        for line_idx, line in enumerate(script_data_batch['conversation']):
            speaker_name = line['speaker']
            print(f"  > Utterance {line_idx + 1} from '{speaker_name}'")
            speaker_pos = next((s['pos'] for s in script_data_batch['speaker'] if s['name'] == speaker_name))
            directional_rir = _generate_8ch_rir_for_speaker(script_data_batch['roomsize'], script_data_batch.get('listener_pos') or script_data_batch.get('listener'), speaker_pos)
            sf.write(os.path.join(brir_output_dir, f'{speaker_name}_dry_8ch.wav'), directional_rir, BRIR_FS)
            env_rir = None
            if env_ir is not None:
                print('    -> Fusing 8-ch RIR with environment IR...')
                env_rir = _convolve_rir_with_env(directional_rir, env_ir, BRIR_FS, sr_env)
            dry_wav_path = os.path.join(dry_audio_dir, f'{speaker_name}_text{line_idx}_output.wav')
            sr_x, x = _read_wav_float_scipy(dry_wav_path)
            if sr_x is None:
                continue
            if target_sr is None:
                target_sr = sr_x
            x_mono = _to_mono(x)
            tail_samples = int(REVERB_TAIL_SECONDS * target_sr)
            desired_length = len(x) + tail_samples
            print('    -> Convolving dry speech with directional 8-ch RIR...')
            h_rs_direct = _resample_to(BRIR_FS, target_sr, directional_rir)
            convolved_direct_channels = [fftconvolve(x_mono, h_rs_direct[:, ch]) for ch in range(8)]
            convolved_direct_full = np.stack(convolved_direct_channels, axis=1)
            convolved_direct_truncated = convolved_direct_full[:desired_length]
            final_mix_direct = np.vstack([final_mix_direct, convolved_direct_truncated])
            seg_name_dir = f'{speaker_name}_text{line_idx}_8ch_direct.wav'
            batch_segments_direct.append((seg_name_dir, convolved_direct_truncated))
            if env_rir is not None:
                print('    -> Convolving dry speech with mixed 8-ch RIR...')
                h_rs_mixed = _resample_to(BRIR_FS, target_sr, env_rir)
                convolved_mixed_channels = [fftconvolve(x_mono, h_rs_mixed[:, ch]) for ch in range(8)]
                convolved_mixed_full = np.stack(convolved_mixed_channels, axis=1)
                convolved_mixed_truncated = convolved_mixed_full[:desired_length]
                final_mix_mixed = np.vstack([final_mix_mixed, convolved_mixed_truncated])
                seg_name_mix = f'{speaker_name}_text{line_idx}_8ch_mixed.wav'
                batch_segments_mixed.append((seg_name_mix, convolved_mixed_truncated))
        print(f'\n[Batch {i}] Step 2/4: Computing global peak for normalization (preserving distance decay)...')
        if final_mix_direct.size == 0:
            print('[Error] No audio to mix.')
            continue
        peak = max(np.max(np.abs(final_mix_direct)), np.max(np.abs(final_mix_mixed)) if final_mix_mixed.size > 0 else 0) + 1e-12
        final_mix_direct_norm = (final_mix_direct / peak * 0.98).astype(np.float32)
        if final_mix_mixed.size > 0:
            final_mix_mixed_norm = (final_mix_mixed / peak * 0.98).astype(np.float32)
        try:
            with open(normalize_path, 'w') as f:
                f.write(str(peak))
        except Exception as e:
            print(f'[Warning] Failed to save peak value: {e}')
        print(f'\n[Batch {i}] Step 3/4: Writing full 8-channel spatial dialogue...')
        wavfile.write(final_output_direct_path, target_sr, final_mix_direct_norm)
        print(f'  [OK] Directional-only long mix: {final_output_direct_path}')
        if final_mix_mixed.size > 0:
            wavfile.write(final_output_mixed_path, target_sr, final_mix_mixed_norm)
            print(f'  [OK] Directional+environment long mix: {final_output_mixed_path}')
        print(f'\n[Batch {i}] Step 4/4: Writing per-utterance 8-channel slices...')
        for seg_name, seg_array in batch_segments_direct:
            seg_norm = (seg_array / peak * 0.98).astype(np.float32)
            seg_path = os.path.join(segments_direct_dir, seg_name)
            wavfile.write(seg_path, target_sr, seg_norm)
        for seg_name, seg_array in batch_segments_mixed:
            seg_norm = (seg_array / peak * 0.98).astype(np.float32)
            seg_path = os.path.join(segments_mixed_dir, seg_name)
            wavfile.write(seg_path, target_sr, seg_norm)
        print(f'  [OK] All per-utterance 8-channel slices written.')
