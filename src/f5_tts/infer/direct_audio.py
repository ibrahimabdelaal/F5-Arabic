import os
import torch
import torchaudio
import numpy as np

# Import specific functions directly from utils_infer
from f5_tts.infer.utils_infer import target_sample_rate, hop_length
from f5_tts.model.utils import convert_char_to_pinyin

# Custom infer_batch_process function that doesn't use temp files
def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=None,
    target_rms=0.1,
    cross_fade_duration=0.1,
    nfe_step=32,
    cfg_strength=1.5,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
):
    """
    Direct audio processing without creating temporary files
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading reference audio directly: {ref_audio}")
    audio, sr = torchaudio.load(ref_audio)
    
    # Handle stereo to mono conversion
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Handle resampling
    if sr != target_sample_rate:
        print(f"Resampling from {sr}Hz to {target_sample_rate}Hz")
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    
    # Move to device
    audio = audio.to(device)

    # Process batches
    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "
    
    # Use progress bar if available
    iter_fn = progress.tqdm if progress else lambda x: x
    
    for i, gen_text in enumerate(iter_fn(gen_text_batches)):
        print(f"Processing text batch {i+1}/{len(gen_text_batches)}")
        
        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # inference
        with torch.inference_mode():
            print(f"Running inference with nfe_step={nfe_step}, cfg_strength={cfg_strength}")
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = generated.permute(0, 2, 1)
            
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated_mel_spec)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated_mel_spec)
            
            # Adjust volume
            generated_wave = generated_wave * 0.95  # Slight volume reduction to avoid clipping

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()
            generated_waves.append(generated_wave)
            spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves with cross-fading
    if cross_fade_duration <= 0:
        # Simply concatenate
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            # Calculate cross-fade samples, ensuring it does not exceed wave lengths
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            # Cross-faded overlap
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            # Combine
            new_wave = np.concatenate(
                [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
            )

            final_wave = new_wave

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)

    return final_wave, target_sample_rate, combined_spectrogram