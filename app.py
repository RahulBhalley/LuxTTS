import gradio as gr
import torch
import soundfile as sf
import tempfile
import os
import re
import time
import gc
import numpy as np
from zipvoice.luxvoice import LuxTTS

# Initialize the LuxTTS model
# Auto-detect best device for Mac (MPS) or others
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading LuxTTS model on {device}...")
lux_tts = LuxTTS('YatharthS/LuxTTS', device=device)
print("Model loaded successfully!")

def preprocess_text(text, clean_punctuation=True):
    """
    Clean and preprocess text for TTS generation.
    We preserve standard punctuation for prosody, trusting the tokenizer to handle
    pronunciation issues (like 'dot' for periods).
    """
    if not clean_punctuation:
        return text.strip()
    
    # Remove only special characters that might cause issues, but KEEP punctuation
    # . , ! ? ; : are essential for pauses and intonation.
    text = re.sub(r'[#*@\[\]{}()<>|]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def generate_speech_with_prompt(
    text, 
    audio_file, 
    num_steps, 
    rms, 
    guidance_scale, 
    t_shift, 
    speed, 
    ref_duration,
    return_smooth,
    clean_punctuation=True,
    use_mps=True
):
    """
    Generate speech using LuxTTS with text input and audio prompt.
    """
    try:
        global lux_tts
        
        # Determine target device
        target_device = "cpu"
        if use_mps:
            if torch.backends.mps.is_available():
                target_device = "mps"
            elif torch.cuda.is_available():
                target_device = "cuda"
        
        # Reload model if device changed
        if lux_tts.device != target_device:
            print(f"Switching device from {lux_tts.device} to {target_device}...")
            # Attempt to free memory
            try:
                del lux_tts
                import gc
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error clearing cache: {e}")
            
            lux_tts = LuxTTS('YatharthS/LuxTTS', device=target_device)
            print(f"Model reloaded on {target_device}!")

        if audio_file is None or not text:
            return None, "Error: Please provide text and a voice prompt audio file."
        
        start_time = time.time()
        
        # Preprocess the text
        cleaned_text = preprocess_text(text, clean_punctuation)
        if not cleaned_text:
            return None, "Error: Text is empty after preprocessing."
        
        print(f"Processing text: {cleaned_text}")
        
        # Handle different audio input types (microphone produces tuple)
        temp_audio_path = None
        if isinstance(audio_file, tuple):
            sample_rate, audio_data = audio_file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                sf.write(temp_audio.name, audio_data, sample_rate)
                temp_audio_path = temp_audio.name
        elif hasattr(audio_file, 'read'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(audio_file.read())
                temp_audio_path = temp_audio.name
        else:
            temp_audio_path = audio_file
        
        try:
            # Encode the prompt audio with user-specified duration and RMS
            encoded_prompt = lux_tts.encode_prompt(temp_audio_path, duration=ref_duration, rms=rms)
            
            # Generate speech
            final_wav = lux_tts.generate_speech(
                cleaned_text, 
                encoded_prompt, 
                num_steps=int(num_steps),
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                return_smooth=return_smooth
            ).numpy().squeeze()
            
            # Save the result to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_output:
                sf.write(temp_output.name, final_wav, 48000)
                output_path = temp_output.name
            
            generation_time = round(time.time() - start_time, 2)
            status_msg = f"✨ Generation complete in **{generation_time}s** on **{lux_tts.device.upper()}**."
            
            return output_path, status_msg
            
        finally:
            # Clean up temporary prompt file if created
            if temp_audio_path and temp_audio_path != audio_file and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            
            # Clear GPU cache after every generation
            try:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Error clearing cache: {e}")
                
    except Exception as e:
        return None, f"### Error generating speech:\n{str(e)}"

# Create the premium Gradio interface
with gr.Blocks(title="LuxTTS Voice Cloning", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ LuxTTS Voice Cloning")
    gr.Markdown(
        """
        Generate high-quality speech using any voice prompt. 
        *Tip: If the output sounds cut-off or unnatural, adjust the **Reference Duration** or **Speed**.*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input Text
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter the text you want the voice to speak...",
                lines=8,
                value="Hello! I am using LuxTTS for voice cloning. It's working quite well, and now it won't say 'dot' for periods anymore."
            )
            
            # Audio Input
            audio_input = gr.Audio(
                label="Voice Prompt Audio",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            # Main Generation Button
            generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")

        with gr.Column(scale=1):
            # Output Display
            audio_output = gr.Audio(
                label="Generated Speech",
                interactive=False,
                type="filepath"
            )
            
            status_output = gr.Markdown("### Status\nReady to generate...")
            
            # Generation Parameters Accordion
            with gr.Accordion("Advanced Parameters", open=True):
                with gr.Row():
                    mps_chk = gr.Checkbox(
                        label="Use MPS/GPU Acceleration",
                        value=True if (torch.backends.mps.is_available() or torch.cuda.is_available()) else False,
                        info="e.g. Metal on Mac (Uncheck to force CPU)"
                    )
                    clean_punct_chk = gr.Checkbox(
                        label="Clean Punctuation",
                        value=True,
                        info="Prevent literal speaking of dots/commas"
                    )
                    smooth_chk = gr.Checkbox(
                        label="Return Smooth",
                        value=False,
                        info="Enable smoothing for the vocoder"
                    )
                
                with gr.Row():
                    steps_sld = gr.Slider(
                        minimum=1, maximum=20, value=8, step=1,
                        label="Num Steps",
                        info="Higher = Better quality, Slower"
                    )
                    speed_sld = gr.Slider(
                        minimum=0.5, maximum=2.0, value=0.9, step=0.1,
                        label="Speech Speed",
                        info="Lower = Slower & Clearer"
                    )
                
                with gr.Row():
                    rms_sld = gr.Slider(
                        minimum=0.001, maximum=0.05, value=0.01, step=0.001,
                        label="RMS (Loudness)",
                        info="Controls normalization volume"
                    )
                    ref_dur_sld = gr.Slider(
                        minimum=1, maximum=10000, value=5, step=1,
                        label="Ref Duration (sec)",
                        info="Slice length of prompt"
                    )
                
                with gr.Row():
                    guidance_sld = gr.Slider(
                        minimum=1.0, maximum=10.0, value=3.0, step=0.1,
                        label="Guidance Scale",
                        info="Prompt adherence"
                    )
                    t_shift_sld = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.9, step=0.1,
                        label="Temperature Shift (T-Shift)",
                        info="Randomness in generation"
                    )

    # Examples for quick testing
    gr.Examples(
        examples=[
            [
                "Settle into a comfortable position. Allow your body to be supported by the surface beneath you.",
                None, 10, 0.01, 3.0, 0.9, 0.8, 5, False, True, True
            ],
            [
                "Hello, this is a test of the LuxTTS voice cloning system. It handles punctuation gracefully now.",
                None, 5, 0.01, 3.0, 0.9, 1.0, 5, False, True, True
            ]
        ],
        inputs=[
            text_input, audio_input, steps_sld, rms_sld, 
            guidance_sld, t_shift_sld, speed_sld, ref_dur_sld, 
            smooth_chk, clean_punct_chk, mps_chk
        ],
    )
    
    # Event handling
    generate_btn.click(
        fn=generate_speech_with_prompt,
        inputs=[
            text_input, audio_input, steps_sld, rms_sld, 
            guidance_sld, t_shift_sld, speed_sld, ref_dur_sld, 
            smooth_chk, clean_punct_chk, mps_chk
        ],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(
        share=True, 
        server_name="0.0.0.0", 
        server_port=7860
    )
