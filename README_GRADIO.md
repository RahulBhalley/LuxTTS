# LuxTTS Gradio Interface

A Gradio web interface for the LuxTTS voice cloning system.

## Setup

1. Activate the virtual environment:
```bash
source ~/venvs/luxtts/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Gradio interface:
```bash
python app.py
```

## Usage

1. Open the provided URL in your browser (usually http://localhost:7860)
2. Upload or record a voice prompt audio file
3. Enter the text you want to generate
4. Adjust advanced parameters if needed
5. Click "Generate Speech" to create the cloned voice

## Features

- **Text Input**: Enter any text to be converted to speech
- **Voice Prompt**: Upload an audio file or record directly to provide voice characteristics
- **Advanced Parameters**:
  - Number of Steps: Controls generation quality vs speed
  - RMS: Volume normalization for the prompt
  - Guidance Scale: How closely to follow the prompt voice
  - Temperature Shift: Controls randomness
  - Speech Speed: Adjusts the speed of generated speech

## Requirements

- PyTorch with MPS support (for Apple Silicon)
- Gradio
- SoundFile
- All LuxTTS dependencies (see requirements.txt)
