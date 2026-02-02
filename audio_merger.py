import os
import argparse
import re
import subprocess
import shutil
import tempfile
import sys
from glob import glob

def get_numeric_key(filename):
    """
    Extracts all numbers from the filename and returns them as a list of integers.
    Useful for sorting filenames like 'file1.wav', 'file10.wav', etc.
    """
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]

def get_audio_properties(filepath):
    """
    Returns (sample_rate, channels) for the given audio file using ffprobe.
    """
    try:
        # Get sample rate
        sr_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1', filepath
        ]
        sr_output = subprocess.check_output(sr_cmd).decode('utf-8').strip()
        sample_rate = int(sr_output)

        # Get channels
        ch_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=channels',
            '-of', 'default=noprint_wrappers=1:nokey=1', filepath
        ]
        ch_output = subprocess.check_output(ch_cmd).decode('utf-8').strip()
        channels = int(ch_output)
        
        return sample_rate, channels
    except Exception as e:
        print(f"Error probing file {filepath}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Convert WAV to MP3, concat with silence.")
    parser.add_argument("folder_path", help="Path to the folder containing WAV files")
    parser.add_argument("--prefix", help="Filter files by this prefix", default="")
    parser.add_argument("--output", help="Output filename (default: combined.mp3)", default="combined.mp3")
    
    args = parser.parse_args()
    
    folder_path = os.path.abspath(args.folder_path)
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)

    # 1. Find and Filter Files
    search_pattern = os.path.join(folder_path, f"{args.prefix}*.wav")
    wav_files = glob(search_pattern)
    
    if not wav_files:
        print(f"No WAV files found in '{folder_path}' matching prefix '{args.prefix}'")
        sys.exit(1)
        
    # 2. Sort Files
    wav_files.sort(key=lambda f: get_numeric_key(os.path.basename(f)))
    
    print(f"Found {len(wav_files)} files. Processing...")
    
    # 3. Analyze first file for properties
    sample_rate, channels = get_audio_properties(wav_files[0])
    print(f"Detected Sample Rate: {sample_rate} Hz, Channels: {channels}")

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Working in temporary directory: {temp_dir}")
        
        mp3_files = []
        
        # 4. Convert each WAV to MP3
        for i, wav_file in enumerate(wav_files):
            filename = os.path.basename(wav_file)
            name_no_ext = os.path.splitext(filename)[0]
            output_mp3 = os.path.join(temp_dir, f"{name_no_ext}.mp3")
            
            print(f"Converting ({i+1}/{len(wav_files)}): {filename} -> MP3")
            
            # Construct ffmpeg command
            # -ar {sample_rate} ensures we keep the same rate
            # -q:a 2 is a good quality VBR setting for MP3
            cmd = [
                'ffmpeg', '-y', '-i', wav_file,
                '-acodec', 'libmp3lame',
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-q:a', '2',
                output_mp3
            ]
            
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            mp3_files.append(output_mp3)

        # 5. Generate Silence MP3
        silence_path = os.path.join(temp_dir, "silence.mp3")
        print("Generating 5s silence...")
        # anullsrc creates a null audio source
        silence_cmd = [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', f'anullsrc=r={sample_rate}:cl={"stereo" if channels == 2 else "mono"}',
            '-t', '5',
            '-acodec', 'libmp3lame',
            '-q:a', '2',
            silence_path
        ]
        
        # Handle channel layout string for anullsrc more robustly if needed, 
        # but usually stereo/mono covers 99% of cases.
        # If channels > 2, anullsrc might need specific layout but let's assume mono/stereo.
        if channels not in [1, 2]:
             print(f"Warning: Uncommon channel count {channels}. Defaulting silence to stereo.")
             silence_cmd[5] = f'anullsrc=r={sample_rate}:cl=stereo'

        subprocess.run(silence_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        # 6. Create Concatenation List
        concat_list_path = os.path.join(temp_dir, "files.txt")
        with open(concat_list_path, 'w') as f:
            # Start with silence
            f.write(f"file '{silence_path}'\n")
            
            for mp3 in mp3_files:
                f.write(f"file '{mp3}'\n")
                # After every file (including the last one), add silence
                f.write(f"file '{silence_path}'\n")

        # 7. Concatenate
        final_output = os.path.abspath(args.output)
        print(f"Concatenating into {final_output}...")
        
        concat_cmd = [
            'ffmpeg', '-y', '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_path,
            '-c', 'copy',
            final_output
        ]
        
        subprocess.run(concat_cmd, check=True)
        print("Done!")

if __name__ == "__main__":
    main()
