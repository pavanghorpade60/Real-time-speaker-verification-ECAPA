import os
from pydub import AudioSegment
from pydub.utils import which

# 🔥 Force pydub to use ffmpeg
AudioSegment.converter = which("ffmpeg")

INPUT_FOLDER = "data/pavan/raw_audio"
OUTPUT_FOLDER = "data/pavan/processed_audio"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = os.listdir(INPUT_FOLDER)

count = 1

for file in files:
    if file.lower().endswith((".mp4", ".m4a", ".mp3", ".wav")):

        input_path = os.path.join(INPUT_FOLDER, file)
        output_path = os.path.join(OUTPUT_FOLDER, f"pavan_{count:02d}.wav")

        print(f"Processing: {file}")

        audio = AudioSegment.from_file(input_path)

        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")

        print(f"✅ Saved: pavan_{count:02d}.wav")
        count += 1

print("\n🎯 Conversion completed successfully!")