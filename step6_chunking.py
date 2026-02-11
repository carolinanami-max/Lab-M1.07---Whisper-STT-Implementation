import os
from pydub import AudioSegment

print("=== STEP 6: Audio Chunking ===")
audio_file = "audio/CA138clip.mp3"
print(f"Audio file: {audio_file}")
audio = AudioSegment.from_file(audio_file)
duration_seconds = len(audio) / 1000
print(f"Duration: {duration_seconds:.1f} seconds")
chunk_length = 30
num_chunks = int(duration_seconds / chunk_length) + 1
print(f"Would create {num_chunks} chunks of {chunk_length} seconds each")
os.makedirs("chunks", exist_ok=True)
with open("chunks/chunking_info.txt", "w") as f:
    f.write(f"Original file: {audio_file}\\n")
    f.write(f"Duration: {duration_seconds:.1f} seconds\\n")
    f.write(f"Chunk size: {chunk_length} seconds\\n")
    f.write(f"Number of chunks: {num_chunks}\\n")
    f.write("\\nFor longer files (>5 minutes), chunking helps with:\\n")
    f.write("1. API file size limits (25MB)\\n")
    f.write("2. Better error handling\\n")
    f.write("3. Parallel processing\\n")
print()
print("✓ Chunking strategy saved to: chunks/chunking_info.txt")
print("=== STEP 6 Complete ===")
