import os
from openai import OpenAI
from dotenv import load_dotenv

print("=== Simple Whisper Test ===")

# Load .env
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
print(f"API Key loaded: {api_key is not None}")

if not api_key:
    print("ERROR: No API key found!")
    exit()

# Setup client
client = OpenAI(api_key=api_key)

# Check audio file
audio_file = "audio/CA138clip.mp3"
if not os.path.exists(audio_file):
    print(f"ERROR: File not found: {audio_file}")
    exit()

print(f"Audio file: {audio_file}")

# Transcribe
print("\nTranscribing...")
try:
    with open(audio_file, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    print("✓ Success!")
except Exception as e:
    print(f"✗ Error: {e}")
    exit()

# Show results
print(f"\nText length: {len(result.text)} characters")
print("\nFirst 200 characters:")
print("-" * 40)
print(result.text[:200])
print("-" * 40)

# Save
os.makedirs("transcripts", exist_ok=True)
with open("transcripts/simple_test.txt", "w") as f:
    f.write(result.text)

print("\n✓ Saved to: transcripts/simple_test.txt")
print("=== Test Complete ===")
