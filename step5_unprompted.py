import os
from openai import OpenAI
from dotenv import load_dotenv

print("=== STEP 5: WITHOUT Prompt ===")

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

with open("audio/CA138clip.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=f
    )

print(f"Text length: {len(result.text)} characters")
print(f"\nFirst 100 chars:\n{result.text[:100]}")
print(f"\n✓ Step 5 complete")
