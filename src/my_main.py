from pathlib import Path
import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

PROJECT_PATH = "building-apps-with-ai-tools-chatgpt-semantic-kernel-langchain-4469616\src"
    
def get_transcript(audio_file_name: list[str]) -> str:
    with open(Path.cwd()/PROJECT_PATH/audio_file_name, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


def generate_librarian_thoughts(transcript: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a librarian bot, return book recommendation with really short reasoning based on provided text."},
                  {"role": "user", "content": transcript}],
                  temperature=0.7,
                  max_tokens = 100
    )
    return response["choices"][0]["message"]["content"]

def main():
    audio_file_names: list[str] = ["03_06.mp3", "03_06_book_rec2.mp3"]
    transcripts: list[str] = []

    for audio_file_name in audio_file_names:
        transcript = get_transcript(audio_file_name)
        transcripts.append(transcript)

    for transcript in transcripts:
        result = generate_librarian_thoughts(transcript)
        print(f"User transcribed request: {transcript}")
        print(f"Librarian bot book recommendation: {result}")

if __name__ == "__main__":
    main()