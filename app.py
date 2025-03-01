import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import whisper
from langchain.chains.base import Chain
from typing import Dict, Any, ClassVar
import gradio as gr

# -------------------------------
# Load models once (this may take some time)

# Load a smaller Whisper ASR model ("tiny") for quick testing.
asr_model = whisper.load_model("tiny")

# Load MBart model & tokenizer for translation
translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# -------------------------------
# Functions from your code

def transcribe_audio(audio_path: str) -> (str, str):
    """
    Uses Whisper to transcribe audio and detect the language.
    Returns:
      - transcription (str): the transcribed text.
      - detected_language (str): MBart language code (e.g., "en_XX", "fr_XX").
    """
    result = asr_model.transcribe(audio_path)
    # Map ISO 639-1 codes to MBart language codes
    iso_to_mbart = {
        "en": "en_XX",
        "fr": "fr_XX",
        "de": "de_DE",
        "es": "es_XX",
        "hi": "hi_IN",
        # Add other mappings as needed.
    }
    detected_iso = result.get("language", "en")  # default to English
    detected_lang = iso_to_mbart.get(detected_iso, "en_XX")
    return result["text"], detected_lang

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translates the input text from src_lang to tgt_lang using MBart.
    """
    tokenizer.src_lang = src_lang  # Set source language for tokenizer
    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    encoded_text = tokenizer(text, return_tensors="pt")
    generated_tokens = translation_model.generate(**encoded_text, forced_bos_token_id=forced_bos_token_id)
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

class AudioTranslationChain(Chain):
    """Custom LangChain chain that transcribes audio and then translates the text."""
    input_keys: ClassVar[list] = ["audio_path", "target_lang"]
    output_keys: ClassVar[list] = ["transcription", "detected_lang", "translation"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = inputs["audio_path"]
        target_lang = inputs["target_lang"]  # e.g., "en_XX", "fr_XX", etc.
        # Step 1: Transcribe audio
        transcription, detected_lang = transcribe_audio(audio_path)
        # Step 2: Translate if detected language differs from target
        if detected_lang != target_lang:
            translation = translate_text(transcription, src_lang=detected_lang, tgt_lang=target_lang)
        else:
            translation = transcription  # No translation needed if already in target language
        return {
            "transcription": transcription,
            "detected_lang": detected_lang,
            "translation": translation
        }

    @property
    def _chain_type(self) -> str:
        return "audio_translation_chain"

# -------------------------------
# Gradio Interface Function

def process_audio(audio_path: str, target_lang: str):
    """
    Takes an audio file path and target language code,
    uses AudioTranslationChain to process the audio,
    and returns detected language, transcription, and translation.
    """
    chain = AudioTranslationChain()
    result = chain({"audio_path": audio_path, "target_lang": target_lang})
    return result["detected_lang"], result["transcription"], result["translation"]

# Define available target language choices (using MBart language codes)
target_lang_choices = ["en_XX", "fr_XX", "de_DE", "es_XX", "hi_IN"]

# Create a Gradio interface:
iface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),  
        gr.Dropdown(choices=target_lang_choices, label="Target Language", value="en_XX")
    ],
    outputs=[
        gr.Textbox(label="Detected Language"),
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Translation")
    ],
    title="Multilingual Speech Translation",
    description="Upload an audio file and select the target language. The app will transcribe the audio using Whisper and then translate the transcription using Facebook's MBart-50."
)

if __name__ == "__main__":
    iface.launch()
