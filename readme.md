# Multilingual Speech Translation

This project is a **Multilingual Speech Translation** application that transcribes audio input using OpenAI Whisper and translates the transcription into a selected target language using Facebook's MBart-50 model. The application is built using **Gradio**, **Hugging Face Transformers**, and **LangChain**.

## 🚀 Features
- **Speech-to-Text (ASR)**: Converts spoken audio to text using OpenAI Whisper.
- **Language Detection**: Automatically detects the spoken language.
- **Text Translation**: Translates detected text into a specified target language using MBart-50.
- **Gradio UI**: Provides a user-friendly web interface for audio input and translation.

## 🛠️ Technologies Used
- **Python**
- **OpenAI Whisper** (Automatic Speech Recognition)
- **Facebook MBart-50** (Machine Translation)
- **LangChain** (LLM pipeline framework)
- **Pydantic v2** (Data validation & serialization)
- **Gradio** (Web UI for easy interaction)
- **Hugging Face API** (For deploying models)

## 📂 Project Structure
```
.
├── app.py             # Main application file
├── requirements.txt   # Python dependencies
├── working_field.ipynb         # Working details
├── README.md          # Documentation
```

## 📥 Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/darkangrycoder/multilingual-speech-translation.git
cd multilingual-speech-translation
```

### 2️⃣ Create a Virtual Environment & Install Dependencies
```sh
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```sh
python app.py
```

## 🖥️ Usage
1. Upload an audio file in the **Gradio UI**.
2. Select the **Target Language** from the dropdown.
3. Click **Submit** to transcribe and translate the audio.
4. View the **Transcription, Detected Language, and Translation** results.

## 🏗️ Deployment
### Deploy on Hugging Face Spaces
1. Push your project to Hugging Face Hub:
   ```sh
   huggingface-cli login
   git push https://huggingface.co/spaces/tdnathmlenthusiast/multilingual_transcription
   ```
2. Hugging Face will automatically install dependencies from **requirements.txt** and deploy the app.

### Alternative: Deploy using Docker
```sh
docker build -t multilingual-translation .
docker run -p 7860:7860 multilingual-translation
```

## 🔧 Troubleshooting
- If you face dependency issues on Hugging Face Spaces, make sure you are using **Pydantic v2** instead of `root_validator` (deprecated in v2).
- If Whisper ASR is slow, try using `whisper.load_model("tiny")` instead of `base` or `large` models.
- For GPU acceleration, use `torch` with CUDA.

## 📝 License
This project is licensed under the **MIT License**.

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository, make changes, and submit a **pull request**.

## 📧 Contact
For any inquiries, reach out to [debnathtirtha391@gmail.com]

