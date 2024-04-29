from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import streamlit as st

# Load the model only once and cache the result
@st.cache(allow_output_mutation=True)
def load_model():
    # Load Whisper model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    return model,processor

def load_audio(filepath):
  """Loads a WAV audio file.

  Args:
      filepath (str): Path to the WAV audio file.

  Returns:
      tuple: A tuple containing the loaded audio data (numpy array)
              and the original sample rate (int).

  Raises:
      FileNotFoundError: If the specified audio file is not found.
  """

  try:
    # Load audio with Librosa (supports WAV)
    audio, sample_rate = librosa.load(filepath)
    return audio, sample_rate
  except FileNotFoundError:
    raise FileNotFoundError(f"Audio file not found: {filepath}")

model, processor = load_model()

st.title("Record and Analyze Audio with Streamlit and librosa")

uploaded_file = st.file_uploader("Choose a WAV format file")


if uploaded_file is not None:
    # Recorded audio data is available in 'recorded_audio' variable
    st.audio(uploaded_file, format='audio/wav')  # Play as WAV
    audio, sample_rate = load_audio(uploaded_file)
else:
    st.stop()


if st.button('Predict!'):
    # Make prediction with Whisper
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    # generate token ids
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    st.title(f":green[{transcription[0]}]")
else:
    st.stop()


