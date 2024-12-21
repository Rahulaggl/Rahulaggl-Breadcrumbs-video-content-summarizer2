import os
import streamlit as st
import moviepy.editor as mp
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import ModelInference
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.DEBUG)

# Set your IBM Watson credentials here
WATSONX_API_KEY = 'ViKnyQacbz2FQkopZF0c7E41wBXPFkGjes1cJaPZ82GJ'
SERVICE_URL = 'https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/445c4600-65b0-46fc-b863-7e159da20a8e'

# Initialize Watsonx AI configuration
def configure_watsonx():
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": WATSONX_API_KEY
    }
    project_id = "471bb780-d5d5-4dbe-ae85-c90faafca3e5"
    model_id = "mistralai/mixtral-8x7b-instruct-v01"

    parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 150
    }

    model = ModelInference(
        model_id=model_id, 
        params=parameters, 
        credentials=credentials,
        project_id=project_id)
    return model

# Function to convert MP4 to WAV
def convert_mp4_to_wav(mp4_file):
    try:
        audio_clip = mp.AudioFileClip(mp4_file)
        wav_file = mp4_file.replace('.mp4', '.wav')
        audio_clip.write_audiofile(wav_file)
        logging.info(f"MP4 converted to WAV: {wav_file}")
        return wav_file
    except Exception as e:
        logging.error(f"Error converting MP4 to WAV: {e}")
        return None

# Function to transcribe audio using IBM Watson Speech-to-Text
def transcribe_audio(audio_file):
    try:
        with open(audio_file, 'rb') as audio:
            response = speech_to_text.recognize(
                audio=audio,
                content_type='audio/wav',  # Ensure the correct audio type is used
                model='en-US_BroadbandModel'
            ).get_result()

        if response['results']:
            transcribed_text = response['results'][0]['alternatives'][0]['transcript']
            logging.info(f"Transcription successful: {transcribed_text}")
            return transcribed_text
        else:
            logging.error("No speech recognized.")
            return None
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return None

# Function to generate captions and tags using Watsonx
def generate_caption_and_tags(summary: str) -> tuple:
    model = configure_watsonx()

    # Generate Caption
    caption_prompt = f"Create a catchy social media caption based on the following text: {summary}"
    caption_response = model.generate(caption_prompt)
    caption = caption_response.get('results')[0]['generated_text']

    # Generate Viral Hashtags
    viral_tags_prompt = f"Generate relevant social media tags to make this content viral: {summary}"
    viral_tags_response = model.generate(viral_tags_prompt)
    viral_tags = viral_tags_response.get('results')[0]['generated_text']

    # Generate Famous People to Tag
    famous_tags_prompt = f"Suggest the top five famous people in the relevant field to tag based on this content: {summary}"
    famous_tags_response = model.generate(famous_tags_prompt)
    famous_tags = famous_tags_response.get('results')[0]['generated_text']

    return caption, viral_tags, famous_tags

# Streamlit App UI
def main():
    st.title("Video to Text Transcription and Social Media Optimization")

    # File upload section
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded file to disk
        video_path = os.path.join(".", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Convert MP4 to WAV
        wav_file = convert_mp4_to_wav(video_path)
        if wav_file:
            # Transcribe the audio from the WAV file
            transcribed_text = transcribe_audio(wav_file)
            if transcribed_text:
                # Generate and display Caption, Tags, Famous People
                st.subheader("Catchy Caption")
                caption, viral_tags, famous_tags = generate_caption_and_tags(transcribed_text)
                st.write(caption)

                st.subheader('Famous People to Tag')
                st.write(famous_tags)

                st.subheader('Viral Hashtags')
                st.write(viral_tags)

            else:
                st.error("Failed to transcribe the audio.")
        else:
            st.error("Failed to convert the video to audio.")

        # Clean up the files after processing
        os.remove(video_path)
        if wav_file:
            os.remove(wav_file)

if __name__ == "__main__":
    main()
