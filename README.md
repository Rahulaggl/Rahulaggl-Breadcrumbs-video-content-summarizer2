### **Video Content Summarizer for Social Media Optimization**  

Here is the updated GitHub description with the link placed at the top:

**Access the Project:**

1. **Offline Access**:  
   To run the project locally, follow the setup instructions below.
   
3. **Online Access**:  
   You can access the project online via the following [link](https://video-content-summarizer-oyfpmfdnv28v4qf8jwhp2j.streamlit.app/).
## Demo Video

Here is the demo video:[link](https://youtu.be/-olEfVXiUik).




---

This repository implements a Video Content Summarizer that analyzes YouTube video content, extracts key insights, and generates optimized outputs for social media engagement. The system leverages AI-powered models to provide captions, hashtags, and tagging recommendations for increasing video virality and audience reach.

### Features

- **YouTube Transcript Extraction**  
  Extracts video transcripts using the YouTube Transcript API for further analysis.

- **Video Summarization**  
  Generates a concise summary of video content, enabling quick comprehension of the core message.

- **Caption Generator**  
  Creates engaging and personalized captions for social media platforms.

- **Hashtag Recommendations**  
  Generates two types of hashtags:
  - **Viral Hashtags**: Optimized for audience engagement.
  - **Field-Specific Tags**: Suggestions of relevant famous personalities or influencers to tag based on video content.

- **Streamlit User Interface**  
  Provides an easy-to-use interface for uploading video URLs and receiving outputs.

### Workflow

1. **Input Video URL**  
   Paste the YouTube video link into the application.

2. **Transcript Extraction**  
   Automatically fetches the transcript of the video.

3. **Content Processing**  
   Summarizes the transcript into clear insights.  
   Generates suggested captions and optimized hashtags.

4. **Tagging Recommendations**  
   Identifies famous personalities or relevant figures related to the video topic for tagging.

5. **Output**  
   Summarized transcript.  
   Suggested captions.  
   Viral hashtags and famous people to tag.

**Sample Outputs**

- **Caption:**  
  "Unlock your potential with AI-powered insights! 🚀 #ArtificialIntelligence #GrowthMindset"

- **Viral Hashtags:**  
  #Trending #ViralVideo #AI #ContentCreator #Inspiration

- **Famous Personalities to Tag:**  
  - **In Tech:** Elon Musk, Sundar Pichai, Satya Nadella  
  - **In Fitness:** Chris Hemsworth, Michelle Lewin

### Setup Instructions

To run the project locally:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<your-username>/video-content-summarizer.git  
   cd video-content-summarizer  
   ```

2. **Install Dependencies**  
   Install required Python libraries:

   ```bash
   pip install -r requirements.txt  
   ```

3. **Add API Keys**  
   Create a `.env` file and add the required keys:

   ```plaintext
   WATSONX_API_KEY=your_ibm_watsonx_api_key  
   PROJECT_ID=your_project_id  
   ```

4. **Run the Application**  
   Start the Streamlit interface:

   ```bash
   streamlit run app.py  
   ```

5. **Input the Video URL**  
   Paste the YouTube video link and let the system generate optimized outputs.

### Project Structure

- **app.py**: Streamlit app for user interaction.
- **transcript_extraction.py**: Fetches YouTube video transcripts.
- **summary_generation.py**: Generates video summaries and key insights.
- **hashtag_suggester.py**: Generates viral hashtags and tagging recommendations.
- **requirements.txt**: Lists project dependencies.

### Technologies Used

- **Streamlit**: Interactive User Interface.
- **YouTube Transcript API**: For extracting video transcripts.
- **IBM Watsonx AI**: To generate summaries, captions, and recommendations.
- **Python Libraries**: Pysbd, Pandas, and Requests for text processing.

### Future Enhancements

- Integration with social media platforms for auto-posting.
- Support for other video platforms (e.g., Vimeo).
- Enhanced AI models for caption personalization.

---
