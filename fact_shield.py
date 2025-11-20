import streamlit as st
import google.generativeai as genai
from streamlit_option_menu import option_menu
import easyocr
from PIL import Image
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import base64
import librosa
import os
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.ensemble import RandomForestClassifier


import streamlit as st
from streamlit_option_menu import option_menu
import google.generativeai as genai

# Configure page
st.set_page_config(page_title="News Verifier", layout="wide")


# Navigation menu
selected = option_menu(
    menu_title=None,
    options=["Home", "News Detector", "Sentiment"],
    icons=["house", "newspaper", "emoji-smile"],
    orientation="horizontal",
)

# Constants
POSITIVE_THRESHOLD = 0.1
NEGATIVE_THRESHOLD = -0.1
genai.configure(api_key = st.secrets["API_KEY"])

def analyze_content(content, is_url=False):
    """Analyze news content using Gemini"""
    verification_prompt = f"""
    Analyze this {'web content from URL' if is_url else 'news article'} for authenticity:
    "{content}"
    
    Provide response in this exact format:
    [Verdict: Fake🔴/Unverified/Likely Real🟢] 
    [Confidence: Low ⬇️/Medium /High ⬆️] 
    [Key Claims: 2-3 main claims from text]
    [Fact Check: Point-by-point verification] 
    [Logical Fallacies: List any detected] 
    [Sources 📰🔗: Credible verification sources]
    """
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(verification_prompt)
    return response.text            

# -------------------- Home Page --------------------
if selected == "Home":
        def get_base64(image_file):
            with open(image_file, "rb") as file:
                encoded = base64.b64encode(file.read()).decode()
                return f"data:image/jpg;base64,{encoded}"  # Adjust for image format (jpg/png)

    # Convert the local image to Base64
        bg_image = get_base64("home_logo1.jpeg")  # Make sure the file is in the same folder as app.py

    # Apply the custom CSS with the Base64-encoded image
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
        st.image("logo.gif", width=1500)

        
    # <div style='text-align: center'>
    #     <h1>Welcome to News Verifier</h1>
    #     <p>A comprehensive tool to check authenticity of news content from various sources</p>
    #     <h3>Features:</h3>
    #     <ul style='list-style: none; padding-left: 0'>
    #         <h1>📝 Text article verification</h1>
    #         <h1>🔗 URL content analysis</li>
    #         <li>📸 Image/Screenshot text extraction and verification</li>
    #         <li>📊 Sentiment analysis</li>
    #         <li>🕵️ Deepfake detection</li>
    #     </ul>
    # </div>
    # """, unsafe_allow_html=True)
    
# -------------------- News Detector --------------------
elif selected == "News Detector":
    def get_base64(image_file):
            with open(image_file, "rb") as file:
                encoded = base64.b64encode(file.read()).decode()
                return f"data:image/jpg;base64,{encoded}"  # Adjust for image format (jpg/png)

    bg_image = get_base64("news_logo3.jpg")  

    # Apply the custom CSS with the Base64-encoded image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    # st.markdown(
    #     """
    #     <style>
    #     /* Change the background color */
    #     .stApp {
    #         background-color: #00022b;
    #     }
    #     /* Change text color (optional) */
    #     .stMarkdown {
    #         color: #333;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )
    st.image("news_logo1.jpg")
    
    # st.markdown("""
    # <div style='text-align: center'>
    #     <h1 style='color:30, 175, 187'>🔍 News Detector</h1>
        
    # </div>
    # """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* Change tab font color and style */
    div[role="tablist"] button {
        font-size: 16px !important;
        font-weight: bold !important;
        color: red !important; /* Change to any color */
        font-family: "Copperplate", sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📝 Paste Article", "🔗 Enter URL", "📸 Upload image"])
    
    with tab1:
        article_text = st.text_area("Paste full news article", height=300, max_chars=5000)
        if st.button("Analyze Article"):
            if len(article_text) < 50:
                st.warning("Please paste a longer article for analysis")
            else:
                with st.spinner("🔍 Analyzing...",):
                    analysis = analyze_content(article_text)
                    st.session_state.last_analysis = analysis

    with tab2:
        url = st.text_input("Enter news article URL", placeholder="https://example.com/article")
        if st.button("Analyze URL"):
            if not url.startswith("http"):
                st.error("Please enter a valid URL")
            else:
                with st.spinner("🔍 Fetching content..."):
                    analysis = analyze_content(url, is_url=True)
                    st.session_state.last_analysis = analysis

    with tab3:
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            reader = easyocr.Reader(['en'])
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            with st.spinner("🔍 Extracting text..."):
                result = reader.readtext(img_array, detail=0)
                extracted_text = "\n".join(result)
                st.session_state.extracted_text = extracted_text

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, use_column_width=True)
            with col2:
                st.text_area("Extracted Text", value=extracted_text, height=300, disabled=True)
                
                if st.button("Analyze Extracted Text"):
                    if len(extracted_text) < 50:
                        st.warning("Not enough text for analysis")
                    else:
                        with st.spinner("🔍 Analyzing..."):
                            analysis = analyze_content(extracted_text)
                            st.session_state.last_analysis = analysis

    if 'last_analysis' in st.session_state:
        analysis = st.session_state.last_analysis
        verdict = "⚠️ Needs Verification"
        if "Verdict: Fake" in analysis:
            verdict = "🚨 Fake News Detected"
        elif "Verdict: Likely Real" in analysis:
            verdict = "✅ Likely Authentic"
        
        st.subheader("Verification Results")
        col = st.columns(1)
        if "Fake" in verdict:
            with col[0]:
                st.error(verdict)
        elif "Authentic" in verdict:
            with col[0]:
                st.success(verdict)
        else:
            with col[0]:
                st.warning(verdict)
        
        with st.expander("📋 Full Analysis"):
            st.markdown(analysis.replace("[", "**").replace("]", "**\n"))


# -------------------- Sentiment Analysis --------------------
elif selected == "Sentiment":

    def get_base64(image_file):
            with open(image_file, "rb") as file:
                encoded = base64.b64encode(file.read()).decode()
                return f"data:image/jpg;base64,{encoded}"  # Adjust for image format (jpg/png)

    # Convert the local image to Base64
    bg_image = get_base64("sentiment_logo2.jpg")  # Make sure the file is in the same folder as app.py

    # Apply the custom CSS with the Base64-encoded image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.image("sentiment_logo1.jpg")
    st.markdown(
    """
    <style>
    /* Change tab text color */
    .stTabs [data-baseweb="tab"] {
        color: black !important;  /* Change text color */
        font-size: 18px !important;  /* Change font size */
        font-weight: bold !important;
    }
    /* Change active tab background */
    .stTabs [aria-selected="true"] {
        background-color: #fe0000 !important;  /* Active tab background */
        color: white !important;  /* Active tab text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

    tab1, tab2 = st.tabs(["📝 Paste News", "🔗 Enter URL"])
    
    with tab1:
        news_text = st.text_area("Paste news content", height=250, placeholder="Enter / Paste news article ")
        analyze_text = st.button("Analyze Text")
    
    with tab2:
        news_url = st.text_input("Enter news URL", placeholder="https://example.com/article")
        analyze_url = st.button("Analyze URL")

    def analyze_sentiment_gemini(text):
        prompt = f"""Analyze this news sentiment considering:
        1. Emotional tone
        2. Word choice
        3. Contextual relationships
        4. Comparative language
        
        {text}
        
        Provide STRICT response format:
        [Overall Sentiment: Positive/Negative/Neutral]
        [Sentiment Score: -1.0 to 1.0]
        [Key Positive Phrases: comma list]
        [Key Negative Phrases: comma list] 
        [Bias Indicators: comma list]
        [Emotional Tone: comma list]
        [Summary: 50-word summary]
        """
        
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return self_validate_response(response.text)

    def self_validate_response(response):
        if "Sentiment Score:" not in response:
            raise ValueError("Missing score")
            
        score_str = response.split("Sentiment Score:")[1].split("\n")[0].strip()
        try:
            score = float(score_str)
            score = max(-1.0, min(1.0, score))
        except ValueError:
            score = 0.0

        sentiment = response.split("Overall Sentiment:")[1].split("\n")[0].strip()
        if (sentiment == "Positive" and score < POSITIVE_THRESHOLD) or \
           (sentiment == "Negative" and score > NEGATIVE_THRESHOLD):
            sentiment = "Neutral"
            
        return parse_sentiment_response(response)

    def parse_sentiment_response(response):
        result = {}
        matches = re.findall(r'\[(.*?):(.*?)\]', response, re.DOTALL)
        for key, value in matches:
            key = key.strip()
            value = value.strip().replace('\n', ' ')
            if key == "Sentiment Score":
                try:
                    value = float(value.split()[0])
                except (ValueError, IndexError):
                    value = 0.0
            result[key] = value
        return result

    def fetch_article_text(url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            return ' '.join(p.get_text() for p in soup.find_all('p'))
        except Exception as e:
            st.error(f"Error: {e}")
            return None

    analysis_result = None
    if analyze_text and news_text:
        if len(news_text) < 100:
            st.warning("Minimum 100 characters required")
        else:
            with st.spinner("🔍 Analyzing..."):
                analysis_result = analyze_sentiment_gemini(news_text)

    if analyze_url and news_url:
        if not news_url.startswith("http"):
            st.error("Invalid URL")
        else:
            with st.spinner("📥 Fetching..."):
                article_text = fetch_article_text(news_url)
                if article_text:
                    with st.spinner("🔍 Analyzing..."):
                        analysis_result = analyze_sentiment_gemini(article_text)

    if analysis_result:
        st.divider()
        st.subheader("Analysis Results")
        
        score = 0.0
        sentiment = "Neutral"
        if "Sentiment Score" in analysis_result:
            try:
                score = float(analysis_result["Sentiment Score"])
            except:
                pass
        if "Overall Sentiment" in analysis_result:
            sentiment = analysis_result["Overall Sentiment"]

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            color = "#2ecc71" if score > POSITIVE_THRESHOLD else \
                    "#e74c3c" if score < NEGATIVE_THRESHOLD else "#f1c40f"
                    
            st.markdown(f"""
            <div style="text-align: center">
                <h3 style="color: {color}">{sentiment}</h3>
                <div style="font-size: 3rem; color: {color}">{score:.2f}</div>
                <div style="height: 10px; background: {color}; 
                     width: {(abs(score)*100):.0f}%; margin: auto"></div>
            </div>
            """, unsafe_allow_html=True)

        cols = st.columns(2)
        with cols[0]:
            st.subheader("🟢 Positives")
            if "Key Positive Phrases" in analysis_result:
                for phrase in analysis_result["Key Positive Phrases"].split(","):
                    st.markdown(f"- {phrase.strip()}")
        
        with cols[1]:
            st.subheader("🔴 Negatives")
            if "Key Negative Phrases" in analysis_result:
                for phrase in analysis_result["Key Negative Phrases"].split(","):
                    st.markdown(f"- {phrase.strip()}")



        with st.expander("📄 Summary"):
            st.write(analysis_result.get("Summary", ""))
            
        with st.expander("🔍 Bias Analysis"):
            if "Bias Indicators" in analysis_result:
                for bias in analysis_result["Bias Indicators"].split(","):
                    st.markdown(f"- {bias.strip()}")
        
        with st.expander("🎭 Emotional Analysis"):
            if "Emotional Tone" in analysis_result:
                for tone in analysis_result["Emotional Tone"].split(","):
                    st.markdown(f"- {tone.strip()}")



