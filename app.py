import os
import re
import ast # Used to safely evaluate the string representation of the list from Gemini
from dotenv import load_dotenv
from flask import Flask, render_template, request
from googleapiclient.discovery import build
import google.generativeai as genai

# --- SETUP ---
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found. Please check your .env file.")
app = Flask(__name__)
genai.configure(api_key=API_KEY)


# --- HELPER FUNCTION ---
def get_video_id(url):
    """Extracts the YouTube video ID from a URL."""
    regex = r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=|shorts\/)|youtu\.be\/)([^\"&?\/\s]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

# --- ROUTES ---
@app.route('/')
def home():
    """Renders the homepage."""
    return render_template('index.html')


@app.route('/analyse', methods=['POST'])
def analyse():
    """Handles the analysis of the YouTube video."""
    url = request.form['youtube_url']
    video_id = get_video_id(url)

    if not video_id:
        return "Error: Invalid YouTube URL provided. Please go back and try again."

    try:
        # === Part 1: Get YouTube Data ===
        youtube = build('youtube', 'v3', developerKey=API_KEY)

        video_request = youtube.videos().list(part="snippet", id=video_id)
        video_response = video_request.execute()
        video_snippet = video_response['items'][0]['snippet']
        video_title = video_snippet['title']
        video_thumbnail = video_snippet['thumbnails']['high']['url']

        request_comments = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=1000, order='relevance')
        response_comments = request_comments.execute()

        comments = []
        if 'items' in response_comments:
            for item in response_comments['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
        
        if not comments:
            return "Could not find any comments on this video."

        # === Part 2: Get Sentiment from Gemini ===
        prompt = f"""
        You are a sentiment analysis expert. Your task is to analyze YouTube comments.
        Classify each comment's sentiment strictly as 'Positive', 'Negative', or 'Neutral'.
        You MUST provide a sentiment for every single comment. If a comment is ambiguous, in another language, or you cannot determine the sentiment for any reason, you MUST classify it as 'Neutral'. Do NOT skip any comments.
        Your entire response must be ONLY a valid Python list of strings, like ["Positive", "Negative", "Neutral", ...].
        The list must have exactly {len(comments)} items in it.
        Do not add any introductory text, explanations, or code formatting like ```python. Your response must start with '[' and end with ']'.

        Here are the comments:
        {comments}
        """
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response_gemini = model.generate_content(prompt)
        
        raw_ai_response = response_gemini.text
        match = re.search(r'\[.*\]', raw_ai_response, re.DOTALL)
        if not match:
            return "Error: The AI response did not contain a valid list. Check the terminal for the raw response."
        
        list_str = match.group(0)
        sentiments = ast.literal_eval(list_str)
        
        # === THE FINAL, PRAGMATIC FIX ===
        # If the AI gives us more sentiments than we have comments, trim the excess.
        if len(sentiments) > len(comments):
            sentiments = sentiments[:len(comments)]
        # ================================

        if len(sentiments) != len(comments):
             print(f"--- LENGTH MISMATCH ---")
             print(f"Expected: {len(comments)} comments")
             print(f"Received: {len(sentiments)} sentiments")
             print(f"-----------------------")
             return "Error: The sentiment analysis returned a different number of results than comments. Check the terminal for details."

        # === Part 3: Calculate Results ===
        positive_count = sentiments.count('Positive')
        negative_count = sentiments.count('Negative')
        neutral_count = sentiments.count('Neutral')
        total_comments = len(sentiments)
        
        analyzed_comments = list(zip(comments, sentiments))

        # === Part 4: Render the Results Page ===
        return render_template(
            'results.html',
            video_title=video_title,
            video_thumbnail=video_thumbnail,
            total_comments=total_comments,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            analyzed_comments=analyzed_comments[:10]
        )

    except Exception as e:
        print(f"--- DETAILED ERROR --- \n{e}\n----------------------")
        return f"An error occurred: {e}"

# --- RUN THE APP ---
if __name__ == "__main__":
    app.run(debug=True)

