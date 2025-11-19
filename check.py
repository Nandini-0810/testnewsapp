import google.generativeai as genai

# Configure your Gemini API key
genai.configure(api_key="AIzaSyCQF7h2HcmmX34cfbMGOW2z5wmrQSjWKHA")

# List all available models for your account
for model in genai.list_models():
    print(model.name)
