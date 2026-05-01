import google.generativeai as genai

genai.configure(api_key="AIzaSyBakZ96SfKsrisPRG-VYNF4dSCMA9QS5l0")

print("Available models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✅ {model.name}")