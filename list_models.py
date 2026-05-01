import google.generativeai as genai

genai.configure(api_key="AIzaSyA3Kmwr8-8HgNQlnBF-eFCOuz1kACOWC_I")

print("Available models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✅ {model.name}")