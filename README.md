# Image and Video Caption Generator

This project is an AI-powered caption generator that automatically creates textual descriptions for images and videos using Transformers from Hugging Face. It leverages state-of-the-art vision-language models such as BLIP, ViT + GPT2, or other multimodal models to generate accurate captions.

Features
1. Image Captioning – Generates descriptive captions for images.
2. Video Captioning – Extracts frames from a video and generates captions.
3. Transformer-based Model – Uses pre-trained models from Hugging Face for high-quality captions.
4. Fast & Scalable – Supports batch processing for multiple images/videos.
5. Fine-Tuning Option – Can be fine-tuned on custom datasets.

Requirements :
Python 3.x
Jupyter Notebook
PyTorch
Transformers (Hugging Face)
OpenCV
PIL (Pillow)
NumPy
Matplotlib

You can install dependencies using:
pip install torch torchvision transformers opencv-python numpy pillow matplotlib


How It Works
1. Feature Extraction – Uses a Vision Transformer (ViT) or BLIP to extract meaningful visual features.
2. Caption Generation – A pre-trained transformer model (e.g., BLIP, ViT-GPT2, OFA) generates text descriptions.
3. Video Captioning – Extracts key frames from videos and applies the image captioning model.

# Clone the repository.

1.Open the image and video caption generator.ipynb notebook.
2.Load the pre-trained model from Hugging Face
3. Run the model on an image


Example Output
  Input Image:
  (Displays an image of a dog running on the beach)
  Generated Caption:
  "A dog is running along the beach on a sunny day."
