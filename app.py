import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gradio as gr
from huggingface_hub import hf_hub_download

# Download models from Hugging Face Hub
ann_model_path = hf_hub_download(repo_id="Shyam-Mashru/Classification", filename="model_ann.h5")
cnn_model_path = hf_hub_download(repo_id="Shyam-Mashru/Classification", filename="model_cnn.h5")

# Load models
ann_model = load_model(ann_model_path)
cnn_model = load_model(cnn_model_path)

# Define EMNIST character classes
emnist_classes = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 107)]

# Preprocessing function
def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = 1 - image  # Invert colors
    return image

# Prediction function
def predict_character(image, model_choice):
    processed_image = preprocess_image(image)
    
    if model_choice == "ANN":
        input_data = processed_image.flatten().reshape(1, 784)
        prediction = ann_model.predict(input_data)
    elif model_choice == "CNN":
        input_data = processed_image.reshape(1, 28, 28, 1)
        prediction = cnn_model.predict(input_data)
    else:
        return "Invalid Model Choice"
    
    predicted_class = np.argmax(prediction)
    predicted_character = emnist_classes[predicted_class]
    confidence = np.max(prediction) * 100
    
    return f"Prediction: {predicted_character} (Confidence: {confidence:.2f}%)"

# Interface
def interface(image, model_choice):
    return predict_character(image, model_choice)

with gr.Blocks() as gui:
    gr.Markdown("## Handwritten Character Classification")
    gr.Markdown(
        "Upload a handwritten character image (digit, uppercase letter, or lowercase letter), "
        "and select the model (ANN or CNN) for prediction."
    )
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Handwritten Character Image")
            model_input = gr.Radio(
                choices=["ANN", "CNN"], value="CNN", label="Select Model"
            )
        with gr.Column():
            output_label = gr.Label(label="Prediction")
    
    predict_button = gr.Button("Predict")
    predict_button.click(
        interface,
        inputs=[image_input, model_input],
        outputs=output_label
    )

# Launch the GUI
gui.launch(share=True)
