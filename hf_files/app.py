import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
import os
from model import (
    Model1_MP_TR_BCE,
    Model2_MP_TR_DICE,
    Model3_Strided_TR_BCE,
    Model4_Strided_Bilinear_DICE,
)

# Configuration
IMG_SIZE = 128
DEVICE = torch.device("cpu") # Use CPU for HF Spaces default

# Define model paths
MODEL_PATHS = {
    "Model 1 (MP + TR + BCE)": "model1_best.pth",
    "Model 2 (MP + TR + Dice)": "model2_best.pth",
    "Model 3 (Strided + TR + BCE)": "model3_best.pth",
    "Model 4 (Strided + Bilinear + Dice)": "model4_best.pth",
}

# Model classes mapping
MODEL_CLASSES = {
    "Model 1 (MP + TR + BCE)": Model1_MP_TR_BCE,
    "Model 2 (MP + TR + Dice)": Model2_MP_TR_DICE,
    "Model 3 (Strided + TR + BCE)": Model3_Strided_TR_BCE,
    "Model 4 (Strided + Bilinear + Dice)": Model4_Strided_Bilinear_DICE,
}

# Load models
models = {}

def load_models():
    print("Loading models...")
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"Warning: Checkpoint not found for {name} at {path}. Skipping.")
            continue
        
        try:
            model_cls = MODEL_CLASSES[name]
            model = model_cls().to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            models[name] = model
            print(f"Loaded {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    print("Model loading complete.")

load_models()

# Transform
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

def predict(image):
    if image is None:
        return [None] * len(MODEL_PATHS)
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    outputs = []
    
    # Run inference for all defined keys to maintain order
    for name in MODEL_PATHS.keys():
        if name in models:
            model = models[name]
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.sigmoid(logits)
                # Binarize for visualization
                mask = (probs > 0.5).float()
                
                # Convert back to PIL
                mask_np = mask.squeeze().cpu().numpy()
                # Scale to 0-255 for display
                mask_img = Image.fromarray((mask_np * 255).astype('uint8'))
                outputs.append((mask_img, name))
        else:
            # Create a placeholder for missing models
            placeholder = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(200, 200, 200))
            outputs.append((placeholder, f"{name} (Missing)"))
            
    return outputs

# Gradio Interface
with gr.Blocks(title="UNet Segmentation Model Comparison") as demo:
    gr.Markdown("# UNet Segmentation Model Comparison")
    gr.Markdown("""
    **Dataset:** Oxford-IIIT Pet Dataset | **Task:** Binary Segmentation | **Input Size:** 128x128

    ### Compare 4 different UNet-like architectures:
    
    *   **Model 1**: Standard MaxPool downsampling + Transposed Conv upsampling + BCE Loss
    *   **Model 2**: Standard MaxPool downsampling + Transposed Conv upsampling + Dice Loss
    *   **Model 3**: Strided Convolution downsampling + Transposed Conv upsampling + BCE Loss
    *   **Model 4**: Strided Convolution downsampling + Bilinear upsampling + Dice Loss
    
    Upload an image to see how each architectural choice affects the segmentation output.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            submit_btn = gr.Button("Segment Image", variant="primary")
            
        with gr.Column():
            # Create a gallery or individual images. Gallery is nice for comparison.
            # But specific labeled outputs might be clearer.
            # Let's use a Gallery
            output_gallery = gr.Gallery(label="Segmentation Results", columns=2, height='auto')

    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=output_gallery
    )

    # Example images if any
    # gr.Examples(examples=["data/example.jpg"], inputs=input_image)

if __name__ == "__main__":
    demo.launch()
