# app.py content - Save this code to a file named 'app.py'

import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import io

# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_PATH = 'test_features.pt'
WEIGHTS_PATH = 'blip_finetuned_captioner.pth'
TOP_K_RETRIEVAL = 3

# --- 2. MODEL LOADING (Caching for efficiency) ---

@st.cache_resource
def load_models():
    # BLIP Model (Captioning)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    try:
        # Load fine-tuned weights
        blip_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    except FileNotFoundError:
        st.warning(f"BLIP fine-tuned weights ({WEIGHTS_PATH}) not found. Using base model.")
    blip_model.eval()

    # CLIP Model (Retrieval)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_model.eval()

    # Pre-calculated Features
    try:
        features_data = torch.load(FEATURE_PATH)
        st.success(f"Loaded {len(features_data['image_features'])} features for retrieval.")
        return blip_processor, blip_model, clip_processor, clip_model, features_data
    except FileNotFoundError:
        st.error(f"Feature file '{FEATURE_PATH}' not found. Retrieval demo disabled.")
        return blip_processor, blip_model, clip_processor, clip_model, None

blip_processor, blip_model, clip_processor, clip_model, features_data = load_models()

# --- 3. HELPER FUNCTIONS ---

def get_caption(pil_image):
    # Generate the caption
    inputs = blip_processor(images=pil_image, return_tensors="pt").to(DEVICE)
    out = blip_model.generate(**inputs, max_length=16, num_beams=5)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def retrieve_images(query_text, k=TOP_K_RETRIEVAL):
    if features_data is None: return [], []
    
    # Get Text Feature
    text_inputs = clip_processor(text=query_text, return_tensors="pt", padding=True).to(DEVICE)
    text_feature = clip_model.get_text_features(**text_inputs)
    text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)

    # Calculate Similarity with Saved Image Features
    img_features = features_data['image_features'].to(DEVICE)
    similarity = (text_feature @ img_features.T).squeeze(0)
    
    # Get Top K indices
    top_k_indices = torch.argsort(similarity, descending=True)[:k].cpu().numpy()
    
    # Fetch results
    retrieved_images = [features_data['images'][i] for i in top_k_indices]
    captions = [features_data['captions'][i] for i in top_k_indices]
    
    return retrieved_images, captions

# --- 4. STREAMLIT UI ---

st.set_page_config(layout="wide", page_title="BLIP & CLIP Demo")
st.title(" Visual-Language Model Demo")
st.markdown("---")

# Sidebar for controls
st.sidebar.header("Controls")
demo_mode = st.sidebar.radio("Select Demo Mode:", ["Image Captioning", "Semantic Search (Text to Image)"])

if demo_mode == "Image Captioning":
    st.header("1. Image Captioning (BLIP)")
    
    uploaded_file = st.file_uploader("Upload an Image to Caption", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(pil_image, caption='Uploaded Image', use_column_width=True)
            
        with col2:
            with st.spinner('Generating caption...'):
                caption = get_caption(pil_image)
            st.subheader("Generated Caption:")
            st.info(caption)

elif demo_mode == "Semantic Search (Text to Image)":
    st.header("2. Semantic Search (CLIP)")
    
    query_text = st.text_input(
        "Enter a description to search for similar images:",
        "a dog jumping in the grass"
    )
    
    if query_text and features_data:
        st.markdown(f"Searching for **'{query_text}'**...")
        
        #  Ensure the f-string is properly terminated here
        with st.spinner(f'Retrieving top {TOP_K_RETRIEVAL} similar images...'): 
            retrieved_images_tensor, captions = retrieve_images(query_text)
            
        st.subheader(f"Top {TOP_K_RETRIEVAL} Retrieved Images:")
        
        cols = st.columns(TOP_K_RETRIEVAL)
        
        for i, (img_tensor, cap) in enumerate(zip(retrieved_images_tensor, captions)):
            # Convert tensor back to PIL image for display
            # Tensors are [C, H, W] in [-1, 1] range. Denormalize to [0, 1] then convert to PIL
            img_tensor_denorm = (img_tensor + 1) / 2
            img_array = img_tensor_denorm.permute(1, 2, 0).cpu().numpy() # C H W -> H W C
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
            
            with cols[i]:
                st.image(img_pil, caption=f"Caption: {cap}", use_column_width=True)

    elif features_data is None:
        st.error("Cannot run Semantic Search. The 'test_features.pt' file is missing or failed to load.")

st.markdown("---")
st.caption("Models: BLIP (Finetuned for Captioning) and CLIP (Pre-trained for Search).")