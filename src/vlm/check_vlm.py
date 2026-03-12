import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from PIL import Image
import cv2
import yaml
import glob
import os

# Load config
with open("config/vlm_config.yaml") as f:
    config = yaml.safe_load(f)['vlm']

model_name = config['model']['name']
model_path = config['model']['local_path']

# Create quantization config if using 4-bit
quantization_config = None
if config['model']['use_4bit']:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

# Check if model exists
if not os.path.exists(os.path.join(model_path, 'config.json')):
    print(f"Downloading {model_name} to {model_path}...")
    os.makedirs(model_path, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
else:
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    
    # Try loading with different approaches
    try:
        # First attempt: with quantization config
        model = AutoModel.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True
        )
    except (TypeError, AttributeError) as e:
        print(f"First loading attempt failed: {e}")
        print("Trying alternative loading method...")
        
        # Second attempt: without quantization
        model = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True
        )

model.eval()

# Get first video
videos = glob.glob("data/test_videos/*.mp4")
if not videos:
    print("No videos found")
    exit()

# Process first frame
cap = cv2.VideoCapture(videos[0])
ret, frame = cap.read()
cap.release()

if ret:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    prompt = config['prompts']['moderation']
    
    # For GLM-4V, we need to handle the input differently
    # The exact format might vary based on the model version
    try:
        # Method 1: Using chat template (if supported)
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "image": image, "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        
        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
    except (AttributeError, TypeError):
        # Method 2: Direct image+text input for GLM-4V
        print("Using direct image+text input method")
        inputs = tokenizer(
            prompt,
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config['generation']['max_new_tokens'],
            temperature=config['generation']['temperature'],
            top_p=config['generation']['top_p'],
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print(response)
    print("="*50)

# Also print model info for debugging
print(f"\nModel type: {type(model).__name__}")
print(f"Model device: {model.device}")