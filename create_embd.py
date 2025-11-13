#%%


from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

IMG_DIR = Path("cl_test/images/test")

# Load data
print('\nLoading test dataset...')
test_df = pd.read_csv('/home/pushpaka/ml_test/dataset/test.csv')
print(f'Test size: {len(test_df):,}')

#%%

IMAGE_MODEL = "facebook/dinov2-large"

print(f'\nLoading {IMAGE_MODEL} on all GPUs...')
dinov2_model = AutoModel.from_pretrained(
    IMAGE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)
dinov2_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL)
dinov2_model.eval()
print('✓ DINOv2-large loaded')

def extract_dinov2_embedding(image_link, img_dir, model, processor):
    """Extract DINOv2 embedding from image (multi-GPU aware)"""
    try:
        img_dir = Path(img_dir) if isinstance(img_dir, str) else img_dir
        filename = Path(image_link).name
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            filename += '.jpg'
        img_path = img_dir / filename
        
        if not img_path.exists():
            return np.zeros(1024, dtype=np.float32)
        
        image = Image.open(img_path).convert('RGB')
        inputs = processor(images=image, return_tensors='pt')
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # DINOv2 uses CLS token from last_hidden_state
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
        
        return embedding.astype(np.float32)
    
    except Exception as e:
        print(f'Error processing {image_link}: {e}')
        return np.zeros(1024, dtype=np.float32)

def extract_image_embeddings(df, img_dir, model, processor, batch_size=16):
    """Process images with batching"""
    embeddings = []
    for i in tqdm(range(0, len(df), batch_size), desc="Processing images"):
        batch_links = df['image_link'].iloc[i:i+batch_size]
        for link in batch_links:
            emb = extract_dinov2_embedding(link, img_dir, model, processor)
            embeddings.append(emb)
        torch.cuda.empty_cache()
    return np.array(embeddings, dtype=np.float32)

print('\nExtracting test image embeddings...')
test_image = extract_image_embeddings(test_df, IMG_DIR, dinov2_model, dinov2_processor)

np.save('test_dinov2_large.npy', test_image)
print(f'\n✓ Test image embeddings shape: {test_image.shape}')

del dinov2_model, dinov2_processor
torch.cuda.empty_cache()
print('✓ DINOv2 embeddings saved')

#%%
TEXT_MODEL = "Qwen/Qwen3-Embedding-4B"  # 3584-dim, top MTEB performer

print(f'Loading {TEXT_MODEL}...')
text_model = AutoModel.from_pretrained(
    TEXT_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True  # Required for GTE models
)
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, trust_remote_code=True)
text_model.eval()
print('✓ Text model loaded')

def extract_text_embeddings_gte(df, model, tokenizer, batch_size=16):
    """
    Extract GTE-Qwen2 text embeddings
    Note: Smaller batch size due to 7B model size
    """
    embeddings = []
    texts = df.apply(
    lambda row: f"Instruct: Retrieve product information\nQuery: {row['catalog_content']}",
    axis=1
).to_list()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings (GTE)"):
        batch_texts = texts[i:i+batch_size]
        
        try:
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embs = outputs.last_hidden_state[:, -1].cpu().numpy()
                embeddings.extend(batch_embs)
        
        except Exception as e:
            print(f"Error processing batch: {e}")
            embeddings.extend([np.zeros(3584, dtype=np.float32)] * len(batch_texts))
        
        torch.cuda.empty_cache()
    
    return np.array(embeddings, dtype=np.float32)


train_df = pd.read_csv('subset_train_70k.csv')
val_df = pd.read_csv('subset_val_5k.csv')
train_text_emb = extract_text_embeddings_gte(train_df, text_model, text_tokenizer)
val_text_emb = extract_text_embeddings_gte(val_df, text_model, text_tokenizer)
del text_model, text_tokenizer
torch.cuda.empty_cache()
np.save('train_Qwen3-Embedding-4B.npy', train_text_emb)
np.save('val_Qwen3-Embedding-4B.npy', val_text_emb)

