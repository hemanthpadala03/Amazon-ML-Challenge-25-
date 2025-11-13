#%%
import re
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# Config
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f'Using device: {DEVICE}')

#%%
# Feature Engineering Utilities
print('Setting up feature engineering utilities...')

currency_regex = re.compile(r"(?i)(rs\.?|inr|usd|eur|gbp|cad|aud|\$|₹|€|£)\s*\d[\d,]*\.?\d*")
mrp_regex = re.compile(r"(?i)\b(mrp|price|list\s*price|deal|now|was|save)\b.{0,12}?\d[\d,]*\.?\d*")
standalone_number_near_price = re.compile(r"(?i)(mrp|price|deal|now|was|save)\D{0,8}(\d[\d,]*\.?\d*)")

pack_of_regex = re.compile(r"(?i)(pack\s*of\s*(\d+))|(\b(\d+)[-\s]*pack\b)")
multiplier_regex = re.compile(r"(?i)(\b(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)(\s*[a-zA-Z]+)?)|(\b(\d+)\s*pcs\b)")
unit_regex = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(ml|l|g|kg|oz|lb|pcs?|ct|tabs?|caps?|sheet|roll|pair|dozen|gb|mb|tb)\b")

def mask_prices(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    t = currency_regex.sub(" <PRICE> ", text)
    t = mrp_regex.sub(" <PRICE> ", t)
    t = standalone_number_near_price.sub(" <PRICE> ", t)
    return t

UNIT_MAP = {
    'ml': ('l', 0.001), 'l': ('l', 1.0),
    'g': ('kg', 0.001), 'kg': ('kg', 1.0),
    'oz': ('lb', 1/16), 'lb': ('lb', 1.0),
    'pc': ('count', 1.0), 'pcs': ('count', 1.0), 'ct': ('count', 1.0),
    'tab': ('count', 1.0), 'tabs': ('count', 1.0), 'cap': ('count', 1.0), 'caps': ('count', 1.0),
    'sheet': ('count', 1.0), 'roll': ('count', 1.0), 'pair': ('count', 2.0), 'dozen': ('count', 12.0),
    'gb': ('gb', 1.0), 'mb': ('gb', 1/1024), 'tb': ('gb', 1024.0)
}

def parse_quantities(text: str):
    if not isinstance(text, str) or not text:
        return math.nan, math.nan, 'unknown', 0, 0
    
    t = text.lower()
    pack_count = 1
    has_multiplier = 0
    
    m_pack = pack_of_regex.search(t)
    if m_pack:
        for g in m_pack.groups():
            if g and g.isdigit():
                pack_count = max(pack_count, int(g))
                break
    
    m_mult = multiplier_regex.search(t)
    if m_mult:
        has_multiplier = 1
        try:
            left = int(m_mult.group(2)) if m_mult.group(2) else (int(m_mult.group(6)) if m_mult.group(6) else 1)
        except:
            left = 1
        pack_count = max(pack_count, left)
    
    unit_size = math.nan
    unit_type = 'unknown'
    sizes = []
    
    for m in unit_regex.finditer(t):
        val = float(m.group(1))
        unit = m.group(2).lower()
        base = unit
        
        if unit in ('pc', 'pcs'):
            base = 'pcs'
        if unit in ('tab', 'tabs'):
            base = 'tabs'
        
        mapped = UNIT_MAP.get(base, None)
        if mapped:
            canon, scale = mapped
            sizes.append((val * scale, canon))
        else:
            sizes.append((val, unit))
    
    total_content = math.nan
    if sizes:
        units = [u for _, u in sizes]
        major = max(set(units), key=units.count)
        vals = sorted([v for v, u in sizes if u == major])
        unit_type = major
        unit_size = float(np.median(vals))
        if not math.isnan(unit_size):
            total_content = unit_size * max(1, pack_count)
    
    return unit_size, total_content, unit_type, pack_count, has_multiplier

BRAND_PATTERNS = [
    'amazon', 'kirkland', 'great value', 'member', 'nature', 'organic',
    'premium', 'fresh', 'pure', 'natural', 'gourmet', 'chef'
]

def extract_brand_features(text):
    if not isinstance(text, str):
        return 0, 0, 0
    
    t = text.lower()
    has_premium = int(any(w in t for w in ['premium', 'gourmet', 'chef', 'artisan']))
    has_organic = int(any(w in t for w in ['organic', 'natural', 'pure']))
    has_brand = int(any(w in t for w in BRAND_PATTERNS))
    
    return has_premium, has_organic, has_brand

print('✓ Feature engineering utilities loaded')

#%%
print('\nLoading data and embeddings...')

#%%
print('\nLoading data and embeddings...')

train_df = pd.read_csv('subset_train_70k.csv')
val_df = pd.read_csv('subset_val_5k.csv')

train_image = np.load('train_dinov2_70k.npy')
val_image = np.load('val_dinov2_5k.npy')
train_text = np.load('train_gte_qwen2_70k.npy')
val_text = np.load('val_gte_qwen2_5k.npy')

print(f'Train size: {len(train_df):,}')
print(f'Val size: {len(val_df):,}')
print(f'Train image embeddings: {train_image.shape}')
print(f'Train text embeddings: {train_text.shape}')
print('✓ Data loaded')

#%%
print('\nApplying feature engineering...')

for df in (train_df, val_df):
    df['text_masked'] = df['catalog_content'].apply(mask_prices)
    
    parsed = df['text_masked'].apply(parse_quantities)
    df['unit_size'] = parsed.apply(lambda x: x[0])
    df['total_content'] = parsed.apply(lambda x: x[1])
    df['unit_type'] = parsed.apply(lambda x: x[2])
    df['pack_count'] = parsed.apply(lambda x: x[3])
    df['has_multiplier'] = parsed.apply(lambda x: x[4])
    
    brand_feats = df['catalog_content'].apply(extract_brand_features)
    df['has_premium'] = brand_feats.apply(lambda x: x[0])
    df['has_organic'] = brand_feats.apply(lambda x: x[1])
    df['has_brand'] = brand_feats.apply(lambda x: x[2])
    
    df['text_len'] = df['catalog_content'].astype(str).str.len()
    df['word_count'] = df['catalog_content'].astype(str).str.split().str.len()

num_cols = ['unit_size', 'total_content', 'pack_count']
for c in num_cols:
    m = float(np.nanmedian(train_df[c].values))
    train_df[c] = train_df[c].fillna(m)
    val_df[c] = val_df[c].fillna(m)

freq = train_df['unit_type'].value_counts().to_dict()
train_df['unit_type_freq'] = train_df['unit_type'].map(freq).fillna(0).astype(float)
val_df['unit_type_freq'] = val_df['unit_type'].map(freq).fillna(0).astype(float)

print('✓ Feature engineering complete')

#%%
print('\nPreparing final features (no TF-IDF)...')

numeric_cols = ['unit_size', 'total_content', 'pack_count', 'has_multiplier',
                'unit_type_freq', 'has_premium', 'has_organic', 'has_brand',
                'text_len', 'word_count']

X_num_train = train_df[numeric_cols].values
X_num_val = val_df[numeric_cols].values
X_train = np.hstack([train_text, train_image, X_num_train])
X_val = np.hstack([val_text, val_image, X_num_val])
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
y_train = np.log1p(train_df['price'].values).astype(np.float32)
y_val = np.log1p(val_df['price'].values).astype(np.float32)

print(f'\n✓ Final feature matrix:')
print(f'  Train X: {X_train.shape}')
print(f'  Val X: {X_val.shape}')


#%%
class PriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = PriceDataset(X_train, y_train)
val_dataset = PriceDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print('✓ DataLoaders created')

#%%
class PricePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            # Layer 3
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 4
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Output
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

model = PricePredictor(X_train.shape[1]).to(DEVICE)
print(f'\n✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters')

#%%
criterion = nn.L1Loss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

def smape(y_true, y_pred):
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0

#%%
print('\n' + '='*60)
print('TRAINING NEURAL NETWORK')
print('='*60)

best_val_loss = float('inf')
best_smape = float('inf')
patience_counter = 0
early_stop_patience = 7

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()
            
            all_preds.append(predictions)
            all_targets.append(y_batch)
    
    val_loss /= len(val_loader)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    val_smape = smape(np.expm1(all_targets.cpu().numpy()), np.expm1(all_preds.cpu().numpy()))
    
    scheduler.step(val_loss)
    
    print(f'Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SMAPE: {val_smape:.4f}%')
    
    if val_smape < best_smape:
        best_smape = val_smape
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

print('\n✓ Training complete!')
print(f'Best validation SMAPE: {best_smape:.4f}%')

#%%
print('\nGenerating final predictions...')

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

val_predictions = []
with torch.no_grad():
    for X_batch, _ in val_loader:
        X_batch = X_batch.to(DEVICE)
        preds = model(X_batch)
        val_predictions.append(preds.cpu().numpy())

val_predictions = np.concatenate(val_predictions)
val_predictions = np.expm1(val_predictions)
val_predictions = np.clip(val_predictions, 0.01, 3000)

output = pd.DataFrame({
    'sample_id': val_df['sample_id'].values,
    'price': val_predictions
})

output.to_csv('val_predictions_nn.csv', index=False)
print('\n✓ Predictions saved to val_predictions_nn.csv')

print(f'\nPrediction statistics:')
print(output['price'].describe())

print(f'\nSample predictions:')
print(output.head(10))

print('\n' + '='*60)
print('FINAL RESULTS')
print('='*60)
print(f'🎯 Best Validation SMAPE: {best_smape:.4f}%')
print('='*60)

#%%
print('\nSaving complete pipeline for test set inference...')
pipeline_artifacts = {
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'input_dim': X_train.shape[1],
    'numeric_cols': numeric_cols,
    'best_smape': best_smape
}

torch.save(pipeline_artifacts, 'price_prediction_pipeline.pth')
print('✓ Complete pipeline saved to: price_prediction_pipeline.pth')
print('\n✓ PIPELINE COMPLETE!')

#%%
def predict_test_set(test_csv_path='subset_test.csv', 
                     test_image_path='test_dinov2_large.npy',
                     test_text_path='test_e5_large.npy',
                     output_path='test_predictions.csv'):
    """
    Load trained model and predict on test set
    
    Args:
        test_csv_path: Path to test CSV file
        test_image_path: Path to test DINOv2 embeddings
        test_text_path: Path to test E5 embeddings
        output_path: Where to save predictions
    """
    print('\n' + '='*60)
    print('INFERENCE ON TEST SET')
    print('='*60)
    
    print('\nLoading trained pipeline...')
    artifacts = torch.load('price_prediction_pipeline.pth')
    
    model = PricePredictor(artifacts['input_dim']).to(DEVICE)
    model.load_state_dict(artifacts['model_state_dict'])
    model.eval()
    
    scaler = artifacts['scaler']
    numeric_cols = artifacts['numeric_cols']
    
    print(f'✓ Model loaded (best validation SMAPE: {artifacts["best_smape"]:.4f}%)')
    print('\nLoading test data...')
    test_df = pd.read_csv(test_csv_path)
    test_image = np.load(test_image_path)
    test_text = np.load(test_text_path)
    
    print(f'Test size: {len(test_df):,}')
    print('\nApplying feature engineering...')
    test_df['text_masked'] = test_df['catalog_content'].apply(mask_prices)
    
    parsed = test_df['text_masked'].apply(parse_quantities)
    test_df['unit_size'] = parsed.apply(lambda x: x[0])
    test_df['total_content'] = parsed.apply(lambda x: x[1])
    test_df['unit_type'] = parsed.apply(lambda x: x[2])
    test_df['pack_count'] = parsed.apply(lambda x: x[3])
    test_df['has_multiplier'] = parsed.apply(lambda x: x[4])
    
    brand_feats = test_df['catalog_content'].apply(extract_brand_features)
    test_df['has_premium'] = brand_feats.apply(lambda x: x[0])
    test_df['has_organic'] = brand_feats.apply(lambda x: x[1])
    test_df['has_brand'] = brand_feats.apply(lambda x: x[2])
    
    test_df['text_len'] = test_df['catalog_content'].astype(str).str.len()
    test_df['word_count'] = test_df['catalog_content'].astype(str).str.split().str.len()
    
    for c in ['unit_size', 'total_content', 'pack_count']:
        test_df[c] = test_df[c].fillna(0)
    
    freq = test_df['unit_type'].value_counts().to_dict()
    test_df['unit_type_freq'] = test_df['unit_type'].map(freq).fillna(0).astype(float)
    
    X_num_test = test_df[numeric_cols].values
    X_test = np.hstack([test_text, test_image, X_num_test])
    X_test = scaler.transform(X_test)
    
    print(f'Test feature matrix: {X_test.shape}')
    print('\nGenerating predictions...')
    test_dataset = torch.FloatTensor(X_test).to(DEVICE)
    
    predictions = []
    batch_size = 512
    
    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch = test_dataset[i:i+batch_size]
            preds = model(batch)
            predictions.append(preds.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    predictions = np.expm1(predictions) 
    predictions = np.clip(predictions, 0.01, 3000)
    output = pd.DataFrame({
        'sample_id': test_df['sample_id'].values,
        'price': predictions
    })
    
    output.to_csv(output_path, index=False)
    
    print(f'\n✓ Predictions saved to: {output_path}')
    print(f'\nPrediction statistics:')
    print(output['price'].describe())
    print(f'\nSample predictions:')
    print(output.head(10))
    print('\n' + '='*60)
    print('TEST SET INFERENCE COMPLETE!')
    print('='*60)
    
    return output

test_predictions = predict_test_set(
    test_csv_path='/home/pushpaka/ml_test/dataset/test.csv',
    test_image_path='test_dinov2_large.npy', 
    test_text_path='test_gte_qwen2_70k.npy',
    output_path='test_predictions15.csv'
)
