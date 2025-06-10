# CLIP Image-Text Retrieval 

A PyTorch implementation of Contrastive Language-Image Pre-training (CLIP) for multimodal learning, enabling semantic image search through natural language queries.

##  Features

- **Multi-Architecture Support**: Three different configurations with varying model complexities
- **Contrastive Learning**: Learns joint representations of images and text in a shared embedding space
- **Image Search**: Query images using natural language descriptions
- **Comprehensive Evaluation**: Visual results and loss tracking across different model configurations

## Model Configurations

| Configuration | Image Encoder | Text Encoder | Image Embedding | Text Embedding |
|---------------|---------------|--------------|-----------------|----------------|
| **Config 1** | ResNet-18 | DistilBERT | 512 | 768 |
| **Config 2** | ResNet-34 | BERT | 512 | 768 |
| **Config 3** | ResNet-50 | RoBERTa | 2048 | 768 |

##  Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/clip-image-text-retrieval.git
cd clip-image-text-retrieval

# Install required packages
pip install torch torchvision transformers timm opencv-python pandas albumentations matplotlib tqdm
```

## Dataset

This implementation uses the **Flickr8k** dataset:
- **Images**: 8,000 images from Flickr
- **Captions**: 5 captions per image (40,000 total)
- **Split**: 80% training, 20% validation

### Download Dataset
```bash
# Using Kaggle API
kaggle datasets download -d adityajn105/flickr8k
unzip flickr8k.zip
```

##  Architecture

### Image Encoder
- **ResNet backbone** (18/34/50) with pretrained weights
- **Global average pooling** for feature extraction
- **Projection head** maps to shared embedding space

### Text Encoder
- **Transformer models** (DistilBERT/BERT/RoBERTa)
- **CLS token** representation for sentence embedding
- **Projection head** maps to shared embedding space

### Loss Function
```python
# Contrastive loss with temperature scaling
logits = (text_embeddings @ image_embeddings.T) / temperature
targets = softmax((image_similarity + text_similarity) / 2 * temperature)
loss = (cross_entropy(logits, targets) + cross_entropy(logits.T, targets.T)) / 2
```

## 🔧 Usage

### Training
```python
# Configure hyperparameters
class Config:
    batch_size = 32
    epochs = 4
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    temperature = 1.0
    projection_dim = 256

# Train model
model = CLIPModel().to(device)
# ... training loop
```

### Image Search
```python
# Load trained model
model.load_state_dict(torch.load('model_weights.pt'))

# Search for images
find_matches(
    model, 
    image_embeddings, 
    query="A dog playing in a grassy field",
    image_filenames=valid_df['image'].values,
    n=9
)
```

##  Results

### Training Performance

| Model | Final Train Loss | Final Valid Loss | Training Stability |
|-------|------------------|------------------|-------------------|
| **ResNet-18 + DistilBERT** | 0.377 | 2.36 | Moderate overfitting |
| **ResNet-34 + BERT** | 0.335 | 2.27 | Most stable |
| **ResNet-50 + RoBERTa** | 0.23 | 2.27 | Best training, similar validation |

### Key Findings
- **ResNet-50 + RoBERTa** achieves the lowest training loss but shows diminishing returns on validation
- **ResNet-34 + BERT** provides the best balance between performance and stability
- All models demonstrate effective image-text alignment for semantic search


##  Technical Details

### Hyperparameters
- **Batch Size**: 32
- **Learning Rates**: Head (1e-3), Image Encoder (1e-4), Text Encoder (1e-5)
- **Weight Decay**: 1e-3
- **Temperature**: 1.0
- **Max Text Length**: 200 tokens
- **Image Size**: 224×224

### Data Preprocessing
- **Images**: Resize to 224×224, normalize to [-1, 1]
- **Text**: Tokenization with padding and truncation
- **Augmentation**: Resize and normalization using Albumentations
