#!/usr/bin/env python3
"""
Script to download and save the paraphrase-multilingual-MiniLM-L12-v2 model locally
for offline use in the Material Comparison Tool.
"""

import os
import sys
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers is available")
except ImportError:
    print("❌ sentence-transformers not found. Please install it first:")
    print("pip install sentence-transformers")
    sys.exit(1)

def download_model():
    """Download and save models locally with fallback options"""
    
    # List of models to try (from largest to smallest, with fallback options)
    models_to_try = [
        {
            'name': 'paraphrase-multilingual-MiniLM-L12-v2',
            'size': '~471MB',
            'description': 'Multilingual model (preferred)'
        },
        {
            'name': 'all-MiniLM-L6-v2', 
            'size': '~23MB',
            'description': 'Lightweight English model (fallback)'
        }
    ]
    
    # Create models directory in the project
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    for model_info in models_to_try:
        model_name = model_info['name']
        local_model_path = models_dir / model_name.replace('/', '_')
        
        print(f"\n🔄 Attempting to download: {model_name}")
        print(f"📝 Description: {model_info['description']}")
        print(f"📁 Saving to: {local_model_path}")
        print(f"💾 Size: {model_info['size']}")
        print("⏳ This may take a few minutes depending on your internet connection...")
        
        try:
            # Download the model
            model = SentenceTransformer(model_name)
            
            # Save it locally
            model.save(str(local_model_path))
            
            print(f"✅ Model '{model_name}' successfully downloaded and saved!")
            print(f"📍 Model location: {local_model_path.absolute()}")
            
            # Test loading from local path
            print("\n🧪 Testing local model loading...")
            test_model = SentenceTransformer(str(local_model_path))
            
            # Quick test
            test_sentences = ["steel", "aluminum", "copper"]
            embeddings = test_model.encode(test_sentences)
            print(f"✅ Local model test successful! Generated embeddings with shape: {embeddings.shape}")
            
            return str(local_model_path), model_name
            
        except Exception as e:
            print(f"❌ Error downloading model '{model_name}': {e}")
            print(f"🔄 Trying next model...")
            continue
    
    print("❌ All model downloads failed!")
    return None, None

if __name__ == "__main__":
    print("🤖 Material Comparison Tool - Model Downloader")
    print("=" * 60)
    
    model_path, model_name = download_model()
    
    if model_path:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Model is ready for offline use.")
        print(f"📦 Downloaded model: {model_name}")
        print("\nNext steps:")
        print("1. The model is now saved locally")
        print("2. The main tool will automatically use the local model")
        print("3. You can now run material comparisons without internet!")
        print("\n💡 The local model path is:")
        print(f"   {model_path}")
        
        # Create a config file to remember which model was downloaded
        config_path = Path('models/local_model_config.txt')
        with open(config_path, 'w') as f:
            f.write(f"model_name={model_name}\n")
            f.write(f"model_path={model_path}\n")
        print(f"\n📄 Model configuration saved to: {config_path}")
        print(f"✅ Model is properly organized in the models/ directory")
    else:
        print("\n❌ All model downloads failed. Please check your internet connection and try again.")
