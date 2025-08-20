#!/usr/bin/env python3
"""
Manual setup script for offline model usage.
This script helps users set up local models for the Material Comparison Tool.
"""

import os
import sys
from pathlib import Path

def create_models_directory():
    """Create the models directory structure"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    print(f"✅ Created models directory: {models_dir.absolute()}")
    return models_dir

def create_manual_instructions():
    """Create instructions for manual model setup"""
    instructions = """
# Manual Model Setup Instructions

Due to network connectivity issues, you can manually set up models for offline use:

## Option 1: Download on another machine with internet
1. On a machine with internet access, run:
   ```
   pip install sentence-transformers
   python -c "
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
   model.save('./paraphrase-multilingual-MiniLM-L12-v2_local')
   print('Model saved to paraphrase-multilingual-MiniLM-L12-v2_local')
   "
   ```

2. Copy the entire `paraphrase-multilingual-MiniLM-L12-v2_local` folder to this project's `models/` directory

3. Create a config file:
   ```
   echo "model_name=paraphrase-multilingual-MiniLM-L12-v2" > models/local_model_config.txt
   echo "model_path=models/paraphrase-multilingual-MiniLM-L12-v2_local" >> models/local_model_config.txt
   ```

## Option 2: Use a smaller fallback model
1. Try downloading the smaller model:
   ```
   python -c "
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   model.save('./all-MiniLM-L6-v2_local')
   print('Model saved to all-MiniLM-L6-v2_local')
   "
   ```

2. Copy to models directory and create config as above

## Option 3: Use HuggingFace Hub cache
If you've previously downloaded models, they might be in your HuggingFace cache:
- Windows: `C:\\Users\\{username}\\.cache\\huggingface\\hub\\`
- Look for folders starting with `models--sentence-transformers--`
- Copy the model folder to this project's `models/` directory

## Testing
Run this script to test if your local model works:
```
python setup_offline_model.py --test
```
"""
    
    with open('OFFLINE_SETUP.md', 'w') as f:
        f.write(instructions)
    
    print("📄 Created OFFLINE_SETUP.md with manual setup instructions")

def test_local_model():
    """Test if a local model is available and working"""
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers is available")
    except ImportError:
        print("❌ sentence-transformers not found. Please install it first:")
        print("pip install sentence-transformers scikit-learn")
        return False
    
    models_dir = Path('models')
    if not models_dir.exists():
        print("❌ No models directory found")
        return False
    
    # Check for config file
    config_path = models_dir / 'local_model_config.txt'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config_lines = f.read().strip().split('\n')
                model_path = None
                for line in config_lines:
                    if line.startswith('model_path='):
                        model_path = line.split('=', 1)[1]
                        break
                
                if model_path and os.path.exists(model_path):
                    print(f"🔄 Testing local model: {model_path}")
                    model = SentenceTransformer(model_path)
                    
                    # Quick test
                    test_sentences = ["steel", "aluminum", "copper"]
                    embeddings = model.encode(test_sentences)
                    print(f"✅ Local model test successful! Generated embeddings with shape: {embeddings.shape}")
                    return True
                else:
                    print(f"❌ Model path in config not found: {model_path}")
        except Exception as e:
            print(f"❌ Error testing local model: {e}")
    
    # Check for common model directories
    potential_models = [
        'paraphrase-multilingual-MiniLM-L12-v2_local',
        'paraphrase-multilingual-MiniLM-L12-v2',
        'all-MiniLM-L6-v2_local',
        'all-MiniLM-L6-v2'
    ]
    
    for model_name in potential_models:
        model_path = models_dir / model_name
        if model_path.exists():
            try:
                print(f"🔄 Testing found model: {model_path}")
                model = SentenceTransformer(str(model_path))
                
                # Quick test
                test_sentences = ["steel", "aluminum", "copper"]
                embeddings = model.encode(test_sentences)
                print(f"✅ Model test successful! Generated embeddings with shape: {embeddings.shape}")
                
                # Create config file
                with open(config_path, 'w') as f:
                    f.write(f"model_name={model_name}\n")
                    f.write(f"model_path={model_path}\n")
                print(f"📄 Created config file: {config_path}")
                return True
            except Exception as e:
                print(f"⚠️  Model found but failed to load: {e}")
    
    print("❌ No working local models found")
    return False

def main():
    """Main function"""
    print("🔧 Material Comparison Tool - Offline Model Setup")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        success = test_local_model()
        if success:
            print("\n🎉 SUCCESS! Local model is ready for offline use.")
        else:
            print("\n❌ No local model found. Please follow the manual setup instructions.")
        return
    
    # Create directory structure
    models_dir = create_models_directory()
    
    # Create instructions
    create_manual_instructions()
    
    print("\n" + "=" * 60)
    print("📋 Next Steps:")
    print("1. Read OFFLINE_SETUP.md for detailed instructions")
    print("2. Set up a local model using one of the provided methods")
    print("3. Run 'python setup_offline_model.py --test' to verify setup")
    print("4. Use the main tool with LLM matching modes")
    
    print(f"\n📁 Models should be placed in: {models_dir.absolute()}")

if __name__ == "__main__":
    main()
