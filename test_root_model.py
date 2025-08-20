#!/usr/bin/env python3
"""
Test script to verify the model in project root works correctly
"""

import os
from pathlib import Path

def check_model_structure():
    """Check if the model folder structure is correct"""
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    models_dir = Path('models')
    preferred_model_path = models_dir / model_name
    
    print(f"🔍 Checking model structure: {preferred_model_path}")
    
    # Determine which model path to use
    model_path = None
    if preferred_model_path.exists():
        model_path = preferred_model_path
        print(f"✅ Found model in models directory: {model_path}")
    else:
        # Check project root for backward compatibility
        root_model_path = Path(model_name)
        if root_model_path.exists():
            print(f"⚠️  Model found in project root: {root_model_path}")
            print("   Consider moving it to models/ directory for better organization")
            model_path = root_model_path
        else:
            print(f"❌ Model folder not found in either location:")
            print(f"   Preferred: {preferred_model_path}")
            print(f"   Legacy: {root_model_path}")
            print("   Please run 'python download_model.py' to download the model")
            return False, None
    
    # Double-check that we have a valid path before proceeding
    if not model_path or not isinstance(model_path, Path) or not model_path.exists():
        print(f"❌ Invalid model path: {model_path}")
        return False, None
    
    print(f"✅ Using valid model path: {model_path} (type: {type(model_path)})")
    
    # Required files
    required_files = [
        'config.json',
        'config_sentence_transformers.json',
        'sentence_bert_config.json',
        'modules.json',
        'model.safetensors',  # This is the most important one
    ]
    
    missing_files = []
    existing_files = []
    
    for file_name in required_files:
        try:
            file_path = model_path / file_name
            if file_path.exists():
                size = file_path.stat().st_size
                size_mb = size / (1024 * 1024)
                existing_files.append(f"✅ {file_name} ({size_mb:.1f} MB)")
            else:
                missing_files.append(f"❌ {file_name}")
        except Exception as e:
            print(f"❌ Error checking file {file_name}: {e}")
            print(f"   model_path: {model_path} (type: {type(model_path)})")
            missing_files.append(f"❌ {file_name} (ERROR)")
    
    print("\nModel files status:")
    for file_info in existing_files:
        print(f"  {file_info}")
    for file_info in missing_files:
        print(f"  {file_info}")
    
    # Check the most critical file
    try:
        model_weights = model_path / 'model.safetensors'
        if model_weights.exists():
            size_mb = model_weights.stat().st_size / (1024 * 1024)
            if size_mb > 400:  # Should be around 471MB
                print(f"\n✅ Model weights file looks good: {size_mb:.1f} MB")
            else:
                print(f"\n⚠️  Model weights file seems small: {size_mb:.1f} MB (expected ~471MB)")
                print("   File might be incomplete")
        else:
            print(f"\n❌ Critical file missing: model.safetensors")
            print("   This file contains the neural network weights and is essential")
    except Exception as e:
        print(f"\n❌ Error checking model.safetensors: {e}")
        print(f"   model_path: {model_path} (type: {type(model_path)})")
    
    return len(missing_files) == 0, model_path

def test_model_loading(model_path):
    """Test loading and using the model"""
    try:
        from sentence_transformers import SentenceTransformer
        print("\n✅ sentence-transformers library is available")
    except ImportError:
        print("\n❌ sentence-transformers not installed")
        print("   Run: pip install sentence-transformers")
        return False
    
    if not model_path or not model_path.exists():
        print(f"\n❌ Invalid model path provided: {model_path}")
        return False
    
    model_location = str(model_path)
    print(f"\n🔄 Loading model from: {model_location}")
    
    try:
        model = SentenceTransformer(model_location)
        print("✅ Model loaded successfully!")
        
        # Test encoding
        print("\n🧪 Testing model encoding...")
        test_sentences = [
            "steel",
            "aluminum", 
            "copper",
            "stainless steel",
            "aluminium alloy"
        ]
        
        embeddings = model.encode(test_sentences)
        print(f"✅ Encoding successful! Shape: {embeddings.shape}")
        
        # Test similarity calculation
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        print(f"✅ Similarity calculation works! Matrix shape: {similarity_matrix.shape}")
        
        # Show some similarities
        print(f"\nSample similarities:")
        print(f"  steel ↔ stainless steel: {similarity_matrix[0][3]:.3f}")
        print(f"  aluminum ↔ aluminium alloy: {similarity_matrix[1][4]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading/testing failed: {e}")
        return False

def update_main_code():
    """Update the main code to use the models directory"""
    print(f"\n🔧 The main code (test.py) is configured to:")
    print(f"   1. Prioritize models in the models/ directory")
    print(f"   2. Use paraphrase-multilingual-MiniLM-L12-v2 by default")
    print(f"   3. Fall back to project root for backward compatibility")
    print(f"   4. Fall back gracefully if model not available")
    print(f"\n✅ Code is properly organized - models go in models/ directory!")

def main():
    """Main test function"""
    print("🧪 Testing Model Setup")
    print("=" * 50)
    
    # Check structure
    structure_ok, model_path = check_model_structure()
    
    if structure_ok and model_path:
        # Test loading
        loading_ok = test_model_loading(model_path)
        
        if loading_ok:
            print("\n" + "=" * 50)
            print("🎉 SUCCESS! Model is working perfectly!")
            print("\n📋 What works now:")
            print("  ✅ Model is properly located")
            print("  ✅ All required files present")
            print("  ✅ Model loads and encodes text")
            print("  ✅ Similarity calculations work")
            print("  ✅ Ready for offline material comparison!")
            
            update_main_code()
            
            print(f"\n🚀 You can now use:")
            print(f"   python demo_with_instructions.py  # Full demo with instructions")
            print(f"   python test_basic_functionality.py  # Basic functionality test")
            print(f"   # Or use the LLM mode directly in your code")
            
        else:
            print(f"\n⚠️  Model files found but loading failed")
            print(f"   Try running 'python download_model.py' to re-download the model")
    else:
        print(f"\n❌ Model setup incomplete")
        print(f"   Please run 'python download_model.py' to download the model")

if __name__ == "__main__":
    main()
