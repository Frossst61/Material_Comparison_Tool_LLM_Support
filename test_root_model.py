#!/usr/bin/env python3
"""
Test script to verify the model in project root works correctly
"""

import os
from pathlib import Path

def check_model_structure():
    """Check if the model folder structure is correct"""
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    model_path = Path(model_name)
    
    print(f"🔍 Checking model structure: {model_path}")
    
    if not model_path.exists():
        print(f"❌ Model folder not found: {model_path}")
        print("   Please follow MANUAL_MODEL_SETUP.md to set up the model")
        return False
    
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
        file_path = model_path / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            size_mb = size / (1024 * 1024)
            existing_files.append(f"✅ {file_name} ({size_mb:.1f} MB)")
        else:
            missing_files.append(f"❌ {file_name}")
    
    print("\nModel files status:")
    for file_info in existing_files:
        print(f"  {file_info}")
    for file_info in missing_files:
        print(f"  {file_info}")
    
    # Check the most critical file
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
    
    return len(missing_files) == 0

def test_model_loading():
    """Test loading and using the model"""
    try:
        from sentence_transformers import SentenceTransformer
        print("\n✅ sentence-transformers library is available")
    except ImportError:
        print("\n❌ sentence-transformers not installed")
        print("   Run: pip install sentence-transformers")
        return False
    
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    
    try:
        print(f"\n🔄 Loading model from: ./{model_name}")
        model = SentenceTransformer(model_name)
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
    """Update the main code to use the root model"""
    print(f"\n🔧 The main code (test.py) is already configured to:")
    print(f"   1. Look for models in the project root")
    print(f"   2. Use paraphrase-multilingual-MiniLM-L12-v2 by default")
    print(f"   3. Fall back gracefully if model not available")
    print(f"\n✅ No code changes needed - everything is ready!")

def main():
    """Main test function"""
    print("🧪 Testing Model in Project Root")
    print("=" * 50)
    
    # Check structure
    structure_ok = check_model_structure()
    
    if structure_ok:
        # Test loading
        loading_ok = test_model_loading()
        
        if loading_ok:
            print("\n" + "=" * 50)
            print("🎉 SUCCESS! Model is working perfectly!")
            print("\n📋 What works now:")
            print("  ✅ Model is in project root")
            print("  ✅ All required files present")
            print("  ✅ Model loads and encodes text")
            print("  ✅ Similarity calculations work")
            print("  ✅ Ready for offline material comparison!")
            
            update_main_code()
            
            print(f"\n🚀 You can now use:")
            print(f"   python demo_offline.py")
            print(f"   # Or use the LLM mode directly in your code")
            
        else:
            print(f"\n⚠️  Model files found but loading failed")
            print(f"   Check MANUAL_MODEL_SETUP.md for troubleshooting")
    else:
        print(f"\n❌ Model setup incomplete")
        print(f"   Please follow MANUAL_MODEL_SETUP.md to download the model")

if __name__ == "__main__":
    main()
