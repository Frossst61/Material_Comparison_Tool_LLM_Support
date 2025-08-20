# -*- coding: utf-8 -*-
"""
Demo showing current setup and instructions for completing the model setup
"""

import os
from pathlib import Path
from test import compare_csv_materials, print_comparison_summary, HAS_LLM_SUPPORT

def check_model_status():
    """Check current model status"""
    print("🔍 Checking Model Setup Status")
    print("=" * 50)
    
    # Check for model in project root
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    root_model_path = Path(model_name)
    
    if root_model_path.exists():
        print(f"✅ Found model folder in project root: {root_model_path}")
        
        # Check for key files
        key_files = ['model.safetensors', 'config.json', 'modules.json']
        missing_files = []
        
        for file_name in key_files:
            if (root_model_path / file_name).exists():
                size = (root_model_path / file_name).stat().st_size / (1024*1024)
                print(f"  ✅ {file_name} ({size:.1f} MB)")
            else:
                missing_files.append(file_name)
                print(f"  ❌ {file_name} - MISSING")
        
        if missing_files:
            print(f"\n⚠️  Model folder exists but {len(missing_files)} files are missing")
            print("   Most critical: model.safetensors (471MB)")
            return False
        else:
            print(f"\n✅ Model appears complete!")
            return True
    else:
        print(f"❌ Model folder not found: {root_model_path}")
        print("   Expected location: ./paraphrase-multilingual-MiniLM-L12-v2/")
        return False

def show_setup_instructions():
    """Show setup instructions"""
    print("\n" + "=" * 60)
    print("📋 SETUP INSTRUCTIONS")
    print("=" * 60)
    
    print("\nTo complete the setup, you need to:")
    print("\n1️⃣  Create the model folder:")
    print("   mkdir paraphrase-multilingual-MiniLM-L12-v2")
    
    print("\n2️⃣  Download these files into that folder:")
    print("   • config.json")
    print("   • config_sentence_transformers.json") 
    print("   • sentence_bert_config.json")
    print("   • modules.json")
    print("   • model.safetensors (471MB) ← MOST IMPORTANT")
    print("   • tokenizer files")
    
    print("\n3️⃣  Get files from:")
    print("   • HuggingFace: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("   • Direct links in MANUAL_MODEL_SETUP.md")
    print("   • Git clone (if you have git-lfs)")
    
    print("\n4️⃣  Verify with:")
    print("   python test_root_model.py")

def demo_current_capabilities():
    """Demo what works right now"""
    print("\n" + "=" * 60)
    print("🚀 CURRENT CAPABILITIES (Working Now)")
    print("=" * 60)
    
    print("Even without the LLM model, the tool works great with string matching:")
    
    # Create sample data
    import csv
    
    materials1 = [
        ("M001", "Steel Grade 304"),
        ("M002", "Aluminum Alloy 6061"),
        ("M003", "Copper C101"),
    ]
    
    materials2 = [
        ("MAT_A", "Stainless Steel 304"),
        ("MAT_B", "Aluminium 6061-T6"),
        ("MAT_C", "Pure Copper"),
    ]
    
    # Write sample files
    with open('demo_materials1.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['ID', 'Name'])
        writer.writerows(materials1)
    
    with open('demo_materials2.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['ID', 'Name'])
        writer.writerows(materials2)
    
    print("\n🔄 Running string-based comparison...")
    
    # Run comparison
    summary = compare_csv_materials(
        'demo_materials1.csv',
        'demo_materials2.csv', 
        'demo_results.csv',
        matching_mode='difflib',
        similarity_threshold=0.4
    )
    
    print_comparison_summary(summary)
    
    print(f"\n✅ Results saved to: demo_results.csv")
    print(f"📊 This mode works completely offline!")

def show_future_capabilities():
    """Show what will work once model is set up"""
    print("\n" + "=" * 60)
    print("🎯 FUTURE CAPABILITIES (After Model Setup)")
    print("=" * 60)
    
    print("Once you have the model in place, you'll be able to use:")
    
    print("\n1️⃣  LLM Semantic Matching:")
    print("   summary = compare_csv_materials(")
    print("       'file1.csv', 'file2.csv', 'output.csv',")
    print("       matching_mode='llm',")
    print("       similarity_threshold=0.7")
    print("   )")
    
    print("\n2️⃣  Hybrid Matching (Best Results):")
    print("   summary = compare_csv_materials(")
    print("       'file1.csv', 'file2.csv', 'output.csv',")
    print("       matching_mode='hybrid',")
    print("       similarity_threshold=0.65")
    print("   )")
    
    print("\n🌟 Benefits of LLM matching:")
    print("   • Understands material semantics")
    print("   • Matches 'Steel 304' with 'Stainless Steel Grade 304'")
    print("   • Handles different languages and naming conventions")
    print("   • Much better accuracy for technical materials")

def main():
    """Main demo"""
    print("🔧 Material Comparison Tool - Setup Status & Demo")
    print("=" * 60)
    
    print(f"📦 LLM Support Available: {HAS_LLM_SUPPORT}")
    
    # Check current status
    model_ready = check_model_status()
    
    if not model_ready:
        show_setup_instructions()
    
    # Demo current capabilities
    demo_current_capabilities()
    
    # Show future capabilities
    show_future_capabilities()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    if model_ready:
        print("🎉 Model is ready! You can use all matching modes.")
    else:
        print("⏳ Model setup needed for LLM features.")
        print("✅ String matching works now (see demo_results.csv)")
        print("🔧 Follow instructions above to enable LLM matching")
    
    print(f"\n📄 Files created:")
    print(f"   • demo_results.csv - Sample comparison results")
    print(f"   • MANUAL_MODEL_SETUP.md - Detailed setup guide")
    print(f"   • test_root_model.py - Model verification script")

if __name__ == "__main__":
    main()
