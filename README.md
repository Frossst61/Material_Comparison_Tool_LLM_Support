# CSV Material Comparison Tool with LLM Support

A powerful Python tool for comparing and matching materials from two CSV files using traditional string similarity and AI-powered semantic matching.

## Features

- **Multiple Matching Modes:**
  - `difflib`: Traditional string similarity (fast, no dependencies)
  - `llm`: AI-powered semantic matching using sentence transformers
  - `hybrid`: Combines both methods for optimal results

- **Flexible CSV Support:**
  - Automatic delimiter detection (comma/semicolon)
  - Configurable column mapping
  - Header detection
  - Robust error handling

- **Advanced Matching:**
  - Semantic understanding of material names
  - Multilingual model support
  - Configurable similarity thresholds
  - Detailed matching reports

## Installation

### Basic Installation (difflib mode only)
```bash
# Clone or download the script
# No additional dependencies required for basic string matching
```

### Full Installation (with LLM support)
```bash
# Install required packages
pip install -r requirements.txt

# Or install individually:
pip install sentence-transformers scikit-learn numpy torch
```

### Model Setup for Offline Use
To use the `paraphrase-multilingual-MiniLM-L12-v2` model offline:

1. **Create model folder**: `mkdir paraphrase-multilingual-MiniLM-L12-v2`
2. **Download model files** from HuggingFace: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
3. **Place files in project root**: Especially `model.safetensors` (471MB)
4. **Test setup**: `python test_root_model.py`

The tool will automatically detect and use the local model.

## Quick Start

### Basic Usage (String Matching)
```python
from test import compare_csv_materials, print_comparison_summary

# Compare two CSV files using traditional string matching
summary = compare_csv_materials(
    'materials_file1.csv', 
    'materials_file2.csv', 
    'output.csv'
)
print_comparison_summary(summary)
```

### LLM-Powered Matching
```python
# Use AI-powered semantic matching with multilingual model
summary = compare_csv_materials(
    'materials_file1.csv', 
    'materials_file2.csv', 
    'llm_output.csv',
    matching_mode='llm',
    llm_model_name='paraphrase-multilingual-MiniLM-L12-v2',
    similarity_threshold=0.7
)
print_comparison_summary(summary)
```

**Note:** Place the model folder `paraphrase-multilingual-MiniLM-L12-v2` in the project root for offline use.

### Hybrid Matching (Recommended)
```python
# Combine string and semantic matching for best results
summary = compare_csv_materials(
    'materials_file1.csv', 
    'materials_file2.csv', 
    'hybrid_output.csv',
    matching_mode='hybrid',
    similarity_threshold=0.65
)
print_comparison_summary(summary)
```

## Available Models

| Model | Description | Size | Speed | Best For |
|-------|-------------|------|-------|----------|
| `all-MiniLM-L6-v2` | Fast and lightweight | ~23MB | Very Fast | General text similarity |
| `all-mpnet-base-v2` | Higher quality embeddings | ~438MB | Medium | Better semantic understanding |
| `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual support | ~471MB | Medium | International material names |
| `sentence-transformers/all-distilroberta-v1` | Balanced performance | ~326MB | Fast | Balanced speed/quality |

## Parameters

### `compare_csv_materials()` Parameters

- `file1_path` (str): Path to first CSV file (source)
- `file2_path` (str): Path to second CSV file (target for matching)
- `output_path` (str): Path for output CSV file
- `file1_id_col` (int): ID column index in first file (default: 0)
- `file1_name_col` (int): Name column index in first file (default: 1)
- `file2_id_col` (int): ID column index in second file (default: 0)
- `file2_name_col` (int): Name column index in second file (default: 1)
- `has_header` (bool): Whether CSV files have headers (default: True)
- `similarity_threshold` (float): Minimum similarity score (0.0-1.0, default: 0.0)
- `matching_mode` (str): Matching algorithm ('difflib', 'llm', 'hybrid', default: 'difflib')
- `llm_model_name` (str): Sentence transformer model name (default: 'all-MiniLM-L6-v2')

## Output Format

The tool generates a CSV file with these columns:

- `Material_ID_File1`: ID from source file
- `Material_Name_File1`: Name from source file
- `Material_ID_File2`: Best matching ID from target file (or 'NO_MATCH')
- `Material_Name_File2`: Best matching name from target file (or 'NO_MATCH')
- `Similarity_Coefficient`: Similarity score (0.0-1.0)
- `Matching_Method`: Method used ('difflib', 'llm', or 'hybrid')

## Examples

### Example 1: Basic Material Matching
```python
# Match materials using string similarity
summary = compare_csv_materials(
    'inventory.csv',
    'catalog.csv', 
    'matches.csv',
    similarity_threshold=0.6
)
```

### Example 2: Semantic Matching for Technical Materials
```python
# Use LLM for better understanding of technical terms
summary = compare_csv_materials(
    'chemical_inventory.csv',
    'supplier_catalog.csv',
    'chemical_matches.csv',
    matching_mode='llm',
    llm_model_name='all-mpnet-base-v2',
    similarity_threshold=0.75
)
```

### Example 3: Multilingual Material Matching
```python
# Handle international material names
summary = compare_csv_materials(
    'materials_en.csv',
    'materials_multi.csv',
    'multilingual_matches.csv',
    matching_mode='hybrid',
    llm_model_name='paraphrase-multilingual-MiniLM-L12-v2',
    similarity_threshold=0.7
)
```

## Performance Tips

1. **For large datasets (>10,000 materials):** Use `hybrid` mode with appropriate thresholds
2. **For speed:** Use `difflib` mode or the lightweight `all-MiniLM-L6-v2` model
3. **For accuracy:** Use `all-mpnet-base-v2` model with `hybrid` mode
4. **For multilingual data:** Use the multilingual model variants

## Troubleshooting

### LLM Support Not Available
```
Error: LLM support not available. Install sentence-transformers and scikit-learn
```
**Solution:** Install required packages:
```bash
pip install sentence-transformers scikit-learn
```

### Model Download Issues
- Models are downloaded automatically on first use
- Ensure stable internet connection
- Models are cached locally after first download

### Memory Issues with Large Models
- Use smaller models like `all-MiniLM-L6-v2`
- Process files in smaller batches
- Increase system RAM or use cloud computing

## License

This tool is provided as-is for material comparison tasks. Modify and distribute as needed.
