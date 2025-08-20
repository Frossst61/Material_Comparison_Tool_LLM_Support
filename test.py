import difflib
import csv
import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings

# Optional imports for LLM functionality
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_LLM_SUPPORT = True
except ImportError:
    HAS_LLM_SUPPORT = False
    warnings.warn("LLM support not available. Install sentence-transformers and scikit-learn for LLM matching: pip install sentence-transformers scikit-learn")

class LLMMaterialMatcher:
    """LLM-based material matcher using sentence transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', local_models_dir: str = 'models'):
        """
        Initialize the LLM matcher
        
        Args:
            model_name (str): Name of the sentence transformer model to use
            local_models_dir (str): Directory where local models are stored
        """
        if not HAS_LLM_SUPPORT:
            raise ImportError("LLM support not available. Please install required packages.")
        
        self.model_name = model_name
        self.local_models_dir = local_models_dir
        self.model = None
        self.is_offline = False
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model (models directory first, then online)"""
        
        # Priority 1: Check if we have a local model configuration in models directory
        local_config_path = os.path.join(self.local_models_dir, 'local_model_config.txt')
        local_model_path = None
        
        if os.path.exists(local_config_path):
            try:
                with open(local_config_path, 'r') as f:
                    config_lines = f.read().strip().split('\n')
                    for line in config_lines:
                        if line.startswith('model_path='):
                            local_model_path = line.split('=', 1)[1]
                            break
                print(f"Found local model configuration: {local_model_path}")
            except Exception as e:
                print(f"Warning: Could not read local model config: {e}")
        
        # Priority 2: Try to find local model by name in models directory
        if not local_model_path:
            potential_local_path = os.path.join(self.local_models_dir, self.model_name.replace('/', '_'))
            if os.path.exists(potential_local_path):
                local_model_path = potential_local_path
                print(f"Found local model directory: {local_model_path}")
        
        # Try loading from models directory
        if local_model_path and os.path.exists(local_model_path):
            try:
                print(f"🔄 Loading local model from: {local_model_path}")
                self.model = SentenceTransformer(local_model_path)
                self.is_offline = True
                print("✅ Local model loaded successfully! (Offline mode)")
                return
            except Exception as e:
                print(f"⚠️  Failed to load local model: {e}")
                print("🔄 Falling back to online download...")
        
        # Priority 3: Check project root for backward compatibility (deprecated)
        root_model_path = self.model_name.replace('/', '_')
        if os.path.exists(root_model_path):
            try:
                print(f"⚠️  Found model in project root: {root_model_path}")
                print(f"   Consider moving it to models/ directory for better organization")
                print(f"🔄 Loading model from project root: {root_model_path}")
                self.model = SentenceTransformer(root_model_path)
                self.is_offline = True
                print("✅ Project root model loaded successfully! (Offline mode)")
                return
            except Exception as e:
                print(f"⚠️  Failed to load model from project root: {e}")
        
        # Priority 4: Fall back to online download
        try:
            print(f"🔄 Downloading model online: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.is_offline = False
            print("✅ Online model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model (all methods failed): {e}")
            print("\n💡 To use offline models:")
            print("   1. Run 'python download_model.py' when you have internet, OR")
            print("   2. Follow setup instructions to manually place models in models/ directory")
            raise
    
    def encode_materials(self, materials: List[str]) -> np.ndarray:
        """
        Encode material names into embeddings
        
        Args:
            materials (List[str]): List of material names
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Clean and prepare material names
        cleaned_materials = [str(material).strip() for material in materials]
        
        # Generate embeddings
        embeddings = self.model.encode(cleaned_materials, convert_to_tensor=False)
        return np.array(embeddings)
    
    def find_best_llm_match(self, target_name: str, candidate_materials: List[Tuple], 
                           similarity_threshold: float = 0.0) -> Tuple:
        """
        Find best match using LLM embeddings
        
        Args:
            target_name (str): Name to match against
            candidate_materials (List[Tuple]): List of (id, name) tuples
            similarity_threshold (float): Minimum similarity score
            
        Returns:
            Tuple: (best_id, best_name, similarity_score) or (None, None, 0.0)
        """
        if not candidate_materials:
            return (None, None, 0.0)
        
        # Extract candidate names
        candidate_names = [name for _, name in candidate_materials]
        
        # Encode target and candidates
        target_embedding = self.encode_materials([target_name])
        candidate_embeddings = self.encode_materials(candidate_names)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(target_embedding, candidate_embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= similarity_threshold:
            best_id, best_name = candidate_materials[best_idx]
            return (best_id, best_name, float(best_similarity))
        
        return (None, None, 0.0)

def find_best_match(target_name, candidate_materials, similarity_threshold=0.0, 
                   matching_mode='difflib', llm_matcher=None):
    """
    Find the best matching material name from a list of candidates
    
    Args:
        target_name (str): Name to match against
        candidate_materials (list): List of (id, name) tuples to search in
        similarity_threshold (float): Minimum similarity score (0.0 to 1.0)
        matching_mode (str): Matching algorithm ('difflib', 'llm', 'hybrid')
        llm_matcher (LLMMaterialMatcher): LLM matcher instance (required for 'llm' and 'hybrid' modes)
    
    Returns:
        tuple: (best_id, best_name, similarity_score) or (None, None, 0.0) if no match
    """
    if matching_mode == 'difflib':
        return _find_best_match_difflib(target_name, candidate_materials, similarity_threshold)
    elif matching_mode == 'llm':
        if not llm_matcher:
            raise ValueError("LLM matcher required for 'llm' matching mode")
        return llm_matcher.find_best_llm_match(target_name, candidate_materials, similarity_threshold)
    elif matching_mode == 'hybrid':
        if not llm_matcher:
            raise ValueError("LLM matcher required for 'hybrid' matching mode")
        return _find_best_match_hybrid(target_name, candidate_materials, similarity_threshold, llm_matcher)
    else:
        raise ValueError(f"Unknown matching mode: {matching_mode}")

def _find_best_match_difflib(target_name, candidate_materials, similarity_threshold=0.0):
    """Original difflib-based matching"""
    best_match = (None, None, 0.0)
    best_similarity = 0.0
    
    for candidate_id, candidate_name in candidate_materials:
        # Calculate similarity using difflib
        similarity = difflib.SequenceMatcher(None, target_name.lower(), candidate_name.lower()).ratio()
        
        if similarity > best_similarity and similarity >= similarity_threshold:
            best_similarity = similarity
            best_match = (candidate_id, candidate_name, similarity)
    
    return best_match

def _find_best_match_hybrid(target_name, candidate_materials, similarity_threshold, llm_matcher, 
                           difflib_weight=0.3, llm_weight=0.7):
    """
    Hybrid matching combining difflib and LLM scores
    
    Args:
        target_name (str): Name to match against
        candidate_materials (list): List of (id, name) tuples
        similarity_threshold (float): Minimum similarity score
        llm_matcher (LLMMaterialMatcher): LLM matcher instance
        difflib_weight (float): Weight for difflib score (default: 0.3)
        llm_weight (float): Weight for LLM score (default: 0.7)
    
    Returns:
        tuple: (best_id, best_name, combined_score) or (None, None, 0.0)
    """
    if not candidate_materials:
        return (None, None, 0.0)
    
    # Get LLM similarities for all candidates at once (more efficient)
    candidate_names = [name for _, name in candidate_materials]
    target_embedding = llm_matcher.encode_materials([target_name])
    candidate_embeddings = llm_matcher.encode_materials(candidate_names)
    llm_similarities = cosine_similarity(target_embedding, candidate_embeddings)[0]
    
    best_match = (None, None, 0.0)
    best_combined_score = 0.0
    
    for i, (candidate_id, candidate_name) in enumerate(candidate_materials):
        # Calculate difflib similarity
        difflib_similarity = difflib.SequenceMatcher(None, target_name.lower(), candidate_name.lower()).ratio()
        
        # Get LLM similarity
        llm_similarity = llm_similarities[i]
        
        # Combine scores
        combined_score = (difflib_weight * difflib_similarity) + (llm_weight * llm_similarity)
        
        if combined_score > best_combined_score and combined_score >= similarity_threshold:
            best_combined_score = combined_score
            best_match = (candidate_id, candidate_name, combined_score)
    
    return best_match

def read_csv_materials(file_path, id_column=0, name_column=1, has_header=True):
    """
    Read CSV file and extract material ID and name columns
    
    Args:
        file_path (str): Path to CSV file
        id_column (int): Column index for material ID (0-based)
        name_column (int): Column index for material name (0-based)
        has_header (bool): Whether the CSV has a header row
    
    Returns:
        list: List of tuples (material_id, material_name)
    """
    materials = []
    total_lines = 0
    processed_lines = 0
    skipped_lines = 0
    
    try:
        # First, let's try to detect the delimiter
        with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
            sample = csvfile.read(1024)
            csvfile.seek(0)
            
            # Check which delimiter is more common
            semicolon_count = sample.count(';')
            comma_count = sample.count(',')
            
            if semicolon_count > comma_count:
                delimiter = ';'
                print(f"Detected semicolon delimiter (found {semicolon_count} semicolons vs {comma_count} commas)")
            else:
                delimiter = ','
                print(f"Detected comma delimiter (found {comma_count} commas vs {semicolon_count} semicolons)")
            
            reader = csv.reader(csvfile, delimiter=delimiter)
            
            # Skip header if present
            if has_header:
                header_row = next(reader, None)
                if header_row:
                    print(f"Header: {header_row}")
                    total_lines += 1
            
            for line_num, row in enumerate(reader, start=(2 if has_header else 1)):
                total_lines += 1
                
                # Show first few rows for debugging
                if line_num <= 5:
                    print(f"Line {line_num}: {row} (length: {len(row)})")
                
                # Check if row has enough columns
                required_columns = max(id_column, name_column) + 1
                if len(row) < required_columns:
                    print(f"Line {line_num}: Skipped - not enough columns (has {len(row)}, needs {required_columns})")
                    skipped_lines += 1
                    continue
                
                # Extract ID and name
                try:
                    material_id = row[id_column].strip() if id_column < len(row) else ""
                    material_name = row[name_column].strip() if name_column < len(row) else ""
                except IndexError:
                    print(f"Line {line_num}: Index error accessing columns {id_column}, {name_column}")
                    skipped_lines += 1
                    continue
                
                # Check for empty values
                if not material_id or not material_name:
                    print(f"Line {line_num}: Skipped - empty values (ID: '{material_id}', Name: '{material_name}')")
                    skipped_lines += 1
                    continue
                
                materials.append((material_id, material_name))
                processed_lines += 1
        
        print(f"\nFile '{file_path}' summary:")
        print(f"  Total lines: {total_lines}")
        print(f"  Processed materials: {processed_lines}")
        print(f"  Skipped lines: {skipped_lines}")
        print(f"  Using delimiter: '{delimiter}'")
        print(f"  ID column: {id_column}, Name column: {name_column}")
                        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading CSV file '{file_path}': {e}")
        return []
    
    return materials

def compare_csv_materials(file1_path, file2_path, output_path, 
                         file1_id_col=0, file1_name_col=1,
                         file2_id_col=0, file2_name_col=1,
                         has_header=True, similarity_threshold=0.0,
                         matching_mode='difflib', llm_model_name='paraphrase-multilingual-MiniLM-L12-v2',
                         local_models_dir='models'):
    """
    Compare materials from two CSV files and create a matching report
    
    Args:
        file1_path (str): Path to first CSV file (source)
        file2_path (str): Path to second CSV file (target for matching)
        output_path (str): Path for output CSV file
        file1_id_col (int): ID column index in first file
        file1_name_col (int): Name column index in first file
        file2_id_col (int): ID column index in second file
        file2_name_col (int): Name column index in second file
        has_header (bool): Whether CSV files have headers
        similarity_threshold (float): Minimum similarity score to consider a match
        matching_mode (str): Matching algorithm ('difflib', 'llm', 'hybrid')
        llm_model_name (str): Name of the sentence transformer model for LLM matching
        local_models_dir (str): Directory where local models are stored
    
    Returns:
        dict: Summary of comparison results
    """
    print(f"Reading materials from '{file1_path}'...")
    materials1 = read_csv_materials(file1_path, file1_id_col, file1_name_col, has_header)
    
    print(f"Reading materials from '{file2_path}'...")
    materials2 = read_csv_materials(file2_path, file2_id_col, file2_name_col, has_header)
    
    if not materials1:
        return {"error": f"No materials found in '{file1_path}'"}
    if not materials2:
        return {"error": f"No materials found in '{file2_path}'"}
    
    print(f"Found {len(materials1)} materials in file 1 and {len(materials2)} materials in file 2")
    print(f"Using matching mode: {matching_mode}")
    
    # Initialize LLM matcher if needed
    llm_matcher = None
    if matching_mode in ['llm', 'hybrid']:
        if not HAS_LLM_SUPPORT:
            return {"error": "LLM support not available. Please install required packages: pip install sentence-transformers scikit-learn"}
        try:
            llm_matcher = LLMMaterialMatcher(llm_model_name, local_models_dir)
        except Exception as e:
            return {"error": f"Failed to initialize LLM matcher: {e}"}
    
    print("Finding best matches...")
    results = []
    matches_found = 0
    
    # Find best match for each material in file1
    for i, (id1, name1) in enumerate(materials1):
        try:
            best_id2, best_name2, similarity = find_best_match(
                name1, materials2, similarity_threshold, matching_mode, llm_matcher
            )
        except Exception as e:
            print(f"Error processing material '{name1}': {e}")
            best_id2, best_name2, similarity = None, None, 0.0
        
        result = {
            'Material_ID_File1': id1,
            'Material_Name_File1': name1,
            'Material_ID_File2': best_id2 if best_id2 else 'NO_MATCH',
            'Material_Name_File2': best_name2 if best_name2 else 'NO_MATCH',
            'Similarity_Coefficient': similarity,
            'Matching_Method': matching_mode
        }
        results.append(result)
        
        if best_id2:
            matches_found += 1
        
        # Progress indicator
        if (i + 1) % 100 == 0 or (i + 1) == len(materials1):
            print(f"Processed {i + 1}/{len(materials1)} materials...")
    
    # Write results to CSV
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
            fieldnames = ['Material_ID_File1', 'Material_Name_File1', 'Material_ID_File2', 'Material_Name_File2', 'Similarity_Coefficient', 'Matching_Method']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';', quoting=csv.QUOTE_NONE, escapechar='\\')
            
            writer.writeheader()
            writer.writerows(results)
            
        print(f"Results written to '{output_path}'")
        
    except Exception as e:
        return {"error": f"Error writing output file: {e}"}
    
    # Summary statistics
    summary = {
        'total_materials_file1': len(materials1),
        'total_materials_file2': len(materials2),
        'matches_found': matches_found,
        'no_matches': len(materials1) - matches_found,
        'match_rate': matches_found / len(materials1) * 100,
        'output_file': output_path
    }
    
    return summary

def print_comparison_summary(summary):
    """Print formatted summary of CSV comparison results"""
    if 'error' in summary:
        print(f"Error: {summary['error']}")
        return
    
    print(f"\n=== CSV MATERIAL COMPARISON SUMMARY ===")
    print(f"Total materials in file 1: {summary['total_materials_file1']}")
    print(f"Total materials in file 2: {summary['total_materials_file2']}")
    print(f"Matches found: {summary['matches_found']}")
    print(f"No matches: {summary['no_matches']}")
    print(f"Match rate: {summary['match_rate']:.1f}%")
    print(f"Output saved to: {summary['output_file']}")

def get_available_models():
    """Get list of recommended sentence transformer models for material matching"""
    models = {
        'all-MiniLM-L6-v2': {
            'description': 'Fast and lightweight model, good for general text similarity',
            'size': '~23MB',
            'speed': 'Very Fast'
        },
        'all-mpnet-base-v2': {
            'description': 'Higher quality embeddings, better semantic understanding',
            'size': '~438MB', 
            'speed': 'Medium'
        },
        'paraphrase-multilingual-MiniLM-L12-v2': {
            'description': 'Multilingual support, good for international material names',
            'size': '~471MB',
            'speed': 'Medium'
        },
        'sentence-transformers/all-distilroberta-v1': {
            'description': 'Balanced performance and speed',
            'size': '~326MB',
            'speed': 'Fast'
        }
    }
    return models

def print_matching_options():
    """Print available matching modes and model options"""
    print("\n=== MATERIAL MATCHING OPTIONS ===")
    print("\nMatching Modes:")
    print("  - 'difflib': Traditional string similarity (fast, no ML required)")
    print("  - 'llm': AI-powered semantic matching (requires model download)")
    print("  - 'hybrid': Combines both methods (recommended for best results)")
    
    print("\nAvailable LLM Models:")
    models = get_available_models()
    for model_name, info in models.items():
        print(f"  - {model_name}")
        print(f"    Description: {info['description']}")
        print(f"    Size: {info['size']}, Speed: {info['speed']}")
        print()

# Example usage functions
def example_basic_matching():
    """Example of basic difflib matching"""
    print("=== BASIC MATCHING EXAMPLE ===")
    summary = compare_csv_materials(
        'materials_file1.csv', 
        'materials_file2.csv', 
        'basic_output.csv',
        matching_mode='difflib',
        similarity_threshold=0.6
    )
    print_comparison_summary(summary)

def example_llm_matching():
    """Example of LLM-based matching"""
    print("=== LLM MATCHING EXAMPLE ===")
    summary = compare_csv_materials(
        'materials_file1.csv', 
        'materials_file2.csv', 
        'llm_output.csv',
        matching_mode='llm',
        llm_model_name='all-MiniLM-L6-v2',
        similarity_threshold=0.7
    )
    print_comparison_summary(summary)

def example_hybrid_matching():
    """Example of hybrid matching (recommended)"""
    print("=== HYBRID MATCHING EXAMPLE ===")
    summary = compare_csv_materials(
        'materials_file1.csv', 
        'materials_file2.csv', 
        'hybrid_output.csv',
        matching_mode='hybrid',
        llm_model_name='all-mpnet-base-v2',
        similarity_threshold=0.65
    )
    print_comparison_summary(summary)

# Main execution
if __name__ == "__main__":
    print("🔧 CSV Material Comparison Tool with LLM Support Ready!")
    print("=" * 60)
    
    print_matching_options()
    
    print("\n=== USAGE EXAMPLES ===")
    print("\n1. Basic string matching (fast, no ML):")
    print("   summary = compare_csv_materials('file1.csv', 'file2.csv', 'output.csv')")
    
    print("\n2. LLM semantic matching (AI-powered):")
    print("   summary = compare_csv_materials('file1.csv', 'file2.csv', 'output.csv',")
    print("                                  matching_mode='llm',")
    print("                                  llm_model_name='all-MiniLM-L6-v2')")
    
    print("\n3. Hybrid matching (recommended):")
    print("   summary = compare_csv_materials('file1.csv', 'file2.csv', 'output.csv',")
    print("                                  matching_mode='hybrid',")
    print("                                  similarity_threshold=0.65)")
    
    print("\n4. Print summary:")
    print("   print_comparison_summary(summary)")
    
    if not HAS_LLM_SUPPORT:
        print("\n⚠️  LLM Support Not Available")
        print("   To enable AI-powered matching, install:")
        print("   pip install sentence-transformers scikit-learn")
    else:
        print("\n✅ LLM Support Available - All matching modes ready!")
    
    print("\n" + "=" * 60)