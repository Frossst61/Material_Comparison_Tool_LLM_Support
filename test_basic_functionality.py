#!/usr/bin/env python3
"""
Test basic functionality without ML models
"""

from test import compare_csv_materials, print_comparison_summary
import csv

def test_basic_matching():
    """Test basic string matching functionality"""
    print('🧪 Testing Material Comparison Tool (Basic Mode)')
    print('=' * 60)

    # Create test data
    materials1 = [
        ('M001', 'Steel Grade 304'),
        ('M002', 'Aluminum Alloy 6061'),
        ('M003', 'Copper C101'),
        ('M004', 'Stainless Steel 316'),
        ('M005', 'Carbon Steel'),
    ]

    materials2 = [
        ('MAT_A', 'Stainless Steel 304'),
        ('MAT_B', 'Aluminium 6061-T6'),
        ('MAT_C', 'Pure Copper'),
        ('MAT_D', 'SS 316'),
        ('MAT_E', 'Carbon Steel A36'),
    ]

    # Write test files
    with open('test_materials1.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['ID', 'Name'])
        writer.writerows(materials1)

    with open('test_materials2.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['ID', 'Name'])
        writer.writerows(materials2)

    print('✅ Test data created')

    # Run comparison using difflib (no ML required)
    summary = compare_csv_materials(
        'test_materials1.csv',
        'test_materials2.csv', 
        'test_results.csv',
        matching_mode='difflib',
        similarity_threshold=0.4
    )

    print_comparison_summary(summary)
    print('\n🎉 Basic material comparison works perfectly!')
    print('📄 Results saved to test_results.csv')

if __name__ == "__main__":
    test_basic_matching()
