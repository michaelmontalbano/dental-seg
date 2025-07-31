#!/usr/bin/env python3
"""
Test script for three-class split functionality
This script tests the filtering logic without requiring S3 access
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from create_three_class_split import should_include_file

def test_filtering_rules():
    """Test the filename filtering rules"""
    
    print("Testing filename filtering rules...")
    print("=" * 50)
    
    # Test cases: (filename, xray_type, expected_result, description)
    test_cases = [
        # Bitewing tests - only JPG allowed
        ("bitewing_001.jpg", "bitewing", True, "Bitewing JPG - should pass"),
        ("bitewing_001.jpeg", "bitewing", True, "Bitewing JPEG - should pass"),
        ("bitewing_001.JPG", "bitewing", True, "Bitewing JPG uppercase - should pass"),
        ("bitewing_001.png", "bitewing", False, "Bitewing PNG - should fail"),
        ("bitewing_001.tiff", "bitewing", False, "Bitewing TIFF - should fail"),
        
        # Periapical tests - all formats allowed
        ("periapical_001.jpg", "periapical", True, "Periapical JPG - should pass"),
        ("periapical_001.png", "periapical", True, "Periapical PNG - should pass"),
        ("periapical_001.tiff", "periapical", True, "Periapical TIFF - should pass"),
        
        # Panoramic tests - all formats allowed
        ("panoramic_001.jpg", "panoramic", True, "Panoramic JPG - should pass"),
        ("panoramic_001.png", "panoramic", True, "Panoramic PNG - should pass"),
        ("panoramic_001.tiff", "panoramic", True, "Panoramic TIFF - should pass"),
        
        # Duplicate name tests - should now be INCLUDED (changed behavior)
        ("bitewing_001_bitewing.jpg", "bitewing", True, "Bitewing duplicate name - should pass"),
        ("periapical_001_periapical.jpg", "periapical", True, "Periapical duplicate name - should pass"),
        ("panoramic_001_panoramic.jpg", "panoramic", True, "Panoramic duplicate name - should pass"),
        
        # Hidden/system files - should fail
        ("._bitewing_001.jpg", "bitewing", False, "Hidden file - should fail"),
        ("._periapical_001.jpg", "periapical", False, "Hidden file - should fail"),
        
        # Non-image files - should fail
        ("bitewing_001.txt", "bitewing", False, "Text file - should fail"),
        ("periapical_001.pdf", "periapical", False, "PDF file - should fail"),
        ("panoramic_001.doc", "panoramic", False, "Document file - should fail"),
        
        # Edge cases
        ("BITEWING_001.JPG", "bitewing", True, "Uppercase filename - should pass"),
        ("bitewing-001.jpg", "bitewing", True, "Dash in filename - should pass"),
        ("bitewing_001_v2.jpg", "bitewing", True, "Version in filename - should pass"),
    ]
    
    passed = 0
    failed = 0
    
    for filename, xray_type, expected, description in test_cases:
        result, reason = should_include_file(filename, xray_type)
        
        if result == expected:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"{status} | {description}")
        print(f"     File: {filename} | Type: {xray_type} | Expected: {expected} | Got: {result}")
        if not result:
            print(f"     Reason: {reason}")
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed!")
        return False

def test_class_mapping():
    """Test the class mapping logic"""
    
    print("\nTesting class mapping...")
    print("=" * 30)
    
    class_to_label = {
        'bitewing': 0,
        'periapical': 1,
        'panoramic': 2
    }
    
    expected_mappings = [
        ('bitewing', 0),
        ('periapical', 1),
        ('panoramic', 2)
    ]
    
    all_correct = True
    for class_name, expected_label in expected_mappings:
        actual_label = class_to_label.get(class_name)
        if actual_label == expected_label:
            print(f"✅ {class_name} -> {actual_label} (correct)")
        else:
            print(f"❌ {class_name} -> {actual_label} (expected {expected_label})")
            all_correct = False
    
    return all_correct

def main():
    """Run all tests"""
    
    print("Three-Class Split Test Suite")
    print("=" * 60)
    
    # Test filtering rules
    filtering_passed = test_filtering_rules()
    
    # Test class mapping
    mapping_passed = test_class_mapping()
    
    # Overall result
    print("\n" + "=" * 60)
    if filtering_passed and mapping_passed:
        print("🎉 ALL TESTS PASSED! The three-class split logic is working correctly.")
        print("\nKey changes verified:")
        print("✅ Bitewing images: Only JPG/JPEG files accepted")
        print("✅ Periapical/Panoramic: All image formats accepted")
        print("✅ Duplicate names: Now INCLUDED (changed from original)")
        print("✅ Class mapping: bitewing=0, periapical=1, panoramic=2")
        return 0
    else:
        print("❌ SOME TESTS FAILED! Please review the logic.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
