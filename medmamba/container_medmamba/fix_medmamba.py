#!/usr/bin/env python3
"""Fix MedMamba.py by removing test code at the end"""

with open('MedMamba_temp.py', 'r') as f:
    lines = f.readlines()

# Find where the test code starts
test_start = None
for i, line in enumerate(lines):
    if line.strip().startswith('medmamba_t = VSSM('):
        test_start = i
        break

if test_start:
    # Keep everything before the test code
    clean_lines = lines[:test_start]
    
    # Add the MedMamba alias that train.py expects
    clean_lines.append('\n# Alias for backward compatibility\n')
    clean_lines.append('MedMamba = VSSM\n')
    
    with open('MedMamba.py', 'w') as f:
        f.writelines(clean_lines)
    
    print(f"Fixed MedMamba.py - removed {len(lines) - test_start} lines of test code")
else:
    print("Could not find test code to remove")
