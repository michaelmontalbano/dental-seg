#!/usr/bin/env python3
"""
Simple runner script for creating three-class train/val split
"""

import subprocess
import sys
import os

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the main script
    main_script = os.path.join(script_dir, 'create_three_class_split.py')
    
    # Default arguments
    cmd = [
        sys.executable, main_script,
        '--bucket', 'codentist-general',
        '--dataset-prefix', 'datasets/bw_pa_pans',
        '--output-dir', './data',
        '--val-ratio', '0.2',
        '--random-seed', '42'
    ]
    
    print("Running three-class split with default parameters...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("\nScript completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nScript failed with return code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\nError running script: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
