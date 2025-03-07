#!/usr/bin/env python3

import glob
import os
import subprocess
import sys
from multiprocessing import Pool

clang_format_binary = '/usr/lib/llvm-18/bin/clang-format'
clang_format_cmd = [clang_format_binary, '-i', '-style=file']

def find_source_files(root_dir):
    source_files = glob.glob(os.path.join(root_dir, '**/*.cpp'), recursive=True);
    source_files.extend(glob.glob(os.path.join(root_dir, '**/*.hpp'), recursive=True));
    return source_files

def reformat_file(source_file):
    try:
        subprocess.run(clang_format_cmd + [source_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error formatting {source_file}: {e}", file=sys.stderr)
        return False
    return True

def run_clang_format(files):
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(reformat_file, files)
    
    return all(results)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    source_files = find_source_files(project_root)
    
    if not source_files:
        print("No source files found")
        return 1
    
    # Run clang-format
    if run_clang_format(source_files):
        print("Successfully formatted all files")
        return 0
    else:
        print("Failed to format some files", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
