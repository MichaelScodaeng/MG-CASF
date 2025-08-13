import os
import sys

print("🔍 Checking Pearl_PE folder structure...")
pearl_base = '/home/s2516027/GLCE/Pearl_PE'

def list_directory_structure(path, max_depth=3, current_depth=0):
    """Recursively list directory structure."""
    if current_depth > max_depth:
        return
    
    try:
        items = sorted(os.listdir(path))
        for item in items:
            item_path = os.path.join(path, item)
            indent = "  " * current_depth
            if os.path.isdir(item_path):
                print(f"{indent}📁 {item}/")
                list_directory_structure(item_path, max_depth, current_depth + 1)
            else:
                print(f"{indent}📄 {item}")
    except PermissionError:
        print(f"{indent}❌ Permission denied")
    except FileNotFoundError:
        print(f"{indent}❌ Directory not found")

print(f"\nStructure of {pearl_base}:")
list_directory_structure(pearl_base)

# Check for Python files that might contain the modules we need
print(f"\n🔍 Looking for Python files with relevant classes...")
search_terms = ['PEARLPositionalEncoder', 'GetSampleAggregator', 'GINSampleAggregator', 'MLP', 'Schema']

def search_for_classes(directory, terms):
    """Search for class definitions in Python files."""
    found = {}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for term in terms:
                            if f'class {term}' in content or f'def {term}' in content:
                                if term not in found:
                                    found[term] = []
                                rel_path = os.path.relpath(file_path, pearl_base)
                                found[term].append(rel_path)
                except Exception as e:
                    continue
    
    return found

found_classes = search_for_classes(pearl_base, search_terms)

for term, files in found_classes.items():
    if files:
        print(f"  ✓ {term} found in:")
        for file in files:
            print(f"    - {file}")
    else:
        print(f"  ❌ {term} not found")

print(f"\n🔍 Checking sys.path additions needed...")
python_files_dirs = set()
for root, dirs, files in os.walk(pearl_base):
    for file in files:
        if file.endswith('.py'):
            python_files_dirs.add(root)

print("Directories containing Python files:")
for dir_path in sorted(python_files_dirs):
    rel_path = os.path.relpath(dir_path, '/home/s2516027/GLCE')
    print(f"  {rel_path}")