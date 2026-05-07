import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Add src/main to path for relative imports in source code
src_main_path = os.path.join(project_root, 'src', 'main')
sys.path.insert(0, src_main_path)
