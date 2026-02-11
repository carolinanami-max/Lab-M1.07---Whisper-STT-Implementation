# test_environment.py
import sys
print("Python version:", sys.version)
print("\nPython executable:", sys.executable)
print("\nPython path:")
for path in sys.path:
    print("  ", path)

print("\n" + "="*50)
print("Trying to import packages...")

try:
    import matplotlib
    print("✓ matplotlib version:", matplotlib.__version__)
except ImportError as e:
    print("✗ matplotlib import failed:", e)

try:
    import pandas as pd
    print("✓ pandas version:", pd.__version__)
except ImportError as e:
    print("✗ pandas import failed:", e)

try:
    import numpy as np
    print("✓ numpy version:", np.__version__)
except ImportError as e:
    print("✗ numpy import failed:", e)

try:
    import sklearn
    print("✓ sklearn version:", sklearn.__version__)
except ImportError as e:
    print("✗ sklearn import failed:", e)

print("\n" + "="*50)
print("Environment check complete!")
