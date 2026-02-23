#!/usr/bin/env python
import sys
import traceback

# Suppress numpy warnings
import warnings
warnings.filterwarnings('ignore')

try:
    print("Testing: from app.core.analysis import analyze_video")
    from app.core.analysis import analyze_video
    print("✓ SUCCESS: analyze_video loaded")
except Exception as e:
    print(f"✗ ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
