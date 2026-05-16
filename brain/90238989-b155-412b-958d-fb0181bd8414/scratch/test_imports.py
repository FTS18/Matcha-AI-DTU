import sys
import os

# Add the services/inference directory to sys.path
sys.path.append(r"c:\Users\dubey\OneDrive\Desktop\Matcha-AI-DTU\services\inference")

try:
    from app.core.analysis import analyze_video
    print("SUCCESS: app.core.analysis.analyze_video imported successfully.")
    
    # Check if sub-modules are also accessible
    from app.core.video.reel_generator import create_highlight_reel
    print("SUCCESS: app.core.video.reel_generator imported successfully.")
    
    from app.core.visuals import VisualsManager
    print("SUCCESS: app.core.visuals.VisualsManager imported successfully.")
    
    from app.core.soccer_analysis.narrative import get_fallback_commentary
    print("SUCCESS: app.core.soccer_analysis.narrative imported successfully.")
    
except Exception as e:
    print(f"FAILURE: Import error - {e}")
    import traceback
    traceback.print_exc()
