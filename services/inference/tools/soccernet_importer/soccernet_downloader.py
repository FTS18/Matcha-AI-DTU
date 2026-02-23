import os
import argparse
import logging
from SoccerNet.Downloader import SoccerNetDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_soccernet_data(local_dir, tasks=["Spotting-v2"], leagues=["england_epl"]):
    """
    Downloads SoccerNet data using the official pip package.
    Note: Some features require signing an NDA.
    """
    os.makedirs(local_dir, exist_ok=True)
    
    # Initialize Downloader
    # You might need to set your password if you signed the NDA
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=local_dir)
    
    logger.info(f"🚀 Starting download for tasks: {tasks} in leagues: {leagues}")
    
    # Download annotations and labels (doesn't require NDA usually)
    for task in tasks:
        logger.info(f"📥 Downloading labels for {task}...")
        mySoccerNetDownloader.downloadDataTask(task=task)
        
    logger.info(f"✅ Metadata download complete for {leagues} in {local_dir}")
    print("\n👉 Note: To download HQ videos, you must sign the SoccerNet NDA.")
    print("👉 Check: https://github.com/SilvioGiancola/SoccerNetv2-DevKit#how-to-download-soccernet-v2")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official SoccerNet Downloader")
    parser.add_argument("--dir", type=str, default="data/soccernet", help="Local directory for data")
    parser.add_argument("--tasks", nargs="+", default=["spotting"], help="Tasks to download (e.g. spotting, tracking, reid)")
    
    args = parser.parse_args()
    
    # Check if SoccerNet is installed
    try:
        import SoccerNet
    except ImportError:
        print("❌ SoccerNet package not found. Run: pip install SoccerNet")
        exit(1)
        
    download_soccernet_data(args.dir, args.tasks)
