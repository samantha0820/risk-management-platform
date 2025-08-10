"""
Clear Data Cache Script
Used to clear old cached data and force re-fetching of latest data
"""

import os
import shutil
import config

def clear_cache():
    """Clear all cached data"""
    cache_dir = config.DATA_CACHE_DIR
    
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"✅ Cache directory cleared: {cache_dir}")
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
    else:
        print("ℹ️ Cache directory doesn't exist, no need to clear")
    
    # Recreate cache directory
    os.makedirs(cache_dir, exist_ok=True)
    print("✅ Cache directory recreated")

def clear_specific_cache(symbol=None):
    """Clear cache for specific stock"""
    cache_dir = config.DATA_CACHE_DIR
    
    if not os.path.exists(cache_dir):
        print("ℹ️ Cache directory doesn't exist")
        return
    
    if symbol:
        # Clear cache for specific stock
        pattern = f"{symbol}_"
        cleared = False
        for filename in os.listdir(cache_dir):
            if filename.startswith(pattern):
                file_path = os.path.join(cache_dir, filename)
                os.remove(file_path)
                print(f"✅ Cleared: {filename}")
                cleared = True
        
        if not cleared:
            print(f"ℹ️ No cache files found for {symbol}")
    else:
        # Clear all cache
        clear_cache()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        print(f"Clearing cache data for {symbol}...")
        clear_specific_cache(symbol)
    else:
        print("Clearing all cache data...")
        clear_cache()
    
    print("\n💡 Tip: The system will re-fetch latest data on next run")
