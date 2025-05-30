"""
Configuration module for Strahlungserkennung project
Loads environment variables and provides default values
"""
import os
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system environment variables
    pass

class Config:
    """Configuration class with path settings"""
    
    # Image paths
    IMAGE_FOLDER_PATH = os.getenv('IMAGE_FOLDER_PATH', r'C:\default\path\to\images')
    OUTPUT_FOLDER_PATH = os.getenv('OUTPUT_FOLDER_PATH', r'C:\default\path\to\output')
    TEMP_FOLDER_PATH = os.getenv('TEMP_FOLDER_PATH', r'C:\temp')
    
    # Other settings
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def ensure_directories_exist(cls):
        """Create directories if they don't exist"""
        paths = [cls.IMAGE_FOLDER_PATH, cls.OUTPUT_FOLDER_PATH, cls.TEMP_FOLDER_PATH]
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)

# Usage example:
if __name__ == "__main__":
    config = Config()
    print(f"Image folder: {config.IMAGE_FOLDER_PATH}")
    print(f"Output folder: {config.OUTPUT_FOLDER_PATH}")
    print(f"Debug mode: {config.DEBUG_MODE}")
