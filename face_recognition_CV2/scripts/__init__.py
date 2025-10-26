"""
scripts package for project modules.
Expose the main classes at package level for convenience.
"""
import os

def get_project_root():
    """Get absolute path to project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_cascade_file():
    """Get absolute path to the Haar cascade classifier file."""
    return os.path.join(get_project_root(), 'haarcascade_frontalface_default.xml')

from .face_recognition_classes import *
