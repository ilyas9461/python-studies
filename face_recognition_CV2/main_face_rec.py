# -*- coding: utf-8 -*-
import numpy as np
#import sqlite3
#import os
import time
import sys
from scripts.face_recognition_classes import (
    FaceRecorder,
    RecognizerTrainer,
    FaceRecognizer,
    ImageDatasetOrganizer,
)

def main():
    """Main program loop with error handling."""
    try:
        print("Initializing face recognition system...")
        organizer = ImageDatasetOrganizer()
        print("System initialized successfully!")

        while True:
            try:
                print("\n__ MENU ___")
                print("1- Save image (K)")
                print("2- Start train (E)")
                print("3- Face Recognition (Y)")
                print("4- Erase person (S)")
                print("5- Edit person (D)")
                print("6- Images organize (R)")
                print("7- Exit (Q)")

                operation = input("\nEnter your choice: ")
                
                if operation.upper() == "K":    # save persons
                    recorder = FaceRecorder()
                    face_id, names, id_list = recorder.get_uid_names()
                    name = input("Enter name for recording: ")
                    if name.strip():
                        recorder.start_recording(face_id, names, name)
                    else:
                        print("Error: Name cannot be empty")

                elif operation.upper() == "E":    # retrain recognizer
                    trainer = RecognizerTrainer()
                    trainer.start_training()

                elif operation.upper() == "Y":    # face recognition
                    recognizer = FaceRecognizer()
                    recognizer.start_recognition()

                elif operation.upper() == "S":    # delete person
                    ids, names = organizer.get_list()
                    if not names:
                        print("No persons found in the dataset.")
                        continue
                    print("Available names:", names)
                    delete_name = input("Name to delete: ")
                    try:
                        id = names.index(delete_name)
                        organizer.delete_person(id)
                        print(f"Successfully deleted {delete_name}")
                    except ValueError:
                        print(f"Error: \"{delete_name}\" not found in the dataset")
                        print("Available names are:", names)

                elif operation.upper() == "D":    # rename person
                    ids, names = organizer.get_list()
                    if not names:
                        print("No persons found in the dataset.")
                        continue
                    print("Available names:", names)
                    old_name = input("Name to change: ")
                    try:
                        id = names.index(old_name)
                        new_name = input("New name: ")
                        if not new_name.strip():
                            print("Error: New name cannot be empty")
                            continue
                        organizer.rename_person(id, new_name)
                        print(f"Successfully renamed {old_name} to {new_name}")
                    except ValueError:
                        print(f"Error: \"{old_name}\" not found in the dataset")
                        print("Available names are:", names)

                elif operation.upper() == "R":    # reorganize images
                    organizer.organize_images()
                    print("Images reorganization completed")

                elif operation.upper() == "Q":    # quit
                    print("Operations finished...")
                    return  # Exit the function cleanly

                else:
                    print("Invalid option. Please try again.")

            except Exception as e:
                print(f"Error during operation: {str(e)}")
                print("Please try again")
                time.sleep(1)  # Brief pause before showing menu again

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        return
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return 1
    
if __name__ == "__main__":
    sys.exit(main() or 0)