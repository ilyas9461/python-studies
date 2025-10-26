# import sys
import cv2
import numpy as np
import os
from PIL import Image
import datetime
import time
import shutil

# Try to import VideoStream from imutils; if unavailable provide a lightweight fallback
try:
    from imutils.video import VideoStream
except Exception:
    VideoStream = None

if VideoStream is None:
    # Lightweight fallback implementation using OpenCV's VideoCapture
    class _VideoStreamFallback:
        def __init__(self, src=0):
            self.src = src
            self.cap = None

        def start(self):
            # On Windows, use DirectShow backend if available for better stability
            try:
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            except Exception:
                self.cap = cv2.VideoCapture(self.src)
            return self

        def read(self):
            if self.cap is None:
                return None
            ret, frame = self.cap.read()
            return frame if ret else None

        def stop(self):
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    VideoStream = _VideoStreamFallback


class ImageDatasetOrganizer():
    def __init__(self):
        self.path_face_images_dataset = os.path.join(os.getcwd(), 'dataset')
        self.recorder = FaceRecorder()
        self.last_id, self.names, self.id_list = self.recorder.get_uid_names()
        # print current dataset info
        print(self.last_id)
        print(self.id_list)
        print(self.names)

    def get_list(self):
        return self.id_list, self.names

    def organize_images(self, move_to_subfolders=False):
        """Rename images so they follow the pattern: id-name-index.jpg

        Args:
            move_to_subfolders (bool): If True, move each person's images into
                a folder named "{id}-{name}" under the dataset directory.

        Behavior:
        - Safely parses filenames. If a name is missing or unknown it will be
          added to the internal lists and assigned a new id.
        - Avoids overwriting by incrementing the index when destination exists.
        - Refreshes internal name/id lists at the end.
        """
        if not os.path.isdir(self.path_face_images_dataset):
            print("Dataset directory does not exist. Nothing to organize.")
            return

        # Refresh current lists
        self.last_id, self.names, self.id_list = self.recorder.get_uid_names()
        name_to_id = {n: i for n, i in zip(self.names, self.id_list)}

        # Collect files deterministically
        files = sorted(os.listdir(self.path_face_images_dataset))
        counts = {}

        for file in files:
            lower = file.lower()
            if not lower.endswith((".jpeg", ".jpg", ".png")):
                continue

            src = os.path.join(self.path_face_images_dataset, file)

            # Robust parsing of person name from filename
            base = os.path.splitext(file)[0]
            parts = base.split('-')
            if len(parts) >= 2:
                person_name = parts[1].strip()
            else:
                # Fallback: treat entire base as name
                person_name = parts[0].strip()

            if person_name == "":
                print(f"Skipping file with empty name: {file}")
                continue

            # Ensure name has an id, create one if missing
            if person_name not in name_to_id:
                # assign next numeric id (as string)
                try:
                    max_id = max(int(x) for x in self.id_list) if self.id_list else 0
                except Exception:
                    max_id = len(self.id_list)
                new_id = str(max_id + 1)
                name_to_id[person_name] = new_id
                self.names.append(person_name)
                self.id_list.append(new_id)

            person_id = name_to_id[person_name]
            idx = counts.get(person_name, 0)

            # Build destination file name and folder
            new_fname = f"{person_id}-{person_name}-{idx}.jpg"
            if move_to_subfolders:
                person_dir = os.path.join(self.path_face_images_dataset, f"{person_id}-{person_name}")
                os.makedirs(person_dir, exist_ok=True)
                dst = os.path.join(person_dir, new_fname)
            else:
                dst = os.path.join(self.path_face_images_dataset, new_fname)

            # Handle collisions by incrementing the index
            while os.path.exists(dst):
                idx += 1
                new_fname = f"{person_id}-{person_name}-{idx}.jpg"
                if move_to_subfolders:
                    dst = os.path.join(person_dir, new_fname)
                else:
                    dst = os.path.join(self.path_face_images_dataset, new_fname)

            try:
                os.rename(src, dst)
            except Exception as e:
                print(f"Failed to rename '{src}' -> '{dst}': {e}")
                # still increment count to avoid infinite loop on same file
                counts[person_name] = idx + 1
                continue

            counts[person_name] = idx + 1

        # Refresh internal lists and print result summary
        self.last_id, self.names, self.id_list = self.recorder.get_uid_names()
        print("Organize completed. Names:", self.names)
        print("IDs:", self.id_list)

    def delete_person(self, id):
        # Delete all images that belong to the person and remove their folder if present
        try:
            name = self.names[id]
        except Exception:
            print(f"Invalid id: {id}")
            return

        # First remove matching files anywhere under dataset
        removed_any = False
        for root, dirs, files in os.walk(self.path_face_images_dataset):
            for file in files:
                if file.lower().endswith(("jpeg", "jpg", "png")):
                    path = os.path.join(root, file)
                    parts = os.path.splitext(file)[0].split("-")
                    # robust: check there is at least a name token
                    if len(parts) >= 2:
                        person_name = parts[1]
                    else:
                        person_name = parts[0]
                    if name == person_name:
                        try:
                            os.remove(path)
                            removed_any = True
                        except Exception:
                            pass

        # Also remove the person's directory if it exists (format: {id}-{name})
        person_id = None
        try:
            person_id = self.id_list[id]
        except Exception:
            # fallback: try to find id from filenames
            for root, dirs, files in os.walk(self.path_face_images_dataset):
                for file in files:
                    if file.lower().endswith(("jpeg", "jpg", "png")):
                        parts = os.path.splitext(file)[0].split("-")
                        if len(parts) >= 2 and parts[1] == name:
                            person_id = parts[0]
                            break
                if person_id:
                    break

        if person_id is not None:
            person_dir = os.path.join(self.path_face_images_dataset, f"{person_id}-{name}")
            if os.path.isdir(person_dir):
                try:
                    shutil.rmtree(person_dir)
                    removed_any = True
                except Exception as e:
                    print(f"Failed to remove directory {person_dir}: {e}")

        if removed_any:
            print("Delete operation completed: removed files/folder for", name)
        else:
            print("No files or folder found for person:", name)

        # Refresh internal lists
        self.last_id, self.names, self.id_list = self.recorder.get_uid_names()
        print(self.names)

    def rename_person(self, id, new_name):
        name_index = 0
        old_name = self.names[id]
        print(old_name)
        for root, dirs, files in os.walk(self.path_face_images_dataset):
            for file in files:
                if file.endswith(("jpeg", "jpg", "png")):
                    path = os.path.join(root, file)
                    person_name = file.split("-")[1]
                    person_id = file.split("-")[0]
                    if old_name == person_name:
                        src = os.path.join(self.path_face_images_dataset, str(id) + '-' + old_name + '-' + str(name_index) + '.jpg')
                        dst = os.path.join(self.path_face_images_dataset, str(id) + '-' + new_name + '-' + str(name_index) + '.jpg')
                        try:
                            os.rename(src, dst)
                        except Exception:
                            pass
                        name_index = name_index + 1
        print("Rename operation completed...")
        self.last_id, self.names, self.id_list = self.recorder.get_uid_names()
        print(self.names)


class FaceRecorder():

    def __init__(self):
        self.path_face_images_dataset = os.path.join(os.getcwd(), 'dataset')

    def get_uid_names(self):
        prev_person_name = ""
        person_name = ""
        person_id = ""
        last_id = 0
        names = []
        id_list = []
        for root, dirs, files in os.walk(self.path_face_images_dataset):
            for file in files:
                if file.endswith(("jpeg", "jpg", "png")):
                    path = os.path.join(root, file)
                    person_name = file.split("-")[1]
                    person_id = file.split("-")[0]
                    if prev_person_name != person_name:
                        names.append(person_name)
                        id_list.append(person_id)
                        last_id = last_id + 1
                        prev_person_name = person_name
        return last_id, names, id_list

    def initialize_camera(self, max_retries=3):
        """Initialize camera with retry mechanism"""
        for attempt in range(max_retries):
            try:
                print(f"Attempting to initialize camera (attempt {attempt + 1}/{max_retries})...")
                cap = VideoStream(src=0).start()
                time.sleep(2.0)  # Give more time for camera initialization
                
                # Test if camera is working
                frame = cap.read()
                if frame is None:
                    raise Exception("Could not read frame from camera")
                    
                print("Camera initialized successfully!")
                return cap
            except Exception as e:
                print(f"Camera initialization failed: {str(e)}")
                if attempt < max_retries - 1:
                    print("Retrying camera initialization...")
                    time.sleep(2)
                    continue
                else:
                    raise Exception("Could not initialize camera after multiple attempts")
        return None

    def check_face_quality(self, face_roi, min_size=100, brightness_threshold=50):
        """Check if the face region meets quality standards"""
        # Check size
        h, w = face_roi.shape[:2]
        if h < min_size or w < min_size:
            return False, "Face too small"
            
        # Check brightness
        avg_brightness = np.mean(face_roi)
        if avg_brightness < brightness_threshold:
            return False, "Image too dark"
            
        # Check contrast
        contrast = np.std(face_roi)
        if contrast < 30:
            return False, "Low contrast"
            
        return True, "OK"

    def start_recording(self, face_id, names, name):
        if not os.path.exists('./dataset'):
            os.makedirs('./dataset')
        from . import get_cascade_file
        face_cascade = cv2.CascadeClassifier(get_cascade_file())

        print("ID:" + str(face_id))
        uname = name
        path = './dataset/'
        sampleNum = 1
        target_samples = 50  # Increased from 20 to 50 samples
        saving = True
        last_save_time = 0
        save_delay = 0.35  # Delay between captures in seconds
        stable_count = 0
        last_face_pos = None
        
        # Define pose suggestions for better variety
        pose_suggestions = [
            "Look straight at camera",
            "Slight right turn",
            "Slight left turn",
            "Tilt head slightly up",
            "Tilt head slightly down",
            "Neutral expression",
            "Smile",
            "Different lighting angle",
            "Slight right tilt",
            "Slight left tilt"
        ]
        
        # Guide rectangle for ideal face position
        def draw_guide(img):
            h, w = img.shape[:2]
            center_x, center_y = w // 2, h // 2
            guide_size = min(w, h) // 3
            cv2.rectangle(img, 
                         (center_x - guide_size//2, center_y - guide_size//2),
                         (center_x + guide_size//2, center_y + guide_size//2),
                         (0, 255, 0), 2)

        try:
            cap = self.initialize_camera()
            if cap is None:
                raise Exception("Failed to initialize camera")

            print("\nInstructions:")
            print("1. Position your face within the green rectangle")
            print("2. Stay still and look at the camera")
            print("3. Turn your head slightly between captures")
            print("4. Press ESC to cancel at any time\n")

            while True:
                try:
                    img = cap.read()
                    if img is None:
                        print("Warning: Could not read frame from camera")
                        continue

                    # Draw guide rectangle
                    draw_guide(img)

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,  # More precise detection
                        minNeighbors=6,    # More strict detection
                        minSize=(100, 100), # Larger minimum face size
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    current_time = time.time()
                    if len(faces) == 1 and saving:
                        x, y, w, h = faces[0]
                        
                        # Check face position stability
                        if last_face_pos is not None:
                            last_x, last_y = last_face_pos
                            if abs(x - last_x) < 10 and abs(y - last_y) < 10:
                                stable_count += 1
                            else:
                                stable_count = 0
                        last_face_pos = (x, y)

                        # Draw rectangle around detected face
                        color = (0, 255, 0) if stable_count >= 3 else (0, 255, 255)
                        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

                        # Check face quality and timing
                        if (current_time - last_save_time >= save_delay and 
                            stable_count >= 3 and w >= 100 and h >= 100):
                            
                            face_roi = gray[y:y+h, x:x+w]
                            quality_ok, reason = self.check_face_quality(face_roi)
                            
                            if quality_ok:
                                # Save the image (sample numbering is 1..N)
                                cv2.imwrite(path + str(face_id) + "-" + str(uname) + "-" + str(sampleNum) + ".jpg", face_roi)
                                print(f"Saved image {sampleNum}/{target_samples}")
                                sampleNum += 1
                                last_save_time = current_time
                                stable_count = 0  # Reset stability counter after saving
                            else:
                                cv2.putText(img, f"Quality check failed: {reason}", (5, 50), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                    elif len(faces) > 1:
                        cv2.putText(img, "Multiple faces detected", (5, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    elif len(faces) == 0:
                        cv2.putText(img, "No face detected", (5, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                    # Show progress (number of saved images so far)
                    saved_count = max(0, sampleNum - 1)
                    cv2.putText(img, f"Progress: {saved_count}/{target_samples}", (5, 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 255, 0), 2)

                    # Show pose suggestion
                    current_pose = pose_suggestions[min(sampleNum // 5, len(pose_suggestions) - 1)]
                    cv2.putText(img, f"Suggestion: {current_pose}", (5, 120), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

                    if saved_count >= target_samples:
                        saving = False
                        cv2.putText(img, "Recording finished. Press ESC or SPACE to exit...", 
                                  (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 255, 0), 2)

                    cv2.imshow("image", img)
                    
                    k = cv2.waitKey(30) & 0xff
                    if k == 27 or (saved_count >= target_samples and k == 32):  # ESC key or SPACE when finished
                        break

                except Exception as e:
                    print(f"Error during frame capture: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please ensure that:")
            print("1. Your webcam is properly connected")
            print("2. No other application is using the webcam")
            print("3. You have given camera access permissions to the application")
            return False
        finally:
            if 'cap' in locals() and cap is not None:
                cap.stop()
            cv2.destroyAllWindows()

        return True


class RecognizerTrainer():

    def __init__(self):
        self.dataset_path = './dataset'

    def get_images_with_ids(self, path):
        # Ensure recognizer directory exists
        if not os.path.exists('./recognizer'):
            os.makedirs('./recognizer')

        # Collect image file paths recursively and only include files (skip directories)
        valid_ext = ('.jpeg', '.jpg', '.png')
        imagePaths = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(valid_ext):
                    imagePaths.append(os.path.join(root, f))

        faces = []
        IDs = []

        for imagePath in sorted(imagePaths):
            # Safety: skip non-files or unreadable entries
            if not os.path.isfile(imagePath):
                continue
            try:
                faceImg = Image.open(imagePath).convert('L')
            except Exception as e:
                print(f"Skipping unreadable image '{imagePath}': {e}")
                continue

            faceNp = np.array(faceImg, 'uint8')

            # Extract ID from filename pattern 'id-name-idx.ext'
            try:
                ID = int(os.path.split(imagePath)[-1].split('-')[0])
            except Exception:
                # If the filename does not match expected pattern, skip it
                print(f"Skipping file with unexpected name format: {imagePath}")
                continue

            faces.append(faceNp)
            IDs.append(ID)

            # Optional small display to show progress during training
            try:
                cv2.imshow("training", faceNp)
                cv2.waitKey(10)
            except Exception:
                # Ignore display errors (headless environments)
                pass

        if len(IDs) == 0:
            raise Exception("No training images found in dataset. Make sure dataset contains image files.")

        return np.array(IDs), faces

    def start_training(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        Ids, faces = self.get_images_with_ids(self.dataset_path)
        recognizer.train(faces, Ids)
        recognizer.save('recognizer/trainingData.yml')
        cv2.destroyAllWindows()
        print("Training data file created...")


class FaceRecognizer:
    def __init__(self):
        self.face_images = os.path.join(os.getcwd(), "dataset")
        self.fname = os.path.join(os.getcwd(), "recognizer", "trainingData.yml")
        self.stop_requested = False

    def request_stop(self):
        """Request the recognition loop to stop gracefully from another thread."""
        self.stop_requested = True

    def initialize_camera(self, max_retries=3):
        """Initialize camera with retry mechanism"""
        for attempt in range(max_retries):
            try:
                print(f"Attempting to initialize camera (attempt {attempt + 1}/{max_retries})...")
                cap = VideoStream(src=0).start()
                time.sleep(2.0)  # Give more time for camera initialization

                # Test if camera is working
                frame = cap.read()
                if frame is None:
                    raise Exception("Could not read frame from camera")

                print("Camera initialized successfully!")
                return cap
            except Exception as e:
                print(f"Camera initialization failed: {str(e)}")
                if attempt < max_retries - 1:
                    print("Retrying camera initialization...")
                    time.sleep(2)
                    continue
                else:
                    raise Exception("Could not initialize camera after multiple attempts")
        return None

    def start_recognition(self):
        if not os.path.isfile(self.fname):
            print("Please train the data first")
            return False

        # Build a robust id->name mapping from dataset filenames
        recorder = FaceRecorder()
        _, names, id_list = recorder.get_uid_names()
        id_to_name = {}
        for n, pid in zip(names, id_list):
            try:
                id_to_name[int(pid)] = n
            except Exception:
                id_to_name[len(id_to_name)] = n

        from . import get_cascade_file
        face_cascade = cv2.CascadeClassifier(get_cascade_file())
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.fname)

        fmt = "%Y-%m-%d %H:%M:%S"

        # Track predictions across frames for each detected face region
        face_trackers = {}  # (x,y) -> [(label, conf), ...] history
        history_size = 3
        required_matches = 2
        CONF_THRESHOLD = 65.0

        try:
            # clear stop flag on start
            self.stop_requested = False
            cap = self.initialize_camera()
            if cap is None:
                raise Exception("Failed to initialize camera")

            while True:
                # allow external stop requests
                if getattr(self, 'stop_requested', False):
                    break

                img = cap.read()
                if img is None:
                    print("Warning: Could not read frame from camera")
                    continue

                matched_img = img.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                # Show number of faces detected and timestamp
                timestamp = datetime.datetime.strftime(datetime.datetime.now(), fmt)
                cv2.putText(matched_img, f"Detected: {len(faces)} faces", (5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 255, 0), 2)
                cv2.putText(matched_img, timestamp, (5, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 255, 0), 1)

                new_trackers = {}

                for (x, y, w, h) in faces:
                    cv2.rectangle(matched_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # find closest previous tracker
                    closest_pos = None
                    min_dist = float('inf')
                    for old_pos in face_trackers:
                        dist = abs(old_pos[0] - x) + abs(old_pos[1] - y)
                        if dist < min_dist and dist < 40:
                            min_dist = dist
                            closest_pos = old_pos

                    face_roi = gray[y:y + h, x:x + w]
                    try:
                        face_roi = cv2.equalizeHist(face_roi)
                    except Exception:
                        pass

                    label, conf = recognizer.predict(face_roi)

                    history = face_trackers.get(closest_pos, []) if closest_pos else []
                    history.append((label, conf))
                    if len(history) > history_size:
                        history = history[-history_size:]

                    new_trackers[(x, y)] = history

                    # count confident matches
                    matches = {}
                    for hlabel, hconf in history:
                        if hconf <= CONF_THRESHOLD:
                            matches[hlabel] = matches.get(hlabel, 0) + 1

                    best_label = None
                    most_matches = required_matches - 1
                    for hlabel, count in matches.items():
                        if count > most_matches:
                            most_matches = count
                            best_label = hlabel

                    if best_label is not None:
                        name = id_to_name.get(int(best_label), "Unknown")
                        cv2.putText(matched_img, name, (x + 2, y + h - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 255, 0), 2)

                        # Save or update single reference photo when very confident
                        try:
                            save_path = os.path.join(os.getcwd(), 'recognized')
                            os.makedirs(save_path, exist_ok=True)
                            save_name = f'{name}.jpg'
                            # stricter confidence required to save
                            if conf < 50.0:
                                info_img = matched_img.copy()
                                cv2.putText(info_img, f"Recognized: {name}", (10, info_img.shape[0] - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 255, 0), 2)
                                cv2.imwrite(os.path.join(save_path, save_name), info_img)
                        except Exception as e:
                            print(f"Failed to save recognition image: {e}")
                            pass
                    else:
                        cv2.putText(matched_img, "No Match", (x + 2, y + h - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                        if len(history) > 0:
                            last_conf = history[-1][1]
                            cv2.putText(matched_img, f"Conf: {last_conf:.1f}", (x + 2, y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                face_trackers = new_trackers

                cv2.imshow('Face Recognizer', matched_img)
                k = cv2.waitKey(30) & 0xff
                if k == 27:  # ESC
                    break

        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please ensure that:")
            print("1. Your webcam is properly connected")
            print("2. No other application is using the webcam")
            print("3. You have given camera access permissions to the application")
            return False
        finally:
            if 'cap' in locals() and cap is not None:
                cap.stop()
            cv2.destroyAllWindows()

        return True