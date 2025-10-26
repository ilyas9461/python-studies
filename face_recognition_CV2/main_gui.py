import threading
import queue
import tkinter as tk
from tkinter import simpledialog, messagebox
import time

# Import the classes from your package
from scripts.face_recognition_classes import (
    FaceRecorder,
    RecognizerTrainer,
    FaceRecognizer,
    ImageDatasetOrganizer,
)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Face Recognition Control Panel")
        root.geometry("520x420")

        self.log_queue = queue.Queue()

        # Menu bar (replaces the previous buttons)
        menubar = tk.Menu(root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.on_exit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Actions menu: primary actions that open OpenCV windows or start long tasks
        actions_menu = tk.Menu(menubar, tearoff=0)
        actions_menu.add_command(label="Record (Save image)", command=self.record)
        actions_menu.add_command(label="Train", command=self.train)
        actions_menu.add_command(label="Start Recognition", command=self.start_recognition)
        actions_menu.add_command(label="Stop Recognition", command=self.stop_recognition)
        actions_menu.add_command(label="Organize images", command=self.organize)
        menubar.add_cascade(label="Actions", menu=actions_menu)

        # Manage menu: dataset management utilities
        manage_menu = tk.Menu(menubar, tearoff=0)
        manage_menu.add_command(label="Delete person", command=self.delete_person)
        manage_menu.add_command(label="Rename person", command=self.rename_person)
        manage_menu.add_command(label="Refresh list", command=self.refresh_list)
        menubar.add_cascade(label="Manage", menu=manage_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Face Recognition Control Panel\nUses OpenCV windows for camera preview."))
        menubar.add_cascade(label="Help", menu=help_menu)

        root.config(menu=menubar)

        # Toolbar with Start/Stop icons
        toolbar = tk.Frame(root, relief=tk.RAISED, bd=1)
        # Use Unicode icons for cross-platform simplicity
        self.start_btn = tk.Button(toolbar, text='▶ Start', width=10, command=self.start_recognition)
        self.stop_btn = tk.Button(toolbar, text='⏹ Stop', width=10, command=self.stop_recognition, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=2, pady=2)
        self.stop_btn.pack(side=tk.LEFT, padx=2, pady=2)
        toolbar.pack(fill=tk.X, padx=4, pady=(4, 2))

        # Names list
        list_frame = tk.Frame(root)
        list_frame.pack(padx=8, pady=(0,8), fill=tk.BOTH, expand=False)
        tk.Label(list_frame, text="Known persons:").pack(anchor="w")
        self.names_listbox = tk.Listbox(list_frame, height=6)
        self.names_listbox.pack(fill=tk.X)

        # Log area
        log_frame = tk.Frame(root)
        log_frame.pack(padx=8, pady=8, fill=tk.BOTH, expand=True)
        tk.Label(log_frame, text="Log:").pack(anchor="w")
        self.log_text = tk.Text(log_frame, state=tk.DISABLED, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Initialize organizer and refresh list
        self.organizer = ImageDatasetOrganizer()
        self.refresh_list()

        # Poll the log queue
        self.root.after(100, self._poll_log)

    def _log(self, msg):
        self.log_queue.put(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    def _poll_log(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.configure(state=tk.NORMAL)
                self.log_text.insert(tk.END, msg)
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def run_in_thread(self, target, *args, **kwargs):
        def runner():
            try:
                self._log(f"Started: {target.__name__}")
                target(*args, **kwargs)
                self._log(f"Finished: {target.__name__}")
            except Exception as e:
                self._log(f"Error in {target.__name__}: {e}")
        t = threading.Thread(target=runner, daemon=True)
        t.start()
        return t

    def record(self):
        # Ask for name
        name = simpledialog.askstring("Record", "Enter name:")
        if not name:
            self._log("Record cancelled: no name given")
            return

        # Get next id and names
        recorder = FaceRecorder()
        last_id, names, id_list = recorder.get_uid_names()

        # last_id is the count; pass it through like the recorder expects
        self._log(f"Recording for name='{name}' (id={last_id})")
        # Run recording in background; it will open OpenCV windows
        self.run_in_thread(recorder.start_recording, last_id, names, name)

        # refresh list after a delay (recording will save files)
        self.root.after(5000, self.refresh_list)

    def train(self):
        trainer = RecognizerTrainer()
        self.run_in_thread(trainer.start_training)

    def recognize(self):
        # Backwards-compatible alias to start recognition
        self.start_recognition()

    def start_recognition(self):
        # Start recognition in background and keep a handle so we can stop it
        if getattr(self, 'current_recognizer', None) is not None:
            self._log('Recognition already running')
            return
        # update toolbar buttons
        try:
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
        except Exception:
            pass

        recognizer = FaceRecognizer()
        self.current_recognizer = recognizer

        def runner(r):
            try:
                self._log('Recognition started')
                r.start_recognition()
            except Exception as e:
                self._log(f'Recognition error: {e}')
            finally:
                self._log('Recognition stopped')
                self.current_recognizer = None
                try:
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                except Exception:
                    pass

        t = threading.Thread(target=runner, args=(recognizer,), daemon=True)
        t.start()

    def stop_recognition(self):
        # Request the running recognizer to stop
        if getattr(self, 'current_recognizer', None) is None:
            self._log('No recognition process running')
            return
        try:
            self.current_recognizer.request_stop()
            self._log('Stop requested for recognition')
            try:
                self.stop_btn.config(state=tk.DISABLED)
            except Exception:
                pass
        except Exception as e:
            self._log(f'Error requesting stop: {e}')

    def organize(self):
        # Ask whether to move into subfolders
        answer = messagebox.askyesno("Organize", "Move each person into a subfolder? (Yes = move)")
        self.run_in_thread(self.organizer.organize_images, answer)
        # refresh after some time
        self.root.after(2000, self.refresh_list)

    def delete_person(self):
        ids, names = self.organizer.get_list()
        if not names:
            messagebox.showinfo("Delete person", "No persons found in the dataset.")
            return
        # Ask for name to delete
        name = simpledialog.askstring("Delete person", f"Available names: {names}\nEnter name to delete:")
        if not name:
            self._log("Delete cancelled")
            return
        try:
            idx = names.index(name)
        except ValueError:
            self._log(f"Name not found: {name}")
            messagebox.showerror("Error", f"Name not found: {name}")
            return
        self.run_in_thread(self.organizer.delete_person, idx)
        self.root.after(2000, self.refresh_list)

    def rename_person(self):
        ids, names = self.organizer.get_list()
        if not names:
            messagebox.showinfo("Rename person", "No persons found in the dataset.")
            return
        old_name = simpledialog.askstring("Rename person", f"Available names: {names}\nEnter name to rename:")
        if not old_name:
            self._log("Rename cancelled")
            return
        try:
            idx = names.index(old_name)
        except ValueError:
            messagebox.showerror("Error", f"Name not found: {old_name}")
            return
        new_name = simpledialog.askstring("Rename person", "Enter new name:")
        if not new_name:
            self._log("Rename cancelled: no new name")
            return
        self.run_in_thread(self.organizer.rename_person, idx, new_name)
        self.root.after(2000, self.refresh_list)

    def refresh_list(self):
        try:
            ids, names = self.organizer.get_list()
        except Exception:
            # re-create organizer if something changed
            self.organizer = ImageDatasetOrganizer()
            ids, names = self.organizer.get_list()
        self.names_listbox.delete(0, tk.END)
        for n in names:
            self.names_listbox.insert(tk.END, n)
        self._log("Refreshed names list")

    def on_exit(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.root.quit()


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
