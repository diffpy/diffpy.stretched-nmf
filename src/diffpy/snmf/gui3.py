import tkinter as tk
from tkinter import messagebox
import os
import threading
import signal
import re

from tkinterdnd2 import TkinterDnD, DND_FILES
import numpy as np
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from pre_processing import run_preprocessing
from snmf_class import SNMFOptimizer


signal.signal(signal.SIGINT, signal.SIG_DFL)

input_files = []
output_path = ""


# --------------------------------------------------------------
#  Utility
# --------------------------------------------------------------

def extract_temperature(filename):
    match = re.search(r'(\d+)C', filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0


# --------------------------------------------------------------
#  RESULTS SCREEN (plots)
# --------------------------------------------------------------

class ResultsScreen:
    def __init__(self, master):
        self.master = master
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill="both", expand=True)

        self.done_label = tk.Label(self.main_frame, text="Done", font=("Helvetica", 16))
        self.done_label.pack(side="top", pady=10)

        self.left_btn_frame = tk.Frame(self.main_frame)
        self.left_btn_frame.pack(side="left", fill="y", padx=10)

        self.right_plot_frame = tk.Frame(self.main_frame)
        self.right_plot_frame.pack(side="right", fill="both", expand=True, padx=10)

        self.btn_components = tk.Button(
            self.left_btn_frame, text="Plot Components", command=self.plot_components
        )
        self.btn_components.pack(pady=5, fill="x")

        self.btn_weights = tk.Button(
            self.left_btn_frame, text="Plot Weights", command=self.plot_weights
        )
        self.btn_weights.pack(pady=5, fill="x")

        self.btn_stretch = tk.Button(
            self.left_btn_frame, text="Plot Stretching", command=self.plot_stretch
        )
        self.btn_stretch.pack(pady=5, fill="x")

        self.recover_btn = tk.Button(self.main_frame, text="Recover", command=self.recover_gui)
        self.recover_btn.pack(side="bottom", pady=10)

        self.current_plots = None

    # --------------- INTERNAL ---------------
    def clear_plots(self):
        if self.current_plots:
            self.current_plots.destroy()
        self.current_plots = tk.Frame(self.right_plot_frame)
        self.current_plots.pack(fill="both", expand=True)

    # --------------- PLOT 1: COMPONENTS ---------------

    filepath = os.path.join(output_path, "components.txt")
    print("Loading components from:", filepath)


    def plot_components(self):
        global output_path
        self.clear_plots()

        try:
            filepath = os.path.join(output_path, "components.txt")
            data = np.loadtxt(filepath, skiprows=1)
            x = data[:, 0]

            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)

            for i in range(1, data.shape[1]):
                ax.plot(x, data[:, i], label=f'Component {i}')

            ax.set_xlabel(r'2θ (degrees)')
            ax.set_ylabel('Intensity')
            ax.set_title('Components')
            ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=self.current_plots)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Error loading components:\n{str(e)}")

    # --------------- PLOT 2: WEIGHTS ---------------

    filepath = os.path.join(output_path, "components_weights.txt")
    print("Loading weights from:", filepath)


    def plot_weights(self):
        global output_path
        self.clear_plots()

        try:
            filepath = os.path.join(output_path, "components_weights.txt")

            with open(filepath, 'r') as f:
                header = f.readline().strip().split()

            df = pd.read_csv(filepath, delim_whitespace=True, skiprows=1, header=None)

            row_labels = df.iloc[:, 0].values
            data = df.iloc[:, 1:].values
            x = np.arange(len(header))

            fig = Figure(figsize=(7, 5))
            ax = fig.add_subplot(111)

            for i in range(data.shape[0]):
                ax.plot(x, data[i], marker='o', label=f"Component {i+1}")

            ax.set_ylabel('Weight')
            ax.set_xticks(x)
            ax.set_xticklabels(header, rotation=45, ha='right')
            ax.set_title('Component Weights')
            ax.legend(fontsize=8)

            canvas = FigureCanvasTkAgg(fig, master=self.current_plots)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Error loading weights:\n{str(e)}")

    # --------------- PLOT 3: STRETCH ---------------

    filepath = os.path.join(output_path, "stretch.txt")
    print("Loading stretch from:", filepath)


    def plot_stretch(self):
        global output_path
        self.clear_plots()

        try:
            filepath = os.path.join(output_path, "stretch.txt")

            with open(filepath, 'r') as f:
                header = f.readline().strip().split()

            temps = [extract_temperature(h) for h in header]
            sorted_idx = np.argsort(temps)

            df = pd.read_csv(filepath, delim_whitespace=True, skiprows=1, header=None)
            data = df.iloc[:, 1:].values
            data = data[:, sorted_idx]
            header = [header[i] for i in sorted_idx]
            temps = [temps[i] for i in sorted_idx]

            x = np.arange(len(header))

            fig = Figure(figsize=(7, 5))
            ax = fig.add_subplot(111)

            for i in range(data.shape[0]):
                ax.plot(x, data[i], marker='s', linestyle='--', label=f"Component {i+1}")

            ax.set_ylabel('Stretch Factor')
            ax.set_xticks(x)
            ax.set_xticklabels([f"{h}\n({t}°C)" for h, t in zip(header, temps)],
                               rotation=45, ha='right')

            ax.set_title('Component Stretching')
            ax.legend(fontsize=8)

            canvas = FigureCanvasTkAgg(fig, master=self.current_plots)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Error loading stretch:\n{str(e)}")

    def recover_gui(self):
        self.main_frame.destroy()
        initialize_main_gui()


# --------------------------------------------------------------
# EXECUTION PIPELINE (preprocessing + SNMF)
# --------------------------------------------------------------

def execute_pipeline(rows_number, n_components, output_path, input_files_list,
                     tol, max_iter, do_norm):
    try:
        # ---- PREPROCESSING ----
        root.after(0, lambda: loading_label.config(text="Preprocessing..."))
        MM, x_values, filenames = run_preprocessing(
            input_files_list,
            rows_number=int(rows_number),
            do_norm=bool(do_norm),
            verbose=True
        )

        # ---- SNMF ----
        root.after(0, lambda: loading_label.config(text="Running SNMF..."))

        model = SNMFOptimizer(
            source_matrix=MM,
            n_components=int(n_components)
        )

        if hasattr(model, "tol"): model.tol = float(tol)
        if hasattr(model, "max_iter"): model.max_iter = int(max_iter)

        if hasattr(model, "fit"):
            model.fit()

        # ---- SAVE OUTPUT ----
        output_path = os.path.abspath(output_path)
        os.makedirs(output_path, exist_ok=True)

        # COMPONENTS  (identico al tuo codice originale)
        combined_X = np.column_stack((x_values, model.components_))
        header_c = "2theta " + " ".join([f"component_{i+1}" for i in range(int(n_components))])

        np.savetxt(
            os.path.join(output_path, "components.txt"),
            combined_X,
            fmt="%.6g",
            delimiter=" ",
            header=header_c,
            comments=""
        )

        # WEIGHTS  (identico al tuo codice originale)
        row_headers = [f"component_{i+1}" for i in range(model.weights_.shape[0])]
        data_with_row_headers = np.column_stack([row_headers, model.weights_.astype(str)])
        header_w = " " + " ".join(filenames)

        np.savetxt(
            os.path.join(output_path, "components_weights.txt"),
            data_with_row_headers,
            fmt="%s",
            delimiter=" ",
            header=header_w,
            comments=""
        )

        # STRETCH (identico al tuo codice originale)
        row_headers_s = [f"component_{i+1}" for i in range(model.stretch_.shape[0])]
        data_with_row_headers_s = np.column_stack([row_headers_s, model.stretch_.astype(str)])
        header_s = " " + " ".join(filenames)

        np.savetxt(
            os.path.join(output_path, "stretch.txt"),
            data_with_row_headers_s,
            fmt="%s",
            delimiter=" ",
            header=header_s,
            comments=""
        )

        root.after(0, reset_gui)

    except Exception as e:
        import traceback
        traceback.print_exc()
        root.after(0, lambda: messagebox.showerror("Error", str(e)))
        root.after(0, reset_gui)


# --------------------------------------------------------------
# GUI MAIN
# --------------------------------------------------------------

def initialize_main_gui():
    global main_frame, left_frame, right_frame, run_button, loading_label
    global rows_entry, components_entry, tol_entry, max_iter_entry, output_entry
    global do_norm, file_list_label

    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    tk.Label(main_frame, text="sNMF analysis", font=("Helvetica", 16)).pack(side="top", pady=10)

    left_frame = tk.Frame(main_frame)
    left_frame.pack(side="left", fill="y", padx=10)

    do_norm = tk.BooleanVar(value=True)
    tk.Checkbutton(left_frame, text="Enable normalization", variable=do_norm).pack(anchor="w")

    def add_entry(label_text):
        tk.Label(left_frame, text=label_text).pack(anchor="w")
        entry = tk.Entry(left_frame)
        entry.pack(fill="x")
        return entry

    rows_entry = add_entry("Number of rows to discard:")
    components_entry = add_entry("Number of components:")
    tol_entry = add_entry("Tolerance:")
    max_iter_entry = add_entry("Maximum iterations:")
    output_entry = add_entry("Output folder path:")

    right_frame = tk.Frame(main_frame)
    right_frame.pack(side="right", fill="both", expand=True, padx=10)

    tk.Label(right_frame, text="Drag & Drop PXRD files (.xy / .xye)", font=("Helvetica", 11)).pack()

    drop_area = tk.Label(right_frame, text="Drop files here", relief="solid",
                         bg="white", width=40, height=10)
    drop_area.pack(pady=10)

    drop_area.drop_target_register(DND_FILES)
    drop_area.dnd_bind('<<Drop>>', on_drop)

    file_list_label = tk.Label(right_frame, text="", justify="left")
    file_list_label.pack()

    run_button = tk.Button(main_frame, text="Run", command=on_run)
    run_button.pack(side="bottom", pady=20)

    loading_label = tk.Label(main_frame, text="", font=("Helvetica", 14))
    loading_label.pack(side="bottom")


def on_drop(event):
    global input_files
    input_files = list(event.data.split())
    input_files.sort(key=lambda x: extract_temperature(os.path.basename(x)))
    file_list_label.config(text="\n".join(input_files))


def validate_inputs():
    if not input_files:
        messagebox.showerror("Error", "No files selected")
        return False

    for f in input_files:
        if not (f.endswith(".xy") or f.endswith(".xye")):
            messagebox.showerror("Error", "Files must be .xy or .xye")
            return False

    try:
        int(rows_entry.get())
        int(components_entry.get())
        float(tol_entry.get())
        int(max_iter_entry.get())
    except:
        messagebox.showerror("Error", "Invalid numeric input")
        return False

    return True


def show_loading_screen():
    for widget in main_frame.winfo_children():
        widget.pack_forget()
    loading_label.config(text="Processing...", font=("Helvetica", 16))
    loading_label.pack(pady=40)


def reset_gui():
    for widget in main_frame.winfo_children():
        widget.destroy()
    ResultsScreen(main_frame)


def on_run():
    global output_path

    if not validate_inputs():
        return

    output_folder = output_entry.get().strip()
    if output_folder == "":
        messagebox.showerror("Error", "Output folder missing")
        return

  
    output_folder = os.path.abspath(output_folder)
    output_path = output_folder 

    args = (
        rows_entry.get(),
        components_entry.get(),
        output_folder,
        input_files.copy(),
        tol_entry.get(),
        max_iter_entry.get(),
        int(do_norm.get())
    )

    show_loading_screen()
    threading.Thread(target=execute_pipeline, args=args, daemon=True).start()



def on_close():
    root.destroy()
    os._exit(0)


# --------------------------------------------------------------
# LAUNCH GUI
# --------------------------------------------------------------

root = TkinterDnD.Tk()
root.title("sNMF analysis")
root.protocol("WM_DELETE_WINDOW", on_close)

root.geometry("900x600")
initialize_main_gui()

root.mainloop()
