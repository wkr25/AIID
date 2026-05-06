from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from predict import predict_image


class DetectorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Image Detector")
        self.root.geometry("960x720")
        self.root.minsize(860, 640)

        self.model_path = Path(__file__).with_name("detector.pth")
        self.image_path: str | None = None
        self.preview_image = None

        self.title_var = tk.StringVar(value="AI Image Detector")
        self.path_var = tk.StringVar(value="Open an image to inspect")
        self.pred_var = tk.StringVar(value="Prediction: -")
        self.ai_var = tk.StringVar(value="AI degree: -")
        self.real_var = tk.StringVar(value="Real degree: -")
        self.status_var = tk.StringVar(value=f"Model: {self.model_path.name}")

        self._build_ui()

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=16)
        header.pack(fill="x")

        ttk.Label(header, textvariable=self.title_var, font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(header, textvariable=self.status_var, foreground="#666666").pack(anchor="w", pady=(4, 0))

        actions = ttk.Frame(self.root, padding=(16, 0, 16, 8))
        actions.pack(fill="x")

        ttk.Button(actions, text="Open Image", command=self.open_image).pack(side="left")
        ttk.Button(actions, text="Predict", command=self.run_prediction).pack(side="left", padx=8)

        body = ttk.Frame(self.root, padding=16)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(body, text="Preview", padding=12)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.preview_label = ttk.Label(preview_frame, text="No image selected", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        info_frame = ttk.LabelFrame(body, text="Prediction", padding=12)
        info_frame.grid(row=0, column=1, sticky="nsew")
        info_frame.columnconfigure(0, weight=1)

        ttk.Label(info_frame, textvariable=self.path_var, wraplength=260, justify="left").grid(row=0, column=0, sticky="w", pady=(0, 12))
        ttk.Label(info_frame, textvariable=self.pred_var, font=("Segoe UI", 14, "bold")).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Label(info_frame, textvariable=self.ai_var, font=("Segoe UI", 12)).grid(row=2, column=0, sticky="w", pady=4)
        ttk.Label(info_frame, textvariable=self.real_var, font=("Segoe UI", 12)).grid(row=3, column=0, sticky="w", pady=4)

        self.progress = ttk.Progressbar(info_frame, orient="horizontal", length=260, mode="determinate", maximum=100)
        self.progress.grid(row=4, column=0, sticky="ew", pady=(18, 6))

        ttk.Label(info_frame, text="AI degree", foreground="#666666").grid(row=5, column=0, sticky="w")

        self.note_var = tk.StringVar(value="Select an image to run inference.")
        ttk.Label(self.root, textvariable=self.note_var, padding=(16, 0, 16, 16), foreground="#666666").pack(fill="x")

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.image_path = path
        self._show_preview(path)
        self.path_var.set(path)
        self.note_var.set("Image loaded. Running prediction...")
        self.run_prediction()

    def _show_preview(self, path: str) -> None:
        image = Image.open(path).convert("RGB")
        image.thumbnail((640, 420))
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")

    def run_prediction(self) -> None:
        if not self.image_path:
            messagebox.showinfo("No image", "Please open an image first.")
            return

        try:
            result = predict_image(self.image_path, model_path=self.model_path)
        except Exception as exc:
            messagebox.showerror("Prediction error", str(exc))
            self.note_var.set("Prediction failed.")
            return

        self.pred_var.set(f"Prediction: {result['prediction']}")
        self.ai_var.set(f"AI degree: {result['ai_degree'] * 100:.2f}%")
        self.real_var.set(f"Real degree: {result['real_degree'] * 100:.2f}%")
        self.progress["value"] = result["ai_degree"] * 100
        self.note_var.set("Prediction completed.")


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    DetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
