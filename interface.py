from __future__ import annotations

import queue
import threading
from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox, ttk

# Ensure the parent directory containing the 'LLM' package is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from LLM.generate import iter_generate_tokens, load_checkpoint_bundle




class TextGenerationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LLM Text Generator")
        self.root.geometry("900x700")

        self.update_queue: queue.Queue[tuple[str, str] | tuple[str, None]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None

        self.checkpoint_dir_var = tk.StringVar(value="checkpoints/pretrain_125M/")
        self.prompt_var = tk.StringVar(value="")
        self.device_var = tk.StringVar(value="auto")
        self.temperature_var = tk.DoubleVar(value=0.7)
        self.top_k_var = tk.IntVar(value=50)
        self.max_tokens_var = tk.IntVar(value=100)

        self._build_ui()
        self.root.after(100, self._poll_queue)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        title = ttk.Label(container, text="LLM Text Generator", font=("Helvetica", 22, "bold"))
        title.pack(anchor="w", pady=(0, 12))

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 12))

        self._field(controls, "Checkpoint dir", self.checkpoint_dir_var, 0)
        self._field(controls, "Prompt", self.prompt_var, 1)
        self._field(controls, "Device", self.device_var, 2)

        sliders = ttk.Frame(container)
        sliders.pack(fill="x", pady=(0, 12))

        self._slider(sliders, "Temperature", self.temperature_var, 0.1, 1.0, 0)
        self._slider(sliders, "Top-k", self.top_k_var, 0, 200, 1)
        self._slider(sliders, "Max new tokens", self.max_tokens_var, 1, 500, 2)

        button_row = ttk.Frame(container)
        button_row.pack(fill="x", pady=(0, 12))

        self.generate_button = ttk.Button(button_row, text="Generate", command=self.generate)
        self.generate_button.pack(side="left")

        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(button_row, textvariable=self.status_var)
        status.pack(side="left", padx=12)

        output_label = ttk.Label(container, text="Live output")
        output_label.pack(anchor="w")

        self.output_text = tk.Text(container, wrap="word", height=24, font=("Menlo", 12))
        self.output_text.pack(fill="both", expand=True)

        self.output_text.insert("1.0", "The generated text will appear here.")

    def _field(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=4)
        parent.columnconfigure(0, weight=1)
        ttk.Label(frame, text=label, width=14).pack(side="left")
        ttk.Entry(frame, textvariable=variable, width=80).pack(side="left", fill="x", expand=True)

    def _slider(self, parent: ttk.Frame, label: str, variable: tk.Variable, minimum: float, maximum: float, row: int) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=4)
        parent.columnconfigure(0, weight=1)
        ttk.Label(frame, text=label, width=14).pack(side="left")

        # Create a display StringVar formatted to 2 decimals for floats, or integers
        display_var = tk.StringVar()

        def update_label(*_):
            val = variable.get()
            if isinstance(variable, tk.DoubleVar):
                display_var.set(f"{val:.2f}")
            else:
                display_var.set(str(int(val)))

        update_label()
        variable.trace_add("write", update_label)

        scale = ttk.Scale(frame, variable=variable, from_=minimum, to=maximum)
        scale.pack(side="left", fill="x", expand=True)
        value_label = ttk.Label(frame, textvariable=display_var, width=8)
        value_label.pack(side="right")

    def generate(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Generation in progress", "Wait for the current generation to finish.")
            return

        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", self.prompt_var.get())
        self.generate_button.config(state="disabled")
        self.status_var.set("Loading checkpoint...")

        self.worker_thread = threading.Thread(target=self._run_generation, daemon=True)
        self.worker_thread.start()

    def _run_generation(self) -> None:
        try:
            model, tokenizer, _ = load_checkpoint_bundle(self.checkpoint_dir_var.get(), device=self.device_var.get())
            prompt_text = self.prompt_var.get()
            prompt_tokens = tokenizer.encode(prompt_text).unsqueeze(0).to(next(model.parameters()).device)

            target_tokens = int(self.max_tokens_var.get())
            # Add a safety buffer (+150 characters) so iter_generate_tokens doesn't hard-stop before finishing the sentence
            safety_cap = target_tokens + 150

            generated = ""
            for tokens in iter_generate_tokens(
                model,
                prompt_tokens=prompt_tokens,
                max_new_tokens=safety_cap,
                temperature=float(self.temperature_var.get()),
                top_k=int(self.top_k_var.get()),
            ):
                generated = tokenizer.decode(tokens[0].detach().cpu())
                self.update_queue.put(("text", generated))

                # Extract only the newly generated characters (excluding prompt)
                new_text = generated[len(prompt_text):]

                # Once target_tokens is reached, stop as soon as the active sentence finishes
                if len(new_text) >= target_tokens and new_text.endswith(('.', '!', '?', '\n\n')):
                    break

            self.update_queue.put(("done", None))
        except Exception as exc:  # pragma: no cover - UI error path
            self.update_queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.update_queue.get_nowait()
                if kind == "text" and payload is not None:
                    self.output_text.delete("1.0", "end")
                    self.output_text.insert("1.0", payload)
                    self.output_text.see("end")
                    self.status_var.set("Generating...")
                elif kind == "done":
                    self.generate_button.config(state="normal")
                    self.status_var.set("Done")
                elif kind == "error" and payload is not None:
                    self.generate_button.config(state="normal")
                    self.status_var.set("Error")
                    messagebox.showerror("Generation error", payload)
        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)


def main() -> None:
    root = tk.Tk()
    TextGenerationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()