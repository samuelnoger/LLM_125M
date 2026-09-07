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


class ChatApplication:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LLM Chat Assistant (SFT)")
        self.root.geometry("900x750")

        self.update_queue: queue.Queue[tuple[str, str] | tuple[str, None]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.response_start_index = "1.0"

        # Configuration variables (defaults to your SFT checkpoint)
        self.checkpoint_dir_var = tk.StringVar(value="checkpoints/sft_dialogue")
        self.device_var = tk.StringVar(value="auto")
        self.temperature_var = tk.DoubleVar(value=0.7)
        self.top_k_var = tk.IntVar(value=40)
        self.max_tokens_var = tk.IntVar(value=150)

        self._build_ui()
        self.root.after(100, self._poll_queue)

        # Welcome message in chat log
        self._append_system_message("System initialized. Ready to chat with your fine-tuned model!\n")

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)

        # Header Title
        title = ttk.Label(container, text="LLM Chat Interface", font=("Helvetica", 18, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        # Settings Frame
        settings_frame = ttk.LabelFrame(container, text="Model & Generation Settings", padding=8)
        settings_frame.pack(fill="x", pady=(0, 8))

        self._field(settings_frame, "Checkpoint", self.checkpoint_dir_var, 0, 0)
        self._field(settings_frame, "Device", self.device_var, 0, 2)

        sliders_frame = ttk.Frame(settings_frame)
        sliders_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=4)
        sliders_frame.columnconfigure((0, 1, 2), weight=1)

        self._slider(sliders_frame, "Temperature", self.temperature_var, 0.1, 1.0, 0)
        self._slider(sliders_frame, "Top-k", self.top_k_var, 0, 200, 1)
        self._slider(sliders_frame, "Max tokens", self.max_tokens_var, 1, 500, 2)

        # Chat History Log Area with Scrollbar
        chat_frame = ttk.Frame(container)
        chat_frame.pack(fill="both", expand=True, pady=(0, 8))

        self.chat_display = tk.Text(chat_frame, wrap="word", font=("Menlo", 11), bg="#fdfdfd", state="disabled")
        scrollbar = ttk.Scrollbar(chat_frame, orient="vertical", command=self.chat_display.yview)
        self.chat_display.configure(yscrollcommand=scrollbar.set)

        self.chat_display.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Configure tags for chat styling
        self.chat_display.tag_config("user", foreground="#1a73e8", font=("Menlo", 11, "bold"))
        self.chat_display.tag_config("assistant", foreground="#0d652d", font=("Menlo", 11))
        self.chat_display.tag_config("system", foreground="#5f6368", font=("Menlo", 10, "italic"))

        # Input Area at Bottom
        input_frame = ttk.Frame(container)
        input_frame.pack(fill="x", pady=(0, 4))

        self.input_entry = ttk.Entry(input_frame, font=("Menlo", 11))
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.input_entry.bind("<Return>", lambda event: self.send_message())

        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side="right")

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(container, textvariable=self.status_var, font=("Helvetica", 9))
        status_label.pack(anchor="w")

    def _field(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int, col: int) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Label(frame, text=label, width=10).pack(side="left")
        ttk.Entry(frame, textvariable=variable).pack(side="left", fill="x", expand=True)

    def _slider(self, parent: ttk.Frame, label: str, variable: tk.Variable, minimum: float, maximum: float, col: int) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=col, sticky="ew", padx=4)
        
        display_var = tk.StringVar()
        def update_label(*_):
            val = variable.get()
            if isinstance(variable, tk.DoubleVar):
                display_var.set(f"{val:.2f}")
            else:
                display_var.set(str(int(val)))
        update_label()
        variable.trace_add("write", update_label)

        ttk.Label(frame, text=label).pack(side="top", anchor="w")
        scale = ttk.Scale(frame, variable=variable, from_=minimum, to=maximum)
        scale.pack(side="top", fill="x", expand=True)
        ttk.Label(frame, textvariable=display_var).pack(side="top", anchor="e")

    def _append_system_message(self, text: str) -> None:
        self.chat_display.config(state="normal")
        self.chat_display.insert("end", text + "\n", "system")
        self.chat_display.config(state="disabled")
        self.chat_display.see("end")

    def send_message(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "Please wait for the current generation to finish.")
            return

        user_text = self.input_entry.get().strip()
        if not user_text:
            return

        self.input_entry.delete(0, "end")

        self.chat_display.config(state="normal")
        
        # FIX: Ensure we always start on a fresh line if the chat isn't empty
        if self.chat_display.index("end-1c") != "1.0":
            self.chat_display.insert("end", "\n\n")
            
        self.chat_display.insert("end", f"User: {user_text}\n", "user")
        self.chat_display.insert("end", "Assistant: ", "assistant")
        self.response_start_index = self.chat_display.index("end-1c")
        self.chat_display.config(state="disabled")
        self.chat_display.see("end")

        self.send_button.config(state="disabled")
        self.status_var.set("Generating response...")

        formatted_prompt = f"User: {user_text}\nAssistant:"

        self.worker_thread = threading.Thread(
            target=self._run_generation, 
            args=(formatted_prompt,), 
            daemon=True
        )
        self.worker_thread.start()

    def _run_generation(self, prompt_text: str) -> None:
        try:
            model, tokenizer, _ = load_checkpoint_bundle(
                self.checkpoint_dir_var.get(), 
                device=self.device_var.get()
            )
            prompt_tokens = tokenizer.encode(prompt_text).unsqueeze(0).to(next(model.parameters()).device)
            prompt_len = prompt_tokens.shape[1]

            target_tokens = int(self.max_tokens_var.get())
            safety_cap = target_tokens + 50  # Give it room to finish the sentence cleanly

            for tokens in iter_generate_tokens(
                model,
                prompt_tokens=prompt_tokens,
                max_new_tokens=safety_cap,
                temperature=float(self.temperature_var.get()),
                top_k=int(self.top_k_var.get()),
                repetition_penalty=1.15,
            ):
                # FIX: Slice the token array directly to prevent character-shifting bugs
                new_token_ids = tokens[0][prompt_len:]
                new_text = tokenizer.decode(new_token_ids.detach().cpu())
                
                # IMMEDIATE STOP: if the model outputs the end token
                if '<|endoftext|>' in new_text:
                    clean_text = new_text.split('<|endoftext|>')[0]
                    self.update_queue.put(("text", clean_text))
                    break

                self.update_queue.put(("text", new_text))

                # GRACEFUL STOP: Wait for punctuation once we pass the target token length
                if len(new_token_ids) >= target_tokens and new_text.endswith(('.', '!', '?', '\n\n')):
                    break

            self.update_queue.put(("done", None))
        except Exception as exc:
            self.update_queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.update_queue.get_nowait()
                if kind == "text" and payload is not None:
                    self.chat_display.config(state="normal")
                    self.chat_display.delete(self.response_start_index, "end-1c")
                    self.chat_display.insert("end-1c", payload, "assistant")
                    self.chat_display.config(state="disabled")
                    self.chat_display.see("end")
                    self.status_var.set("Generating...")
                elif kind == "done":
                    self.send_button.config(state="normal")
                    self.status_var.set("Ready")
                elif kind == "error" and payload is not None:
                    self.send_button.config(state="normal")
                    self.status_var.set("Error")
                    messagebox.showerror("Generation error", payload)
        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)


def main() -> None:
    root = tk.Tk()
    ChatApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()