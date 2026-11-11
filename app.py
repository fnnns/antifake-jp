"""AntiFake JP 実験版をローカルで操作する日本語デスクトップ画面。"""

from __future__ import annotations

import math
import os
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


BASE_DIR = Path(__file__).resolve().parent
AUDIO_TYPES = [("音声ファイル", "*.wav *.flac *.ogg"), ("すべてのファイル", "*.*")]


class AntiFakeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AntiFake JP — 日本語音声の保護")
        self.root.geometry("940x900")
        self.root.minsize(840, 840)
        self.root.configure(background="#f3f6fb")
        self.source = tk.StringVar()
        self.output = tk.StringVar()
        self.target = tk.StringVar()
        self.iterations = tk.StringVar(value="150")
        self.snr = tk.StringVar(value="30")
        self.device = tk.StringVar(value="自動（GPU 優先）")
        self.model = tk.StringVar(value="自動（登録された推奨モデル）")
        self.status = tk.StringVar(value="録音ファイルを選ぶと、処理を開始できます。")
        self.progress_text = tk.StringVar(value="待機中")
        self.result_text = tk.StringVar(value="処理後に、保存先と音質の参考値をここへ表示します。")
        self.worker: threading.Thread | None = None
        self.events: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy = False
        self.closing = False
        self.last_auto_output = ""
        self.last_output: Path | None = None
        self.last_report: Path | None = None
        self._style()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll)

    def _style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        font = ("Yu Gothic UI", 10)
        self.root.option_add("*Font", font)
        style.configure("TFrame", background="#f3f6fb")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("TLabel", background="#ffffff", foreground="#18324d", font=font)
        style.configure("Heading.TLabel", background="#f3f6fb", font=("Yu Gothic UI", 25, "bold"), foreground="#153858")
        style.configure("Subtitle.TLabel", background="#f3f6fb", foreground="#4f657c", font=font)
        style.configure("Section.TLabel", font=("Yu Gothic UI", 12, "bold"))
        style.configure("Muted.TLabel", foreground="#61758b", font=("Yu Gothic UI", 9))
        style.configure("Notice.TLabel", background="#fff3d8", foreground="#6b4a13", padding=(14, 10), font=("Yu Gothic UI", 9))
        style.configure("TEntry", padding=7)
        style.configure("TCombobox", padding=6)
        style.configure("TButton", padding=(13, 7), font=font)
        style.configure("Primary.TButton", background="#1764a0", foreground="#ffffff", padding=(20, 10), font=("Yu Gothic UI", 11, "bold"))
        style.map("Primary.TButton", background=[("disabled", "#a2b7c8"), ("active", "#124f80")], foreground=[("disabled", "#eef3f6")])
        style.configure("Horizontal.TProgressbar", troughcolor="#e6eef6", background="#247ebc", thickness=12)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=(28, 21, 28, 18))
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="AntiFake JP", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(outer, text="日本語の録音を、公開前にローカルで処理する音声保護ツール", style="Subtitle.TLabel").pack(anchor="w", pady=(2, 13))
        ttk.Label(
            outer,
            text="実験版・防御効果は未確立です。声の無断複製を確実に防ぐものではありません。\n出力は 16 kHz・モノラルの WAV です。元の録音は保存しておいてください。",
            style="Notice.TLabel",
            wraplength=830,
            justify="left",
        ).pack(fill="x", pady=(0, 14))

        files = ttk.Frame(outer, style="Card.TFrame", padding=18)
        files.pack(fill="x")
        files.columnconfigure(0, weight=1)
        ttk.Label(files, text="1  録音と保存先", style="Section.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        self.editable_widgets: list[ttk.Widget] = []
        self._file_row(files, 1, "保護する録音", self.source, self._choose_source)
        self._file_row(files, 3, "保護した音声の保存先", self.output, self._choose_output)
        self._file_row(files, 5, "別の話者の参考音声（任意）", self.target, self._choose_target)
        ttk.Label(files, text="参考音声を使う場合は、利用する権利のある別の話者の録音を選んでください。", style="Muted.TLabel").grid(row=7, column=0, columnspan=2, sticky="w", pady=(7, 0))

        settings = ttk.Frame(outer, style="Card.TFrame", padding=(18, 14, 18, 16))
        settings.pack(fill="x", pady=12)
        ttk.Label(settings, text="2  処理の設定", style="Section.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        settings.columnconfigure((0, 1, 2), weight=1)
        labels = ["繰り返し回数", "音質の目安（最小 SNR / dB）", "使用するデバイス"]
        values = [["50", "150", "300"], ["20", "25", "30", "35", "40", "45"], ["自動（GPU 優先）", "CPU", "NVIDIA GPU（CUDA）"]]
        variables = [self.iterations, self.snr, self.device]
        for index, (label, options, variable) in enumerate(zip(labels, values, variables)):
            ttk.Label(settings, text=label).grid(row=1, column=index, sticky="w", padx=(0, 12))
            combo = ttk.Combobox(settings, textvariable=variable, values=options, state="readonly", width=25)
            combo.grid(row=2, column=index, sticky="ew", padx=(0, 12), pady=(5, 0))
            self.editable_widgets.append(combo)
        ttk.Label(settings, text="SNR は大きいほど変化が小さくなります。回数や変化の大きさは、防御効果を保証しません。", style="Muted.TLabel").grid(row=3, column=0, columnspan=3, sticky="w", pady=(9, 0))
        ttk.Label(settings, text="使用するモデル").grid(row=4, column=0, sticky="w", pady=(11, 0))
        model_combo = ttk.Combobox(
            settings, textvariable=self.model,
            values=["自動（登録された推奨モデル）", "原版の事前学習モデル", "日本語で微調整したモデル"],
            state="readonly")
        model_combo.grid(row=4, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=(11, 0))
        self.editable_widgets.append(model_combo)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(0, 10))
        self.start_button = ttk.Button(actions, text="録音の保護を開始", style="Primary.TButton", command=self._start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="中止", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=9)
        self.play_source_button = ttk.Button(actions, text="元の録音を再生", command=lambda: self._play(self.source.get()))
        self.play_source_button.pack(side="right")
        self.play_output_button = ttk.Button(actions, text="保護音声を再生", command=lambda: self._play(str(self.last_output or "")), state="disabled")
        self.play_output_button.pack(side="right", padx=8)

        progress_card = ttk.Frame(outer, style="Card.TFrame", padding=(18, 12, 18, 14))
        progress_card.pack(fill="both", expand=True)
        ttk.Label(progress_card, textvariable=self.status, wraplength=810, justify="left").pack(anchor="w")
        self.progress_bar = ttk.Progressbar(progress_card, maximum=100)
        self.progress_bar.pack(fill="x", pady=(10, 5))
        ttk.Label(progress_card, textvariable=self.progress_text, style="Muted.TLabel").pack(anchor="w")
        ttk.Separator(progress_card).pack(fill="x", pady=12)
        ttk.Label(progress_card, textvariable=self.result_text, wraplength=810, justify="left").pack(anchor="w")
        output_actions = ttk.Frame(progress_card, style="Card.TFrame")
        output_actions.pack(fill="x", pady=(12, 0), side="bottom")
        self.folder_button = ttk.Button(output_actions, text="保存先フォルダーを開く", command=self._open_folder, state="disabled")
        self.folder_button.pack(side="left")
        self.report_button = ttk.Button(output_actions, text="処理レポートを開く", command=self._open_report, state="disabled")
        self.report_button.pack(side="left", padx=8)
        ttk.Label(output_actions, text="録音は外部へ送信されません", style="Muted.TLabel").pack(side="right")

    def _file_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, columnspan=2, sticky="w", pady=(7 if row > 1 else 0, 3))
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row + 1, column=0, sticky="ew", padx=(0, 9))
        button = ttk.Button(parent, text="選択…", command=command)
        button.grid(row=row + 1, column=1, sticky="e")
        self.editable_widgets.extend((entry, button))

    def _choose_source(self) -> None:
        path = filedialog.askopenfilename(title="保護する録音を選択", filetypes=AUDIO_TYPES)
        if path:
            self.source.set(path)
            if not self.output.get() or self.output.get() == self.last_auto_output:
                source_path = Path(path)
                candidate = source_path.with_name(source_path.stem + "_保護.wav")
                index = 2
                while candidate.exists():
                    candidate = source_path.with_name(f"{source_path.stem}_保護_{index}.wav")
                    index += 1
                self.last_auto_output = str(candidate)
                self.output.set(self.last_auto_output)

    def _choose_output(self) -> None:
        current = Path(self.output.get()) if self.output.get() else BASE_DIR / "保護音声.wav"
        path = filedialog.asksaveasfilename(title="保護音声の保存先", defaultextension=".wav", filetypes=[("WAV 音声", "*.wav")], initialdir=str(current.parent), initialfile=current.name)
        if path:
            self.output.set(path)

    def _choose_target(self) -> None:
        path = filedialog.askopenfilename(title="別の話者の参考音声を選択（任意）", filetypes=AUDIO_TYPES)
        if path:
            self.target.set(path)

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        for widget in self.editable_widgets:
            state = "disabled" if busy else "readonly" if isinstance(widget, ttk.Combobox) else "normal"
            widget.configure(state=state)
        self.start_button.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if busy else "disabled")

    def _start(self) -> None:
        if self.busy:
            return
        try:
            if not self.source.get().strip() or not self.output.get().strip():
                raise ValueError("保護する録音と保存先を選んでください。")
            source = Path(self.source.get()).expanduser().resolve()
            output = Path(self.output.get()).expanduser().resolve()
            target = Path(self.target.get()).expanduser().resolve() if self.target.get().strip() else None
            if not source.is_file():
                raise ValueError("保護する録音ファイルが見つかりません。")
            if target and not target.is_file():
                raise ValueError("参考音声ファイルが見つかりません。")
            if output == source or (target and output == target):
                raise ValueError("元の録音や参考音声と異なる保存先を選んでください。")
            if output.suffix.lower() != ".wav":
                raise ValueError("保存先の拡張子は .wav にしてください。")
            if output.exists():
                raise ValueError("保存先に同名のファイルがあります。別のファイル名を選んでください。")
            options = {
                "iterations": int(self.iterations.get()),
                "min_snr_db": float(self.snr.get()),
                "device": {"自動（GPU 優先）": "auto", "CPU": "cpu", "NVIDIA GPU（CUDA）": "cuda"}[self.device.get()],
                "model_name": {"自動（登録された推奨モデル）": "auto", "原版の事前学習モデル": "original", "日本語で微調整したモデル": "japanese"}[self.model.get()],
            }
        except (ValueError, OSError) as error:
            messagebox.showerror("入力内容を確認してください", str(error), parent=self.root)
            return
        self.cancel_event.clear()
        self._set_busy(True)
        self.progress_bar["value"] = 0
        self.progress_text.set("準備中…")
        self.status.set("音声とモデルを読み込んでいます。初回は時間がかかることがあります。")
        self.result_text.set("処理中です。元の録音は変更しません。")
        self.worker = threading.Thread(target=self._work, args=(source, output, target, options), daemon=True)
        self.worker.start()

    def _work(self, source: Path, output: Path, target: Path | None, options: dict) -> None:
        cancellation_type = None
        try:
            from core import CancelledError, ProtectionConfig, protect_file
            cancellation_type = CancelledError

            def progress(done: int, total: int, message: str = "") -> None:
                self.events.put(("progress", done, total, message))

            result = protect_file(source, output, target_path=target, config=ProtectionConfig(**options), progress=progress, cancel=self.cancel_event)
            self.events.put(("done", result))
        except Exception as error:
            if cancellation_type is not None and isinstance(error, cancellation_type):
                self.events.put(("cancelled",))
            else:
                self.events.put(("error", error.__class__.__name__, str(error)))

    def _poll(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    _, done, total, message = event
                    percent = max(0, min(100, done * 100 / total)) if total else 0
                    self.progress_bar["value"] = percent
                    self.progress_text.set(f"処理中  {percent:.0f}%  （{done} / {total}）")
                    if not self.cancel_event.is_set():
                        self.status.set(message or "録音の保護処理を進めています。")
                elif kind == "done":
                    self._set_busy(False)
                    self._show_result(event[1])
                elif kind == "cancelled":
                    self._set_busy(False)
                    self.status.set("処理を中止しました。")
                    self.progress_text.set("中止")
                    self.result_text.set("処理は完了していません。新しい設定で開始できます。")
                elif kind == "error":
                    self._set_busy(False)
                    self.status.set("処理を完了できませんでした。")
                    self.progress_text.set("エラー")
                    error_type, detail = event[1:]
                    if error_type in {"ModuleNotFoundError", "ImportError"}:
                        detail = "必要なライブラリが見つかりません。付属のセットアップ手順を実行してください。\n\n" + detail
                    elif error_type == "FileNotFoundError" and ("encoder" in detail.lower() or "model" in detail.lower()):
                        detail = "音声モデルが見つかりません。付属のセットアップ手順でモデルを用意してください。\n\n" + detail
                    self.result_text.set(detail)
                    if not self.closing:
                        messagebox.showerror("処理エラー", detail, parent=self.root)
        except queue.Empty:
            pass
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            self._stop_playback()
            self.root.destroy()
            return
        self.root.after(100, self._poll)

    @staticmethod
    def _format_number(value, digits: int = 2) -> str:
        try:
            number = float(value)
            if math.isinf(number) and number > 0:
                return "∞"
            return f"{number:.{digits}f}" if math.isfinite(number) else "算出なし"
        except (TypeError, ValueError):
            return "算出なし"

    def _show_result(self, result: dict) -> None:
        self.last_output = Path(result.get("output_path", self.output.get()))
        report_path = result.get("report_path")
        self.last_report = Path(report_path) if report_path else None
        self.progress_bar["value"] = 100
        self.progress_text.set("完了")
        self.status.set("保護した音声を保存しました。再生して聞こえ方を確認してください。")
        self.result_text.set(
            f"保存先：{self.last_output}\n"
            + (f"使用したモデル：{result['model_name_ja']}\n" if result.get("model_name_ja") else "")
            +
            f"SNR：{self._format_number(result.get('snr_db'))} dB　／　"
            f"処理前後の声紋類似度：{self._format_number(result.get('source_cosine_before_after'), 4)}\n"
            "声紋類似度は、使用したモデル内の参考値です。防御成功率を示すものではありません。"
        )
        self.play_output_button.configure(state="normal" if self.last_output.is_file() else "disabled")
        self.folder_button.configure(state="normal" if self.last_output.is_file() else "disabled")
        self.report_button.configure(state="normal" if self.last_report and self.last_report.is_file() else "disabled")

    def _cancel(self) -> None:
        if self.busy:
            self.cancel_event.set()
            self.cancel_button.configure(state="disabled")
            self.status.set("中止しています。現在の計算が終わるまでお待ちください。")

    def _play(self, path_text: str) -> None:
        path = Path(path_text) if path_text else None
        if not path or not path.is_file():
            messagebox.showinfo("音声を選択してください", "再生する録音ファイルが見つかりません。", parent=self.root)
            return
        if sys.platform != "win32" or path.suffix.lower() != ".wav":
            messagebox.showinfo("音声の再生", "この画面の再生ボタンは Windows の WAV ファイルに対応しています。\nFLAC / OGG はお使いの音声プレーヤーで確認してください。", parent=self.root)
            return
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except (OSError, RuntimeError) as error:
            messagebox.showerror("再生できません", f"この音声形式を再生できませんでした。別の音声プレーヤーで確認してください。\n\n{error}", parent=self.root)

    @staticmethod
    def _stop_playback() -> None:
        if sys.platform == "win32":
            try:
                import winsound

                winsound.PlaySound(None, 0)
            except (RuntimeError, OSError):
                pass

    def _open_folder(self) -> None:
        if self.last_output:
            self._open_path(self.last_output.parent)

    def _open_report(self) -> None:
        if self.last_report:
            self._open_path(self.last_report)

    def _open_path(self, path: Path) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(str(path))
            else:
                import subprocess

                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        except OSError as error:
            messagebox.showerror("開けません", f"次の場所を開けませんでした。\n{path}\n\n{error}", parent=self.root)

    def _close(self) -> None:
        if self.busy:
            self.closing = True
            self._cancel()
            self.status.set("処理を中止して終了しています。しばらくお待ちください。")
        else:
            self._stop_playback()
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    AntiFakeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
