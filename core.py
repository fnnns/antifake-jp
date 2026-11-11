"""AntiFake（CCS 2023）を参考にした RTVC 単一モデルの実験用音声保護。

最適化処理は独自の簡略実装です。原版のアンサンブル構成は再現していません。
話者エンコーダーの推論構造は CorentinJ/RTVC（MIT）を参考にしています。
著作権表記は third_party/RTVC-LICENSE.txt、出典は THIRD_PARTY.md を参照してください。
"""
from __future__ import annotations

import bootstrap  # noqa: F401  専用ライブラリの場所を先に設定します。
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 16000
MODEL_SHA256 = "39373b86598fa3da9fcddee6142382efe09777e8d37dc9c0561f41f0070f134e"
MODEL_PATH = ROOT / "models" / "encoder.pt"
REGISTRY_PATH = ROOT / "models" / "registry.json"
Progress = Callable[[int, int, str], None]


class ProtectionError(RuntimeError):
    pass


class CancelledError(ProtectionError):
    pass


@dataclass(frozen=True)
class ProtectionConfig:
    iterations: int = 150
    min_snr_db: float = 30.0
    learning_rate: float = 0.001
    device: str = "auto"
    chunk_seconds: float = 4.0
    seed: int = 20261110
    model_name: str = "auto"

    def validate(self):
        if not 1 <= self.iterations <= 1000:
            raise ProtectionError("反復回数は 1〜1000 にしてください。")
        if not math.isfinite(self.min_snr_db) or not 20 <= self.min_snr_db <= 60:
            raise ProtectionError("SNR 下限は 20〜60 dB にしてください。")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.02:
            raise ProtectionError("学習率は 0 より大きく 0.02 以下にしてください。")
        if not math.isfinite(self.chunk_seconds) or not 2 <= self.chunk_seconds <= 10:
            raise ProtectionError("処理区間は 2〜10 秒にしてください。")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ProtectionError("デバイスは auto / cpu / cuda を指定してください。")
        if self.model_name not in {"auto", "original", "japanese"}:
            raise ProtectionError("モデルは auto / original / japanese を指定してください。")


class SpeakerEncoder(nn.Module):
    """RTVC の事前学習済み推論構造。MIT の著作権表記は third_party にあります。"""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(40, 256, 3, batch_first=True)
        self.linear = nn.Linear(256, 256)

    def forward(self, frames):
        _, (hidden, _) = self.lstm(frames)
        raw = torch.relu(self.linear(hidden[-1]))
        return raw / (torch.linalg.vector_norm(raw, dim=1, keepdim=True) + 1e-5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_model_spec(name="auto") -> dict:
    """登録済みモデルを解決し、保存場所と検証値を確認します。"""
    entries = {"original": {"path": "encoder.pt", "sha256": MODEL_SHA256,
                            "display_name_ja": "原版の事前学習モデル"}}
    active = "original"
    if REGISTRY_PATH.is_file():
        try:
            if REGISTRY_PATH.stat().st_size > 65536:
                raise ValueError("登録情報が大きすぎます。")
            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8-sig"))
            if registry.get("schema_version") != 1:
                raise ValueError("登録情報の形式に対応していません。")
            entries.update(registry["models"])
            active = registry["active"]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ProtectionError("モデル登録情報を読み込めません。registry.json を確認してください。") from exc
    selected = active if name == "auto" else name
    if selected not in {"original", "japanese"} or selected not in entries:
        raise ProtectionError("指定モデルがありません。日本語モデルの学習・登録を完了してください。")
    entry = entries[selected]
    try:
        path = (ROOT / "models" / entry["path"]).resolve()
        path.relative_to((ROOT / "models").resolve())
        checksum = entry["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("検証値の形式が正しくありません。")
        if selected == "original" and (checksum != MODEL_SHA256 or path != MODEL_PATH.resolve()):
            raise ValueError("原版モデルの登録情報は変更できません。")
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtectionError("モデルの保存場所または検証値が正しくありません。") from exc
    return {"name": selected, "path": path, "sha256": checksum,
            "display_name_ja": entry.get("display_name_ja", selected)}


def _load_encoder(device: torch.device, model_spec: dict):
    path = model_spec["path"]
    if not path.is_file():
        raise ProtectionError("モデルがありません。モデル取得.cmd を実行してください。")
    if _sha256(path) != model_spec["sha256"]:
        raise ProtectionError("モデルの検証に失敗しました。登録したモデルを確認してください。")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state"]
    model = SpeakerEncoder()
    expected = model.state_dict()
    model.load_state_dict({key: state[key] for key in expected}, strict=True)
    model.to(device)
    # cuDNN の LSTM は、重みを固定していても逆伝播時に学習モードを必要とします。
    # ドロップアウトはゼロなので、推論結果にランダムな変化は入りません。
    model.train()
    model.requires_grad_(False)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=400, win_length=400, hop_length=160,
        n_mels=40, norm="slaney", mel_scale="slaney", power=2.0,
    ).to(device)
    return model, mel


def _embed(wave: torch.Tensor, model, mel, gain: float = 1.0):
    # ゼロ埋めは解析時だけ行い、保存する録音は引き延ばしません。
    signal = F.pad(wave * gain, (0, max(0, 25600 - wave.numel())))
    frames = mel(signal).transpose(0, 1)
    starts = list(range(0, max(1, frames.shape[0] - 159), 80))
    tail = max(0, frames.shape[0] - 160)
    if starts[-1] != tail:
        if tail - starts[-1] < 40:
            starts[-1] = tail
        else:
            starts.append(tail)
    partials = torch.stack([frames[start:start + 160] for start in starts])
    embeddings = model(partials)
    return F.normalize(embeddings.mean(dim=0), dim=0)


def _analysis_gain(wave) -> float:
    rms = float(torch.sqrt(torch.mean(wave.square())).item())
    return max(1.0, 10 ** (-30 / 20) / max(rms, 1e-8))


def _read_audio(path: Path):
    if not path.is_file():
        raise ProtectionError(f"音声ファイルが見つかりません: {path.name}")
    try:
        info = sf.info(str(path))
        if info.duration < 0.5 or info.duration > 300:
            raise ProtectionError("0.5 秒以上、5 分以内の録音を選んでください。")
        if info.channels > 8 or info.samplerate < 8000 or info.samplerate > 192000:
            raise ProtectionError("対応範囲は 8〜192 kHz、8 チャンネル以下です。")
        data, original_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except ProtectionError:
        raise
    except Exception as exc:
        raise ProtectionError("音声を読み込めません。WAV / FLAC / OGG を選んでください。") from exc
    if not np.isfinite(data).all():
        raise ProtectionError("音声に無効な数値が含まれています。")
    if np.max(np.abs(data)) > 1.0001:
        raise ProtectionError("音声の振幅が範囲を超えています。音量を下げて保存してください。")
    wave = torch.from_numpy(data.mean(axis=1).copy())
    if original_rate != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, original_rate, SAMPLE_RATE)
    # 再サンプリングによる端数や振幅の増加で、基準音声が飽和しないようにします。
    if float(wave.abs().max()) > 1.0:
        wave = wave / (wave.abs().max() + 1e-7)
    if float(torch.sqrt(torch.mean(wave.square()))) < 1e-5:
        raise ProtectionError("無音、または音量が小さすぎます。別の録音を選んでください。")
    return wave.contiguous(), {
        "sample_rate": original_rate, "channels": info.channels,
        "frames": info.frames, "duration_seconds": info.duration,
    }


def _ranges(length: int, chunk_samples: int):
    overlap = int(0.25 * SAMPLE_RATE)
    regions = []
    start = 0
    while start < length:
        stop = min(length, start + chunk_samples)
        regions.append((start, stop))
        if stop == length:
            break
        start = stop - overlap
    return regions


def _snr(reference: np.ndarray, candidate: np.ndarray):
    signal = float(np.mean(reference.astype(np.float64) ** 2))
    noise = float(np.mean((candidate.astype(np.float64) - reference) ** 2))
    return 10 * math.log10(max(signal, 1e-30) / max(noise, 1e-30))


def protect_file(source_path, output_path, target_path=None, config=None,
                 progress: Progress | None = None, cancel: threading.Event | None = None) -> dict:
    config = config or ProtectionConfig()
    config.validate()
    started = time.perf_counter()
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    target = Path(target_path).expanduser().resolve() if target_path else None
    report_path = output.with_suffix(".report.json")
    if output.suffix.lower() != ".wav":
        raise ProtectionError("出力ファイル名の末尾は .wav にしてください。")
    if output in {source, target} or report_path in {source, target}:
        raise ProtectionError("元の録音・参照音声と同じ場所には保存できません。")
    if output.exists() or report_path.exists():
        raise ProtectionError("出力先にファイルがあります。別のファイル名を選んでください。")
    if target == source:
        raise ProtectionError("参照音声には別の話者の録音を選んでください。")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ProtectionError("CUDA が利用できません。自動または CPU を選んでください。")
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available()
                          else "cpu" if config.device == "auto" else config.device)
    # 小さな LSTM の処理では CPU のスレッド数を抑えます。
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    generator = torch.Generator(device=device).manual_seed(config.seed)

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise CancelledError("処理を中止しました。音声は保存されていません。")

    def emit(done, total, message):
        check_cancel()
        if progress:
            progress(done, total, message)

    emit(0, 1, "音声とモデルを読み込んでいます…")
    original, input_info = _read_audio(source)
    model_spec = get_model_spec(config.model_name)
    model, mel = _load_encoder(device, model_spec)
    target_embedding = None
    if target:
        target_wave, _ = _read_audio(target)
        # 参考音声の長さにかかわらず、解析に使うメモリー量を制限します。
        span = SAMPLE_RATE * 10
        candidates = [target_wave[start:start + span] for start in range(0, target_wave.numel(), span)]
        target_wave = max(candidates, key=lambda wave: float(wave.square().mean())).to(device)
        with torch.no_grad():
            target_embedding = _embed(target_wave, model, mel, _analysis_gain(target_wave))

    regions = _ranges(original.numel(), int(config.chunk_seconds * SAMPLE_RATE))
    total = len(regions) * config.iterations
    delta_sum = torch.zeros_like(original)
    weight_sum = torch.zeros_like(original)
    chunk_reports = []
    for index, (start, stop) in enumerate(regions):
        check_cancel()
        clean = original[start:stop].to(device)
        signal_rms = float(clean.square().mean().sqrt())
        if signal_rms < 1e-5:
            weight_sum[start:stop] += 1.0
            chunk_reports.append({"index": index, "status": "silent_skipped"})
            emit((index + 1) * config.iterations, total, "無音区間を保持しています…")
            continue
        gain = _analysis_gain(clean)
        with torch.no_grad():
            source_embedding = _embed(clean, model, mel, gain).detach()
        # RMS による変化量の上限を設けます。聴感上の音質を保証する指標ではありません。
        # 16 bit 量子化の誤差に備え、上限に少し余裕を残します。
        rms_budget = signal_rms / 10 ** (config.min_snr_db / 20) * 0.98
        peak_budget = min(0.03, rms_budget * 4)
        envelope = F.avg_pool1d(clean.square()[None, None], 801, stride=1,
                                padding=400).sqrt().flatten()
        envelope = (envelope / 0.01).clamp(0, 1).detach()
        noise = (torch.rand(clean.shape, device=device, generator=generator) * 2 - 1)
        delta = (noise * rms_budget * 0.1 * envelope).detach()

        def project(candidate):
            candidate = candidate.clamp(-peak_budget, peak_budget) * envelope
            rms = candidate.square().mean().sqrt().clamp_min(1e-12)
            candidate = candidate * torch.clamp(rms_budget / rms, max=1.0)
            return (clean + candidate).clamp(-1, 1) - clean

        def score(candidate):
            embedded = _embed(clean + candidate, model, mel, gain)
            if target_embedding is None:
                return torch.sum(embedded * source_embedding)
            return (torch.sum(embedded * source_embedding)
                    - torch.sum(embedded * target_embedding))

        best_delta = torch.zeros_like(clean)
        with torch.no_grad():
            best_score = float(score(best_delta))
        for iteration in range(config.iterations):
            check_cancel()
            delta = delta.detach().requires_grad_(True)
            loss = score(delta)
            if not torch.isfinite(loss):
                raise ProtectionError("最適化が不安定になりました。別の録音で試してください。")
            if float(loss.detach()) < best_score:
                best_score = float(loss.detach())
                best_delta = delta.detach().clone()
            grad, = torch.autograd.grad(loss, delta)
            # 更新幅を徐々に小さくし、制約の範囲へ戻しながら波形を最適化します。
            step = min(config.learning_rate, rms_budget * 0.5) * (0.3 + 0.7 * (1 - iteration / config.iterations))
            with torch.no_grad():
                delta = project(delta - step * grad.sign())
            if iteration % 5 == 0 or iteration + 1 == config.iterations:
                emit(index * config.iterations + iteration + 1, total,
                     f"区間 {index + 1}/{len(regions)} · 反復 {iteration + 1}/{config.iterations} · {device}")
        with torch.no_grad():
            final_score = float(score(delta))
            if final_score < best_score:
                best_delta = delta.detach()
            final_embedding = _embed(clean + best_delta, model, mel, gain)
            similarity = float(torch.sum(final_embedding * source_embedding))
        # 摂動だけを交差フェードし、元の時間軸と無音区間を保ちます。
        weights = torch.ones(stop - start)
        fade = min(int(0.25 * SAMPLE_RATE), weights.numel() // 2)
        if start:
            weights[:fade] = torch.linspace(0.001, 1, fade)
        if stop < original.numel():
            weights[-fade:] = torch.linspace(1, 0.001, fade)
        delta_sum[start:stop] += best_delta.cpu() * weights
        weight_sum[start:stop] += weights
        chunk_reports.append({"index": index, "start_seconds": start / SAMPLE_RATE,
                              "end_seconds": stop / SAMPLE_RATE,
                              "optimized_encoder_cosine": similarity})

    check_cancel()
    baseline = original.numpy()
    residual = (delta_sum / weight_sum.clamp_min(1e-8)).numpy()
    global_budget = float(np.sqrt(np.mean(baseline.astype(np.float64) ** 2))) / 10 ** (config.min_snr_db / 20) * 0.98
    residual_rms = float(np.sqrt(np.mean(residual.astype(np.float64) ** 2)))
    if residual_rms > global_budget:
        residual *= global_budget / residual_rms
    result = np.clip(baseline + residual, -1, 1)
    # 保存時の量子化を先に行い、実際の出力に対する SNR を測ります。
    encoded = np.clip(np.round(result * 32768), -32768, 32767).astype(np.int16)
    exported = encoded.astype(np.float32) / 32768
    snr_db = _snr(baseline, exported)
    if snr_db < config.min_snr_db - 0.1:
        raise ProtectionError("保存形式（16 bit WAV）で指定 SNR を保てません。SNR 下限を下げるか、音量を調整してください。")
    reference_cosines = []
    for index, (start, stop) in enumerate(regions):
        check_cancel()
        clean = original[start:stop].to(device)
        if float(clean.square().mean()) < 1e-10:
            continue
        with torch.no_grad():
            gain = _analysis_gain(clean)
            before = _embed(clean, model, mel, gain)
            after = _embed(torch.from_numpy(exported[start:stop].copy()).to(device), model, mel, gain)
            cosine = float(torch.sum(before * after))
            reference_cosines.append(cosine)
            chunk_reports[index]["exported_encoder_cosine"] = cosine
    report = {
        "schema_version": 1,
        "status": "completed_experimental",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "AntiFake を参考にした RTVC 単一モデルの射影勾配法による研究用原型",
        "input_path": str(source), "output_path": str(output), "report_path": str(report_path),
        "input_sha256": _sha256(source), "input_format": input_info,
        "target_path": str(target) if target else None,
        "target_sha256": _sha256(target) if target else None,
        "output_format": {"sample_rate": SAMPLE_RATE, "channels": 1, "subtype": "PCM_16", "frames": len(encoded)},
        "config": asdict(config), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__), "model_sha256": model_spec["sha256"],
        "model_name": model_spec["name"], "model_name_ja": model_spec["display_name_ja"],
        "snr_db": snr_db,
        "snr_reference": "16 kHz・モノラルに変換した入力音声を基準とし、PCM 量子化誤差を含みます。",
        "snr_budget_met": snr_db >= config.min_snr_db - 0.1,
        "source_cosine_before_after": float(np.mean(reference_cosines)) if reference_cosines else None,
        "cosine_scope": "最適化に使った RTVC の区間別平均診断値です。防御成功率を示すものではありません。",
        "japanese_cloning_evaluation": "not_performed",
        "intelligibility_evaluation": "not_performed",
        "chunks": chunk_reports,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations_ja": ["RTVC 単一モデルの研究用原型です。",
                           "日本語の音声明瞭度・クローン防御率は未検証です。",
                           "圧縮・再録音・ノイズ除去などで効果が変わる可能性があります。"],
    }
    check_cancel()
    output.parent.mkdir(parents=True, exist_ok=True)
    # 排他的にファイルを作成し、同時に保存された既存ファイルも上書きしません。
    created = []
    try:
        with output.open("xb") as stream:
            created.append(output)
            sf.write(stream, encoded, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        report["output_sha256"] = _sha256(output)
        with report_path.open("x", encoding="utf-8") as stream:
            created.append(report_path)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    except BaseException:
        for owned_path in created:
            owned_path.unlink(missing_ok=True)
        raise
    # 保存が正常に完了した後は、中止状態に戻しません。
    if progress:
        progress(total, total, "保存が完了しました。試聴で音質を確認してください。")
    return report
