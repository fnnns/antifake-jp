"""JVS の日本語音声を使う RTVC 話者エンコーダーの追加学習。"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: E402, F401
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchaudio  # noqa: E402
from core import SpeakerEncoder  # noqa: E402

SR = 16000
FRAMES = 160
SEED = 20261111
RESAMPLERS = {}


class JapaneseArgumentParser(argparse.ArgumentParser):
    """ヘルプの見出しを日本語で表示する。"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("add_help", False)
        super().__init__(*args, **kwargs)
        self._positionals.title = "位置引数"
        self._optionals.title = "オプション"
        self.add_argument("-h", "--help", action="help", help="この説明を表示して終了する")

    def format_help(self):
        return super().format_help().replace("usage:", "使い方:")

    def format_usage(self):
        return super().format_usage().replace("usage:", "使い方:")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mel_transform(device="cpu"):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=400, win_length=400, hop_length=160,
        n_mels=40, norm="slaney", mel_scale="slaney", power=2.0,
    ).to(device)


def parse_genders(archive):
    genders = {}
    for name in archive.namelist():
        if Path(name).name.lower() not in {"speaker_info.txt", "speaker_info.tsv", "speaker_info.csv", "gender_f0range.txt"}:
            continue
        raw = archive.read(name)
        for encoding in ("utf-8-sig", "cp932"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                text = ""
        for line in text.splitlines():
            fields = re.split(r"[\s,]+", line.strip())
            if len(fields) > 1 and re.fullmatch(r"jvs\d{3}", fields[0], flags=re.I):
                value = fields[1].lower()
                if value in {"m", "male", "男性"}:
                    genders[fields[0].lower()] = "male"
                elif value in {"f", "female", "女性"}:
                    genders[fields[0].lower()] = "female"
    return genders


def speaker_split(speakers, genders, seed=SEED):
    """同じ話者が複数の集合へ入らない固定分割を作る。"""
    speakers = sorted(speakers)
    if len(speakers) < 20:
        raise ValueError("追加学習には少なくとも 20 人の話者を用意してください。")
    rng = random.Random(seed)
    result = {"train": [], "val": [], "test": []}
    stratified = all(speaker in genders for speaker in speakers)
    if stratified:
        for gender in sorted(set(genders[speaker] for speaker in speakers)):
            group = [speaker for speaker in speakers if genders[speaker] == gender]
            rng.shuffle(group)
            n_val = max(1, round(len(group) * 0.1))
            n_test = max(1, round(len(group) * 0.1))
            result["val"].extend(group[:n_val])
            result["test"].extend(group[n_val:n_val + n_test])
            result["train"].extend(group[n_val + n_test:])
    else:
        rng.shuffle(speakers)
        n = max(2, round(len(speakers) * 0.1))
        result = {"val": speakers[:n], "test": speakers[n:n * 2], "train": speakers[n * 2:]}
    for split in result:
        result[split].sort()
    assert not (set(result["train"]) & set(result["val"]))
    assert not (set(result["train"]) & set(result["test"]))
    assert not (set(result["val"]) & set(result["test"]))
    return result, stratified


def prepare_wave(data: bytes):
    """学習用に音声を正規化し、長い無音を取り除く。"""
    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError("音声に非有限値があります。")
    wave = torch.from_numpy(audio.mean(axis=1).copy())
    if rate != SR:
        if rate not in RESAMPLERS:
            RESAMPLERS[rate] = torchaudio.transforms.Resample(rate, SR)
        wave = RESAMPLERS[rate](wave)
    if wave.numel() < 400:
        return None, "too_short", 0.0
    rms = float(wave.square().mean().sqrt())
    if rms < 1e-5:
        return None, "too_quiet", wave.numel() / SR
    # 30 ms のエネルギー判定に前後 120 ms の余白を加え、子音の切断を減らす。
    window = 480
    padded = F.pad(wave, (0, (-wave.numel()) % window))
    energies = padded.reshape(-1, window).square().mean(dim=1).sqrt()
    voiced = (energies > max(0.001, rms * 0.03)).float()
    voiced = F.max_pool1d(voiced[None, None], 9, stride=1, padding=4).flatten()
    mask = voiced.repeat_interleave(window)[:wave.numel()].bool()
    wave = wave[mask]
    if wave.numel() < 25600:
        return None, "too_short_after_vad", wave.numel() / SR
    rms = float(wave.square().mean().sqrt())
    wave = wave * max(1.0, 10 ** (-30 / 20) / max(rms, 1e-8))
    return wave, None, wave.numel() / SR


def preprocess(args):
    started = time.perf_counter()
    torch.set_num_threads(4)
    archive_path = Path(args.archive).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("status") == "completed" and old.get("archive_sha256") == sha256(archive_path):
            print("前処理済みデータとアーカイブを確認しました。再処理は不要です。", flush=True)
            return old
        raise ValueError("出力先に別の処理記録があります。新しい出力先を指定してください。")
    transform = mel_transform()
    records, discarded = [], []
    with zipfile.ZipFile(archive_path) as archive:
        members = []
        for name in archive.namelist():
            match = re.search(r"(?:^|/)(jvs\d{3})/(parallel100|nonpara30)/.*\.wav$", name, flags=re.I)
            if match:
                members.append((match[1].lower(), match[2].lower(), name))
        speakers = sorted({speaker for speaker, _, _ in members})
        genders = parse_genders(archive)
        splits, stratified = speaker_split(speakers, genders, args.seed)
        split_for = {speaker: split for split, selected in splits.items() for speaker in selected}
        print(f"日本語データ: {len(speakers)} 人、{len(members)} 発話。前処理を開始します。", flush=True)
        for index, (speaker, subset, member) in enumerate(sorted(members)):
            raw = archive.read(member)
            try:
                wave, reason, duration = prepare_wave(raw)
                if reason:
                    discarded.append({"speaker": speaker, "member": member, "reason": reason})
                    continue
                with torch.no_grad():
                    frames = transform(wave).transpose(0, 1).numpy().astype(np.float32)
                if len(frames) < FRAMES:
                    discarded.append({"speaker": speaker, "member": member, "reason": "too_few_frames"})
                    continue
                stem = Path(member).stem
                relative = Path(split_for[speaker]) / speaker / f"{subset}_{stem}.npy"
                destination = out / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ValueError(f"保存先の重複があります: {relative}")
                np.save(destination, frames, allow_pickle=False)
                records.append({"speaker": speaker, "split": split_for[speaker], "member": member,
                                "audio_sha256": hashlib.sha256(raw).hexdigest(), "feature_path": relative.as_posix(),
                                "frames": len(frames), "voiced_duration_seconds": duration,
                                "feature_sha256": sha256(destination)})
            except Exception as exc:
                raise RuntimeError(f"前処理に失敗しました: {member}") from exc
            if (index + 1) % 250 == 0:
                print(f"前処理 {index + 1}/{len(members)}、保存 {len(records)}、除外 {len(discarded)}", flush=True)
    usable = {speaker: sum(record["speaker"] == speaker for record in records) for speaker in speakers}
    member_counts = {speaker: {subset: sum(s == speaker and part == subset for s, part, _ in members)
                              for subset in ("parallel100", "nonpara30")} for speaker in speakers}
    expected_counts = {"parallel100": 100, "nonpara30": 30}
    deficits = [{"speaker": speaker, "subset": subset, "expected": expected_counts[subset], "found": count}
                for speaker, parts in member_counts.items() for subset, count in parts.items()
                if count != expected_counts[subset]]
    invalid = {speaker: count for speaker, count in usable.items() if count < 10}
    if invalid:
        raise ValueError(f"有効な発話数が 10 未満の話者があります: {invalid}")
    manifest = {
        "schema_version": 1, "status": "completed", "dataset": "JVS の通常読み上げ音声",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "archive_path": str(archive_path),
        "archive_sha256": sha256(archive_path), "seed": args.seed,
        "subsets": ["parallel100", "nonpara30"], "speaker_splits": splits,
        "gender_stratified": stratified, "genders": genders,
        "preprocessing": {"sample_rate": SR, "n_fft": 400, "hop_length": 160, "mel_channels": 40,
                          "mel_scale": "slaney", "mel_norm": "slaney", "power": 2.0,
                          "normalization": "increase_only_to_minus30dBFS", "vad": "30ms_energy_with_120ms_margin"},
        "records": records, "discarded": discarded, "utterances": len(records),
        "archive_member_counts": member_counts, "archive_count_deficits": deficits,
        "voiced_hours": sum(record["voiced_duration_seconds"] for record in records) / 3600,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(manifest_path, manifest)
    print(f"前処理完了: {len(records)} 発話、{manifest['voiced_hours']:.2f} 時間。", flush=True)
    return manifest


class TrainingEncoder(SpeakerEncoder):
    """既存の推論ネットワークへ GE2E の学習パラメーターを加える。"""
    def __init__(self):
        super().__init__()
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]))


def ge2e_loss(model, embeddings):
    """自分自身を除いた重心で同一話者の類似度を計算する。"""
    p, m, _ = embeddings.shape
    if p < 2 or m < 2:
        raise ValueError("GE2E には 2 人以上、各 2 発話以上が必要です。")
    inclusive = F.normalize(embeddings.mean(dim=1), dim=-1)
    exclusive = F.normalize((embeddings.sum(dim=1, keepdim=True) - embeddings) / (m - 1), dim=-1)
    similarity = torch.einsum("pmd,kd->pmk", embeddings, inclusive)
    own = (embeddings * exclusive).sum(dim=-1)
    identity = torch.eye(p, device=embeddings.device)[:, None, :]
    similarity = similarity * (1 - identity) + own[:, :, None] * identity
    logits = similarity * model.similarity_weight.clamp_min(1e-6) + model.similarity_bias
    targets = torch.arange(p, device=embeddings.device).repeat_interleave(m)
    return F.cross_entropy(logits.reshape(p * m, p), targets)


class FeatureStore:
    def __init__(self, root: Path, manifest):
        self.root = root
        self.records = manifest["records"]
        self.by_split = {}
        membership = {}
        for split, speakers in manifest["speaker_splits"].items():
            if split not in {"train", "val", "test"}:
                raise ValueError("不明なデータ分割があります。")
            for speaker in speakers:
                if speaker in membership:
                    raise ValueError("同じ話者が複数の分割に登録されています。")
                membership[speaker] = split
        paths, audio_splits = set(), {}
        for record in self.records:
            if membership.get(record["speaker"]) != record["split"]:
                raise ValueError("発話の話者分割が分割記録と一致しません。")
            relative = record["feature_path"]
            resolved = (root / relative).resolve()
            if not resolved.is_relative_to(root.resolve()) or relative in paths:
                raise ValueError("特徴量の保存先に重複または範囲外のパスがあります。")
            paths.add(relative)
            audio_hash = record["audio_sha256"]
            if audio_hash in audio_splits and audio_splits[audio_hash] != record["split"]:
                raise ValueError("同一の音声が複数の分割に含まれています。")
            audio_splits[audio_hash] = record["split"]
            self.by_split.setdefault(record["split"], {}).setdefault(record["speaker"], []).append(record["feature_path"])
        if set(membership) != {record["speaker"] for record in self.records}:
            raise ValueError("分割記録に発話のない話者が含まれています。")
        self.cache = OrderedDict()

    def verify_files(self):
        """学習を始める前に固定した全特徴量のハッシュを検証する。"""
        for index, record in enumerate(self.records):
            path = self.root / record["feature_path"]
            if not path.is_file() or sha256(path) != record["feature_sha256"]:
                raise ValueError(f"特徴量の検証に失敗しました: {record['feature_path']}")
            if (index + 1) % 2500 == 0:
                print(f"特徴量の検証 {index + 1}/{len(self.records)}", flush=True)

    def get(self, relative):
        if relative in self.cache:
            value = self.cache.pop(relative)
            self.cache[relative] = value
            return value
        value = np.load(self.root / relative, allow_pickle=False)
        if value.ndim != 2 or value.shape[1] != 40 or len(value) < FRAMES or not np.isfinite(value).all():
            raise ValueError(f"特徴量の形式が不正です: {relative}")
        self.cache[relative] = value
        if len(self.cache) > 128:
            self.cache.popitem(last=False)
        return value

    def batch(self, rng, p, m):
        group = self.by_split["train"]
        selected = rng.choice(sorted(group), size=p, replace=False)
        values = []
        for speaker in selected:
            utterances = rng.choice(group[speaker], size=m, replace=False)
            for relative in utterances:
                frames = self.get(relative)
                start = int(rng.integers(0, len(frames) - FRAMES + 1))
                values.append(frames[start:start + FRAMES])
        return torch.from_numpy(np.stack(values))


def evaluation_plan(store, split, seed, enrollment=3, probes=7):
    rng = random.Random(seed + (1 if split == "val" else 2))
    plan = {}
    for speaker, filenames in sorted(store.by_split[split].items()):
        filenames = sorted(filenames)
        rng.shuffle(filenames)
        if len(filenames) < enrollment + probes:
            raise ValueError(f"評価用の発話数が不足しています: {speaker}")
        plan[speaker] = {"enrollment": filenames[:enrollment], "probes": filenames[enrollment:enrollment + probes]}
    return plan


def eer_from_scores(labels, scores, return_threshold=False):
    """同点を一括処理した ROC 上で線形補間し、EER を求める。"""
    label_values = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if (label_values.ndim != 1 or scores.ndim != 1 or len(label_values) != len(scores)
            or len(scores) == 0 or not np.isfinite(scores).all()
            or not np.isin(label_values, [False, True, 0, 1]).all()):
        raise ValueError("EER のラベル・スコアの形状または値が不正です。")
    labels = label_values.astype(bool)
    if not labels.any() or labels.all():
        raise ValueError("EER の計算には同一話者と異なる話者の両方が必要です。")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    tp = np.r_[0, np.cumsum(sorted_labels)[ends]]
    fp = np.r_[0, np.cumsum(~sorted_labels)[ends]]
    far = fp / (~labels).sum()
    frr = 1 - tp / labels.sum()
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), sorted_scores[ends]]
    diff = far - frr
    right = int(np.flatnonzero(diff >= 0)[0])
    if right == 0:
        result = float((far[0] + frr[0]) / 2)
        threshold = float(thresholds[0])
        return (result, threshold) if return_threshold else result
    left = right - 1
    fraction = -diff[left] / max(diff[right] - diff[left], 1e-15)
    result = float(far[left] + fraction * (far[right] - far[left]))
    threshold = float(thresholds[left] + fraction * (thresholds[right] - thresholds[left]))
    return (result, threshold) if return_threshold else result


def utterance_partials(frames):
    starts = list(range(0, len(frames) - FRAMES + 1, 80))
    tail = len(frames) - FRAMES
    if starts[-1] != tail:
        if tail - starts[-1] < 40:
            starts[-1] = tail
        else:
            starts.append(tail)
    return np.stack([frames[start:start + FRAMES] for start in starts])


@torch.no_grad()
def evaluate_model(model, store, plan, device):
    model.eval()
    embeddings = {}
    for relative in sorted({filename for item in plan.values() for group in item.values() for filename in group}):
        partials = torch.from_numpy(utterance_partials(store.get(relative))).to(device)
        encoded = []
        for start in range(0, len(partials), 64):
            encoded.append(model(partials[start:start + 64]))
        embeddings[relative] = F.normalize(torch.cat(encoded).mean(dim=0), dim=0).cpu().numpy()
    enrollment = {speaker: np.mean([embeddings[name] for name in item["enrollment"]], axis=0)
                  for speaker, item in plan.items()}
    enrollment = {speaker: vector / max(np.linalg.norm(vector), 1e-12) for speaker, vector in enrollment.items()}
    labels, scores = [], []
    for speaker, item in sorted(plan.items()):
        for relative in item["probes"]:
            for enrolled_speaker, centroid in sorted(enrollment.items()):
                labels.append(speaker == enrolled_speaker)
                scores.append(float(np.dot(embeddings[relative], centroid)))
    eer, threshold = eer_from_scores(labels, scores, return_threshold=True)
    result = {"eer": eer, "eer_threshold_interpolated": threshold, "speakers": len(plan),
              "genuine_trials": int(sum(labels)), "impostor_trials": len(labels) - int(sum(labels)),
              "mean_genuine_cosine": float(np.mean(np.array(scores)[np.array(labels)])),
              "mean_impostor_cosine": float(np.mean(np.array(scores)[~np.array(labels)]))}
    result["cosine_separation_gap"] = result["mean_genuine_cosine"] - result["mean_impostor_cosine"]
    model.train()
    return result


def checkpoint_payload(model, optimizer, step, config, manifest_sha, pretrained_sha, best_eer):
    return {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "step": step,
            "config": config, "data_manifest_sha256": manifest_sha, "pretrained_sha256": pretrained_sha,
            "best_validation_eer": best_eer, "language": "ja", "method": "GE2E による話者モデルの日本語適応"}


def save_checkpoint(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    started = time.perf_counter()
    data = Path(args.data).resolve()
    pretrained = Path(args.pretrained).resolve()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("実行先が空ではありません。別の実行先を指定してください。")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("前処理が完了していません。")
    splits = manifest["speaker_splits"]
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(splits[a]) & set(splits[b]):
            raise ValueError("話者の分割が重複しています。")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA が利用できません。")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = TrainingEncoder().to(device)
    checkpoint = torch.load(pretrained, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    # 事前学習の最適化状態・反復回数を引き継がず、新しい Adam を使う。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = {key: value for key, value in vars(args).items() if key != "func"}
    manifest_sha, pretrained_sha = sha256(manifest_path), sha256(pretrained)
    write_json(run_dir / "config.json", config)
    store = FeatureStore(data, manifest)
    store.verify_files()
    if len(store.by_split["train"]) < args.speakers:
        raise ValueError("バッチ内の話者数が学習話者数を超えています。")
    if any(len(files) < args.utterances for files in store.by_split["train"].values()):
        raise ValueError("バッチ内で必要な発話数を確保できません。")
    val_plan = evaluation_plan(store, "val", args.seed)
    test_plan = evaluation_plan(store, "test", args.seed)
    write_json(run_dir / "evaluation_protocol.json", {"val": val_plan, "test": test_plan,
        "description_ja": "固定した登録 3 発話・照合 7 発話。検証話者でモデルを選び、テスト話者は選択に使いません。"})
    baseline_val = evaluate_model(model, store, val_plan, device)
    best_eer = math.inf
    best_criterion = (math.inf, math.inf)
    best_step = None
    best_validation = None
    log_path = run_dir / "history.jsonl"

    def log(value):
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    log({"step": 0, "validation": baseline_val, "event": "baseline"})
    print(f"開始時の検証 EER: {baseline_val['eer']:.4%}。日本語での追加学習を開始します。", flush=True)
    rng = np.random.default_rng(args.seed)
    losses = []
    stale = 0
    reason = "max_steps"
    step = 0
    completed_steps = 0
    model.train()
    try:
        for step in range(1, args.steps + 1):
            batch = store.batch(rng, args.speakers, args.utterances).to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(batch).reshape(args.speakers, args.utterances, -1)
            loss = ge2e_loss(model, embeddings)
            if not torch.isfinite(loss):
                raise RuntimeError("学習損失が非有限値になりました。")
            loss.backward()
            for parameter in (model.similarity_weight, model.similarity_bias):
                if parameter.grad is not None:
                    parameter.grad.mul_(0.01)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
            optimizer.step()
            completed_steps = step
            losses.append(float(loss.detach()))
            if step % args.log_every == 0:
                entry = {"step": step, "mean_training_loss": float(np.mean(losses)),
                         "elapsed_seconds": time.perf_counter() - started}
                log(entry)
                print(f"学習 {step}/{args.steps}、平均損失 {entry['mean_training_loss']:.5f}、経過 {entry['elapsed_seconds']:.1f} 秒", flush=True)
                losses.clear()
            if step % args.eval_every == 0 or step == args.steps:
                validation = evaluate_model(model, store, val_plan, device)
                criterion = (validation["eer"], -validation["cosine_separation_gap"])
                improved = criterion < best_criterion
                if improved:
                    best_eer, best_step, stale = validation["eer"], step, 0
                    best_criterion = criterion
                    best_validation = validation
                    save_checkpoint(run_dir / "best.pt", checkpoint_payload(model, optimizer, step, config, manifest_sha, pretrained_sha, best_eer))
                else:
                    stale += 1
                log({"step": step, "validation": validation, "selected": improved})
                save_checkpoint(run_dir / "last.pt", checkpoint_payload(model, optimizer, step, config, manifest_sha, pretrained_sha, best_eer))
                print(f"検証 {step}: EER {validation['eer']:.4%}、最良 {best_eer:.4%}（反復 {best_step}）", flush=True)
                if args.patience > 0 and stale >= args.patience and step >= args.min_steps:
                    reason = "validation_early_stopping"
                    break
    except KeyboardInterrupt:
        reason = "interrupted"
        print("中断を受け付けました。現在の状態を保存し、選択済みモデルを評価します。", flush=True)
    step = completed_steps
    if best_step is None:
        if step < 1:
            raise RuntimeError("学習反復が完了していないため、新しいモデルは作成されていません。")
        best_validation = evaluate_model(model, store, val_plan, device)
        best_eer = best_validation["eer"]
        best_step = step
        save_checkpoint(run_dir / "best.pt", checkpoint_payload(model, optimizer, step, config, manifest_sha, pretrained_sha, best_eer))
    save_checkpoint(run_dir / "last.pt", checkpoint_payload(model, optimizer, step, config, manifest_sha, pretrained_sha, best_eer))
    # テスト集合は、検証によるモデル選択が終わってから初めて評価する。
    baseline_model = TrainingEncoder().to(device)
    baseline_model.load_state_dict(checkpoint["model_state"], strict=True)
    baseline_test = evaluate_model(baseline_model, store, test_plan, device)
    selected = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model_state"], strict=True)
    selected_test = evaluate_model(model, store, test_plan, device)
    report = {
        "schema_version": 1, "status": "completed" if reason != "interrupted" else "interrupted_evaluated",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "stop_reason": reason,
        "config": config, "completed_steps": step, "selected_step": best_step,
        "baseline_validation": baseline_val, "selected_validation_eer": best_eer,
        "selected_validation": best_validation,
        "validation_eer_improved_over_baseline": best_eer < baseline_val["eer"] - 1e-8,
        "selection_criterion": "追加学習済みの候補から検証 EER で選択。同点は同一・異話者の余弦類似度の分離幅で選択。",
        "baseline_test": baseline_test, "selected_test": selected_test,
        "checkpoint_path": str(run_dir / "best.pt"), "checkpoint_sha256": sha256(run_dir / "best.pt"),
        "last_checkpoint_sha256": sha256(run_dir / "last.pt"),
        "initial_checkpoint_sha256": pretrained_sha, "data_manifest_sha256": manifest_sha,
        "training_code_sha256": sha256(Path(__file__)),
        "training_speakers": len(splits["train"]), "validation_speakers": len(splits["val"]),
        "test_speakers": len(splits["test"]), "data_utterances": manifest["utterances"],
        "data_voiced_hours": manifest["voiced_hours"], "torch_version": str(torch.__version__),
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "elapsed_seconds": time.perf_counter() - started,
        "anti_cloning_evaluation": "not_performed",
        "limitations_ja": ["評価は固定された少数の登録・照合発話を使う話者認証の実験です。",
                           "日本語の音声クローンに対する防御成功率は測定していません。",
                           "話者認証 EER の改善は、保護用摂動の転移性や防御効果を保証しません。"],
    }
    write_json(run_dir / "report.json", report)
    japanese_report = (
        "# 日本語話者エンコーダーの追加学習結果\n\n"
        f"実行済み反復数: {step}。検証で選択した反復: {best_step}。\n\n"
        f"検証 EER: 学習前 {baseline_val['eer']:.4%}、選択モデル {best_eer:.4%}。\n\n"
        f"独立テスト EER: 学習前 {baseline_test['eer']:.4%}、選択モデル {selected_test['eer']:.4%}。\n\n"
        f"学習 {len(splits['train'])} 人、検証 {len(splits['val'])} 人、テスト {len(splits['test'])} 人。話者の重複はありません。\n\n"
        f"モデルの SHA-256: `{report['checkpoint_sha256']}`\n\n"
        "日本語の音声クローンに対する防御効果は未測定です。この結果は話者認証の評価です。\n"
    )
    (run_dir / "学習結果.md").write_text(japanese_report, encoding="utf-8")
    print(f"追加学習完了。選択モデル: {run_dir / 'best.pt'}", flush=True)
    print(f"独立テスト EER: 学習前 {baseline_test['eer']:.4%}、選択モデル {selected_test['eer']:.4%}。", flush=True)
    return report


def smoke(args):
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = TrainingEncoder().to(device)
    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    started = time.perf_counter()
    original = model.linear.weight.detach().clone()
    reference_embeddings = F.normalize(torch.randn(3, 3, 8, device=device), dim=-1)
    reference_scores = []
    for speaker in range(3):
        for utterance in range(3):
            row = []
            for candidate in range(3):
                selected = [item for item in range(3) if candidate != speaker or item != utterance]
                centroid = F.normalize(reference_embeddings[candidate, selected].mean(dim=0), dim=0)
                row.append(torch.dot(reference_embeddings[speaker, utterance], centroid))
            reference_scores.append(torch.stack(row))
    reference_logits = torch.stack(reference_scores) * model.similarity_weight + model.similarity_bias
    reference_loss = F.cross_entropy(reference_logits, torch.arange(3, device=device).repeat_interleave(3))
    assert torch.allclose(ge2e_loss(model, reference_embeddings), reference_loss, atol=1e-6)
    for _ in range(args.steps):
        inputs = torch.rand(8 * 5, FRAMES, 40, device=device) * 0.02
        optimizer.zero_grad(set_to_none=True)
        loss = ge2e_loss(model, model(inputs).reshape(8, 5, -1))
        loss.backward()
        assert torch.isfinite(model.lstm.weight_ih_l0.grad).all()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    assert not torch.equal(original, model.linear.weight)
    assert abs(eer_from_scores([1, 0], [1.0, 0.0])) < 1e-9
    assert abs(eer_from_scores([1, 0], [0.0, 1.0]) - 1) < 1e-9
    assert abs(eer_from_scores([1, 0], [0.5, 0.5]) - 0.5) < 1e-9
    splits, _ = speaker_split([f"jvs{i:03d}" for i in range(1, 101)], {}, SEED)
    assert [len(splits[key]) for key in ("train", "val", "test")] == [80, 10, 10]
    result = {"status": "passed", "device": str(device), "steps": args.steps,
              "elapsed_seconds": time.perf_counter() - started,
              "checks_ja": ["自己除外重心の独立ループ照合", "GE2E の逆伝播と重み更新", "EER の完全分離・逆転・同点", "話者の固定分割"]}
    if args.report:
        write_json(Path(args.report), result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = JapaneseArgumentParser(description="日本語音声による話者エンコーダーの追加学習")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("preprocess", help="JVS ZIP を直接読み、特徴量と話者分割を保存する")
    prepare.add_argument("--archive", required=True, help="JVS の ZIP ファイル")
    prepare.add_argument("--out", required=True, help="特徴量の保存先")
    prepare.add_argument("--seed", type=int, default=SEED)
    prepare.set_defaults(func=preprocess)
    learning = commands.add_parser("train", help="既存の重みから日本語で追加学習する")
    learning.add_argument("--data", required=True, help="前処理したデータの保存先")
    learning.add_argument("--pretrained", default=str(ROOT / "models" / "encoder.pt"))
    learning.add_argument("--run-dir", required=True, help="新しい実行結果の保存先")
    learning.add_argument("--steps", type=int, default=3000)
    learning.add_argument("--eval-every", type=int, default=500)
    learning.add_argument("--log-every", type=int, default=50)
    learning.add_argument("--patience", type=int, default=5)
    learning.add_argument("--min-steps", type=int, default=1000)
    learning.add_argument("--speakers", type=int, default=8)
    learning.add_argument("--utterances", type=int, default=5)
    learning.add_argument("--lr", type=float, default=1e-5)
    learning.add_argument("--seed", type=int, default=SEED)
    learning.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    learning.set_defaults(func=train)
    check = commands.add_parser("smoke", help="逆伝播・指標・話者分割を小規模に確認する")
    check.add_argument("--pretrained", default=str(ROOT / "models" / "encoder.pt"))
    check.add_argument("--steps", type=int, default=10)
    check.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    check.add_argument("--report")
    check.set_defaults(func=smoke)
    args = parser.parse_args()
    if args.command == "train" and (args.steps < 1 or args.eval_every < 1 or args.log_every < 1 or args.speakers < 2 or args.utterances < 2 or not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("反復数・評価間隔・バッチ数・学習率を確認してください。")
    args.func(args)


if __name__ == "__main__":
    main()
