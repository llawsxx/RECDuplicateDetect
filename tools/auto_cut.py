#!/usr/bin/env python3
"""Automatically cut MPEG-TS recordings using recdup programme guesses."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Sequence


TS_PACKET_SIZE = 188
COPY_CHUNK_SIZE = 8 * 1024 * 1024
PARTITION_VERSION = 1


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SourceSnapshot:
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class CutSegment:
    start_byte: int
    end_byte: int
    start_seconds: float
    end_seconds: float
    kind: str


@dataclass(frozen=True)
class FolderConfig:
    name: str
    folder: Path
    output_folder: Path
    check_times: tuple[dt.time, ...]
    params: tuple[str, ...]
    delete_source: bool
    mtime_over: float
    enable: bool


@dataclass(frozen=True)
class ProcessOutcome:
    source: SourceSnapshot
    outputs: tuple[Path, ...]


class ProcessedState:
    def __init__(self, path: Path, processed: dict[str, dict[str, Any]] | None = None):
        self.path = path
        self.processed = processed or {}

    @classmethod
    def load(cls, path: Path) -> "ProcessedState":
        if not path.exists():
            return cls(path)
        with path.open("r", encoding="utf-8-sig") as stream:
            document = json.load(stream)
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != 1
            or not isinstance(document.get("processed"), dict)
        ):
            raise ConfigError(f"invalid processed state file: {path}")
        processed = document["processed"]
        if not all(isinstance(key, str) and isinstance(value, dict)
                   for key, value in processed.items()):
            raise ConfigError(f"invalid processed records in: {path}")
        return cls(path, processed)

    @staticmethod
    def key(item: FolderConfig, source: Path) -> str:
        output = os.path.normcase(str(item.output_folder.resolve()))
        input_file = os.path.normcase(str(source.resolve()))
        return f"{item.name}\n{output}\n{input_file}"

    @staticmethod
    def processing_signature(item: FolderConfig) -> str:
        return json.dumps(
            {"partition_version": PARTITION_VERSION, "params": item.params},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def contains(self, item: FolderConfig, source: Path) -> bool:
        return self.key(item, source) in self.processed

    def outcome(
        self, item: FolderConfig, source: Path
    ) -> ProcessOutcome | None:
        record = self.processed.get(self.key(item, source))
        if record is None:
            return None
        if record.get("processing_signature") != self.processing_signature(item):
            return None
        size = record.get("source_size")
        mtime_ns = record.get("source_mtime_ns")
        outputs = record.get("outputs")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or isinstance(mtime_ns, bool)
            or not isinstance(mtime_ns, int)
            or not isinstance(outputs, list)
            or not all(isinstance(path, str) for path in outputs)
        ):
            raise ConfigError(f"invalid processed record for: {source}")
        outcome = ProcessOutcome(
            SourceSnapshot(size, mtime_ns), tuple(Path(path) for path in outputs)
        )
        try:
            if snapshot(source) != outcome.source:
                return None
            if not outputs_are_complete(outcome):
                return None
        except OSError:
            return None
        return outcome

    def record(
        self, item: FolderConfig, source: Path, outcome: ProcessOutcome
    ) -> None:
        self.processed[self.key(item, source)] = {
            "config_name": item.name,
            "source": str(source.resolve()),
            "source_size": outcome.source.size,
            "source_mtime_ns": outcome.source.mtime_ns,
            "output_folder": str(item.output_folder.resolve()),
            "outputs": [str(path.resolve()) for path in outcome.outputs],
            "processing_signature": self.processing_signature(item),
            "processed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                {"schema_version": 1, "processed": self.processed},
                stream,
                ensure_ascii=False,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


def log(message: str) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be true or false")
    return value


def split_params(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ConfigError(f"{field} list entries must be strings")
        result = list(value)
    elif isinstance(value, str):
        try:
            result = shlex.split(value, posix=False)
        except ValueError as error:
            raise ConfigError(f"invalid {field}: {error}") from error
        result = [
            token[1:-1]
            if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'"
            else token
            for token in result
        ]
    else:
        raise ConfigError(f"{field} must be a string or string array")

    managed = {
        "--input",
        "--output",
        "--no-programme-inference",
        "--store",
        "--db",
        "--max-recordings",
    }
    for token in result:
        option = token.split("=", 1)[0]
        if option in managed:
            raise ConfigError(f"{field} must not contain {option}")
    return tuple(result)


def parse_check_times(value: Any, field: str) -> tuple[dt.time, ...]:
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list) and all(isinstance(part, str) for part in value):
        values = [part.strip() for part in value if part.strip()]
    else:
        raise ConfigError(f"{field} must be a comma-separated string or string array")
    if not values:
        raise ConfigError(f"{field} must contain at least one time")
    try:
        parsed = {dt.datetime.strptime(value, "%H:%M").time() for value in values}
    except ValueError as error:
        raise ConfigError(f"{field} times must use HH:MM") from error
    return tuple(sorted(parsed))


def config_path(value: Any, field: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(path: Path) -> list[FolderConfig]:
    with path.open("r", encoding="utf-8-sig") as stream:
        document = json.load(stream)
    if not isinstance(document, list):
        raise ConfigError("configuration root must be an array")

    result: list[FolderConfig] = []
    base = path.resolve().parent
    for index, item in enumerate(document):
        prefix = f"config[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{prefix} must be an object")
        name = item.get("name", f"item-{index}")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"{prefix}.name must be a non-empty string")
        mtime_over = item.get("mtime_over", 0)
        if isinstance(mtime_over, bool) or not isinstance(mtime_over, (int, float)):
            raise ConfigError(f"{prefix}.mtime_over must be a non-negative number")
        if mtime_over < 0:
            raise ConfigError(f"{prefix}.mtime_over must be a non-negative number")

        folder = config_path(item.get("folder"), f"{prefix}.folder", base)
        output_folder = config_path(
            item.get("output_folder"), f"{prefix}.output_folder", base
        )
        if folder == output_folder:
            raise ConfigError(f"{prefix} input and output folders must differ")
        result.append(
            FolderConfig(
                name=name,
                folder=folder,
                output_folder=output_folder,
                check_times=parse_check_times(
                    item.get("check_time"), f"{prefix}.check_time"
                ),
                params=split_params(item.get("params", ""), f"{prefix}.params"),
                delete_source=require_bool(
                    item.get("delete_source", False), f"{prefix}.delete_source"
                ),
                mtime_over=float(mtime_over),
                enable=require_bool(item.get("enable", True), f"{prefix}.enable"),
            )
        )
    return result


def snapshot(path: Path) -> SourceSnapshot:
    stat = path.stat()
    return SourceSnapshot(stat.st_size, stat.st_mtime_ns)


def partition_ranges(
    segments: Sequence[dict[str, Any]],
    source_size: int,
    source_duration: float,
) -> list[CutSegment]:
    if (
        isinstance(source_duration, bool)
        or not isinstance(source_duration, (int, float))
        or not math.isfinite(source_duration)
        or source_duration <= 0.0
    ):
        raise ValueError("input duration_seconds must be positive")
    candidates: list[tuple[int, int, float, float]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"programme_guesses[{index}] is not an object")
        start = segment.get("start_byte")
        end = segment.get("end_byte")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ValueError(f"programme_guesses[{index}] has invalid byte positions")
        start = max(0, min(start, source_size))
        end = max(0, min(end, source_size))
        if end <= start:
            continue
        start_time = segment.get("start_time_seconds")
        end_time = segment.get("end_time_seconds")
        if (
            isinstance(start_time, bool)
            or isinstance(end_time, bool)
            or not isinstance(start_time, (int, float))
            or not isinstance(end_time, (int, float))
            or not math.isfinite(start_time)
            or not math.isfinite(end_time)
            or end_time <= start_time
        ):
            raise ValueError(f"programme_guesses[{index}] has invalid times")
        candidates.append((start, end, float(start_time), float(end_time)))

    candidates.sort(key=lambda value: (value[0], value[1]))
    merged: list[tuple[int, int, float, float]] = []
    for start, end, start_time, end_time in candidates:
        aligned_start = min(
            source_size, start // TS_PACKET_SIZE * TS_PACKET_SIZE
        )
        aligned_end = min(
            source_size,
            ((end + TS_PACKET_SIZE - 1) // TS_PACKET_SIZE) * TS_PACKET_SIZE,
        )
        if aligned_end <= aligned_start:
            continue
        start_time = max(0.0, min(float(source_duration), start_time))
        end_time = max(start_time, min(float(source_duration), end_time))
        if merged and aligned_start <= merged[-1][1]:
            previous = merged[-1]
            merged[-1] = (
                previous[0],
                max(previous[1], aligned_end),
                min(previous[2], start_time),
                max(previous[3], end_time),
            )
        else:
            merged.append((aligned_start, aligned_end, start_time, end_time))

    ranges: list[CutSegment] = []
    cursor = 0
    cursor_time = 0.0
    for aligned_start, aligned_end, start_time, end_time in merged:
        start_time = max(cursor_time, min(float(source_duration), start_time))
        end_time = max(start_time, min(float(source_duration), end_time))
        if aligned_start > cursor:
            ranges.append(
                CutSegment(cursor, aligned_start, cursor_time, start_time, "between")
            )
        if aligned_end <= aligned_start:
            continue
        ranges.append(
            CutSegment(
                aligned_start,
                aligned_end,
                start_time,
                end_time,
                "programme",
            )
        )
        cursor = aligned_end
        cursor_time = end_time
    if cursor < source_size:
        ranges.append(
            CutSegment(
                cursor,
                source_size,
                cursor_time,
                float(source_duration),
                "between",
            )
        )
    if not ranges and source_size > 0:
        ranges.append(
            CutSegment(0, source_size, 0.0, float(source_duration), "between")
        )
    if sum(segment.end_byte - segment.start_byte for segment in ranges) != source_size:
        raise RuntimeError("partitioned byte ranges do not cover the complete source")
    return ranges


def run_recdup(
    executable: Path, source: Path, params: Sequence[str], working_directory: Path
) -> dict[str, Any]:
    command = [str(executable), "scan", "--input", str(source), *params]
    log(f"Scanning {source}")
    completed = subprocess.run(
        command,
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"recdup exited with status {completed.returncode}")
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"recdup returned invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise RuntimeError("recdup JSON root is not an object")
    return document


def copy_segment(source: Path, target: Path, segment: CutSegment) -> int:
    written = 0
    created = False
    try:
        with source.open("rb") as input_stream:
            with target.open("xb") as output_stream:
                created = True
                input_stream.seek(segment.start_byte)
                remaining = segment.end_byte - segment.start_byte
                while remaining:
                    block = input_stream.read(min(COPY_CHUNK_SIZE, remaining))
                    if not block:
                        raise IOError(f"unexpected end of file while reading {source}")
                    output_stream.write(block)
                    remaining -= len(block)
                    written += len(block)
                output_stream.flush()
                os.fsync(output_stream.fileno())
    except BaseException:
        if created:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        raise
    return written


def time_label(seconds: float, round_up: bool = False) -> str:
    value = math.ceil(seconds) if round_up else math.floor(seconds)
    hours, remainder = divmod(max(0, value), 3600)
    minutes, second = divmod(remainder, 60)
    return f"{hours:02d}-{minutes:02d}-{second:02d}"


def segment_targets(
    output_folder: Path, source: Path, segments: Sequence[CutSegment]
) -> list[Path]:
    return [
        output_folder
        / (
            f"{source.stem}.part{index:03d}_"
            f"{segment.kind}_"
            f"{time_label(segment.start_seconds)}_"
            f"{time_label(segment.end_seconds, round_up=True)}{source.suffix}"
        )
        for index, segment in enumerate(segments, start=1)
    ]


def existing_segment_outputs(output_folder: Path, source: Path) -> list[Path]:
    if not output_folder.is_dir():
        return []
    prefix = f"{source.stem}.part".casefold()
    suffix = source.suffix.casefold()
    return sorted(
        (
            path
            for path in output_folder.iterdir()
            if path.is_file()
            and path.name.casefold().startswith(prefix)
            and path.suffix.casefold() == suffix
        ),
        key=lambda path: path.name.casefold(),
    )


def outputs_are_complete(outcome: ProcessOutcome) -> bool:
    if not outcome.outputs:
        return False
    try:
        return (
            all(path.is_file() for path in outcome.outputs)
            and sum(path.stat().st_size for path in outcome.outputs)
            == outcome.source.size
        )
    except OSError:
        return False


def process_file(
    item: FolderConfig,
    source: Path,
    executable: Path,
    config_directory: Path,
) -> ProcessOutcome | None:
    existing_outputs = existing_segment_outputs(item.output_folder, source)
    if existing_outputs:
        log(f"[{item.name}] Validating existing segments: {source.name}")

    before = snapshot(source)
    age = time.time() - before.mtime_ns / 1_000_000_000
    if age < item.mtime_over:
        log(
            f"[{item.name}] Skipping active file ({age:.0f}s < "
            f"{item.mtime_over:.0f}s): {source.name}"
        )
        return None
    if before.size < TS_PACKET_SIZE:
        log(f"[{item.name}] Skipping empty/short TS file: {source.name}")
        return None

    result = run_recdup(executable, source, item.params, config_directory)
    after_scan = snapshot(source)
    if after_scan != before:
        log(f"[{item.name}] Source changed during scan; postponing: {source.name}")
        return None

    segments = result.get("programme_guesses")
    if not isinstance(segments, list):
        raise RuntimeError("recdup result has no programme_guesses array")
    input_info = result.get("input")
    if not isinstance(input_info, dict):
        raise RuntimeError("recdup result has no input object")
    ranges = partition_ranges(
        segments, before.size, input_info.get("duration_seconds")
    )
    if not ranges:
        raise RuntimeError("cannot partition an empty source")

    item.output_folder.mkdir(parents=True, exist_ok=True)
    targets = segment_targets(item.output_folder, source, ranges)
    expected = {path.resolve() for path in targets}
    actual = {path.resolve() for path in existing_outputs}
    unexpected = actual - expected
    if unexpected:
        raise RuntimeError(
            "existing segments do not match current inference: "
            + ", ".join(str(path) for path in sorted(unexpected, key=str))
        )
    for segment, target in zip(ranges, targets):
        if target.exists() and target.stat().st_size != segment.end_byte - segment.start_byte:
            raise RuntimeError(f"existing segment has the wrong size: {target}")
    created: list[Path] = []
    try:
        for segment, target in zip(ranges, targets):
            if target.exists():
                continue
            copy_segment(source, target, segment)
            created.append(target)
        after_copy = snapshot(source)
        if after_copy != before:
            raise RuntimeError("source changed during copy")
        outcome = ProcessOutcome(before, tuple(targets))
        if not outputs_are_complete(outcome):
            raise RuntimeError("output segments do not reproduce the complete source size")
        for target in targets:
            os.utime(target, ns=(before.mtime_ns, before.mtime_ns))
    except BaseException:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    for target in created:
        log(f"[{item.name}] Wrote {target} ({target.stat().st_size} bytes)")
    if not created:
        log(f"[{item.name}] Existing segment set is complete: {source.name}")
    return outcome


def delete_source_if_requested(
    item: FolderConfig, source: Path, outcome: ProcessOutcome
) -> bool:
    if not item.delete_source:
        return True
    if not outputs_are_complete(outcome):
        raise RuntimeError("source was not deleted because outputs are incomplete")
    if snapshot(source) != outcome.source:
        log(f"[{item.name}] Source changed before deletion; keeping: {source.name}")
        return False
    source.unlink()
    log(f"[{item.name}] Deleted source: {source}")
    return True


def process_folder(
    item: FolderConfig,
    executable: Path,
    config_directory: Path,
    state: ProcessedState,
) -> int:
    if not item.folder.is_dir():
        log(f"[{item.name}] Input folder does not exist: {item.folder}")
        return 0
    item.output_folder.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        (
            path
            for path in item.folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".ts"
        ),
        key=lambda path: path.name.lower(),
    )
    log(f"[{item.name}] Found {len(sources)} TS file(s) in {item.folder}")
    failures = 0
    for source in sources:
        try:
            recorded = state.outcome(item, source)
            if recorded is not None:
                if not delete_source_if_requested(item, source, recorded):
                    continue
                log(f"[{item.name}] Already processed: {source.name}")
                continue
            outcome = process_file(item, source, executable, config_directory)
            if outcome is not None:
                if not delete_source_if_requested(item, source, outcome):
                    continue
                state.record(item, source, outcome)
                state.save()
                log(f"[{item.name}] Recorded processed source: {source.name}")
        except FileExistsError:
            failures += 1
            log(f"[{item.name}] Output was created by another process: {source.name}")
        except Exception as error:
            failures += 1
            log(f"[{item.name}] Failed {source.name}: {error}")
    return failures


def latest_trigger(now: dt.datetime, times: Sequence[dt.time]) -> dt.datetime | None:
    due = [dt.datetime.combine(now.date(), value) for value in times]
    elapsed = [value for value in due if value <= now]
    return max(elapsed) if elapsed else None


def run_once(
    items: Sequence[FolderConfig],
    executable: Path,
    config_directory: Path,
    state: ProcessedState,
) -> int:
    failures = 0
    for item in items:
        if item.enable:
            failures += process_folder(item, executable, config_directory, state)
        else:
            log(f"[{item.name}] Disabled")
    return failures


def run_daemon(
    items: Sequence[FolderConfig],
    executable: Path,
    config_directory: Path,
    state: ProcessedState,
) -> None:
    last_runs: dict[int, dt.datetime] = {}
    while True:
        now = dt.datetime.now()
        for index, item in enumerate(items):
            if not item.enable:
                continue
            trigger = latest_trigger(now, item.check_times)
            if trigger is None or last_runs.get(index) == trigger:
                continue
            last_runs[index] = trigger
            log(f"[{item.name}] Check time reached: {trigger:%Y-%m-%d %H:%M}")
            try:
                failures = process_folder(item, executable, config_directory, state)
                if failures:
                    log(f"[{item.name}] Completed with {failures} failed file(s)")
            except Exception as error:
                log(f"[{item.name}] Folder check failed: {error}")
        time.sleep(30)


def default_recdup() -> Path:
    name = "recdup.exe" if os.name == "nt" else "recdup"
    return Path(__file__).resolve().parent.parent / "build" / name


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically cut MPEG-TS recordings using recdup JSON output."
    )
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("auto_folder.json"))
    )
    parser.add_argument("--recdup", default=str(default_recdup()))
    parser.add_argument(
        "--state",
        help="processed-state JSON path (default: CONFIG with .state.json suffix)",
    )
    parser.add_argument(
        "--once", action="store_true", help="scan all enabled folders once and exit"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    config_file = Path(arguments.config).resolve()
    executable = Path(arguments.recdup).resolve()
    state_file = (
        Path(arguments.state).resolve()
        if arguments.state
        else config_file.with_suffix(".state.json")
    )
    try:
        items = load_config(config_file)
        state = ProcessedState.load(state_file)
        if not executable.is_file():
            raise ConfigError(f"recdup executable does not exist: {executable}")
        if arguments.once:
            failures = run_once(items, executable, config_file.parent, state)
            if failures:
                log(f"Completed with {failures} failed file(s)")
                return 1
        else:
            log(
                f"Loaded {len(items)} folder configuration(s) from {config_file}; "
                f"processed state: {state_file}"
            )
            run_daemon(items, executable, config_file.parent, state)
    except KeyboardInterrupt:
        log("Stopped")
        return 0
    except (ConfigError, OSError, json.JSONDecodeError) as error:
        log(f"Error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
