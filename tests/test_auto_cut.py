import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "auto_cut.py"
SPEC = importlib.util.spec_from_file_location("auto_cut", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
auto_cut = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = auto_cut
SPEC.loader.exec_module(auto_cut)


def segment(start, end, start_time, end_time):
    return {
        "start_byte": start,
        "end_byte": end,
        "start_time_seconds": start_time,
        "end_time_seconds": end_time,
    }


class AutoCutTests(unittest.TestCase):
    def test_params_reject_options_managed_by_auto_cut(self):
        for option in ("--input", "--output", "--no-programme-inference"):
            with self.subTest(option=option):
                with self.assertRaises(auto_cut.ConfigError):
                    auto_cut.split_params(option, "params")

    def test_partition_ranges_covers_complete_source(self):
        ranges = auto_cut.partition_ranges(
            [segment(200, 400, 10, 20), segment(800, 1100, 40, 60)],
            1880,
            100,
        )
        self.assertEqual(
            ranges,
            [
                auto_cut.CutSegment(0, 188, 0.0, 10.0, "between"),
                auto_cut.CutSegment(188, 564, 10.0, 20.0, "programme"),
                auto_cut.CutSegment(564, 752, 20.0, 40.0, "between"),
                auto_cut.CutSegment(752, 1128, 40.0, 60.0, "programme"),
                auto_cut.CutSegment(1128, 1880, 60.0, 100.0, "between"),
            ],
        )
        self.assertEqual(
            sum(value.end_byte - value.start_byte for value in ranges), 1880
        )

    def test_partition_ranges_without_programmes_keeps_complete_source(self):
        ranges = auto_cut.partition_ranges([], 1128, 6)
        self.assertEqual(
            ranges, [auto_cut.CutSegment(0, 1128, 0.0, 6.0, "between")]
        )

    def test_partition_ranges_includes_edges_and_interval_between_programmes(self):
        ranges = auto_cut.partition_ranges(
            [segment(188, 376, 0, 1), segment(400, 600, 2, 3)],
            940,
            5,
        )
        self.assertEqual(
            ranges,
            [
                auto_cut.CutSegment(0, 188, 0.0, 0.0, "between"),
                auto_cut.CutSegment(188, 376, 0.0, 1.0, "programme"),
                auto_cut.CutSegment(376, 752, 2.0, 3.0, "programme"),
                auto_cut.CutSegment(752, 940, 3.0, 5.0, "between"),
            ],
        )

    def test_copy_segment_writes_one_final_file(self):
        packets = [bytes([index]) * auto_cut.TS_PACKET_SIZE for index in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.ts"
            target = Path(directory) / "target.ts"
            source.write_bytes(b"".join(packets))
            written = auto_cut.copy_segment(
                source,
                target,
                auto_cut.CutSegment(752, 1128, 4.0, 6.0, "programme"),
            )
            self.assertEqual(written, 2 * auto_cut.TS_PACKET_SIZE)
            self.assertEqual(target.read_bytes(), packets[4] + packets[5])

    def test_copy_segment_does_not_remove_an_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.ts"
            target = Path(directory) / "target.ts"
            source.write_bytes(b"S" * auto_cut.TS_PACKET_SIZE)
            target.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                auto_cut.copy_segment(
                    source,
                    target,
                    auto_cut.CutSegment(
                        0,
                        auto_cut.TS_PACKET_SIZE,
                        0.0,
                        1.0,
                        "programme",
                    ),
                )
            self.assertEqual(target.read_bytes(), b"existing")

    def test_process_file_streams_selected_segments_to_final_output(self):
        packets = [bytes([index]) * auto_cut.TS_PACKET_SIZE for index in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_folder = root / "input"
            output_folder = root / "output"
            input_folder.mkdir()
            source = input_folder / "recording.ts"
            source.write_bytes(b"".join(packets))
            item = auto_cut.FolderConfig(
                name="test",
                folder=input_folder,
                output_folder=output_folder,
                check_times=(),
                params=(),
                delete_source=False,
                mtime_over=0,
                enable=True,
            )
            result = {
                "input": {"duration_seconds": 8},
                "programme_guesses": [
                    segment(188, 376, 0, 1),
                    segment(940, 1316, 5, 7),
                ]
            }
            with mock.patch.object(auto_cut, "run_recdup", return_value=result):
                auto_cut.process_file(item, source, root / "recdup", root)
            first = (
                output_folder
                / "recording.part001_between_00-00-00_00-00-00.ts"
            )
            second = (
                output_folder
                / "recording.part002_programme_00-00-00_00-00-01.ts"
            )
            third = (
                output_folder / "recording.part003_between_00-00-01_00-00-05.ts"
            )
            fourth = (
                output_folder
                / "recording.part004_programme_00-00-05_00-00-07.ts"
            )
            fifth = (
                output_folder / "recording.part005_between_00-00-07_00-00-08.ts"
            )
            self.assertEqual(first.read_bytes(), packets[0])
            self.assertEqual(second.read_bytes(), packets[1])
            self.assertEqual(third.read_bytes(), packets[2] + packets[3] + packets[4])
            self.assertEqual(fourth.read_bytes(), packets[5] + packets[6])
            self.assertEqual(fifth.read_bytes(), packets[7])
            self.assertEqual(
                sum(path.stat().st_size for path in output_folder.iterdir()),
                source.stat().st_size,
            )

    def test_processed_state_survives_script_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_folder = root / "input"
            output_folder = root / "output"
            input_folder.mkdir()
            source = input_folder / "recording.ts"
            source.write_bytes(b"S" * (4 * auto_cut.TS_PACKET_SIZE))
            item = auto_cut.FolderConfig(
                name="test",
                folder=input_folder,
                output_folder=output_folder,
                check_times=(),
                params=(),
                delete_source=False,
                mtime_over=0,
                enable=True,
            )
            result = {
                "input": {"duration_seconds": 4},
                "programme_guesses": [
                    segment(0, 2 * auto_cut.TS_PACKET_SIZE, 0, 2)
                ]
            }
            state_path = root / "processed.state.json"
            first_state = auto_cut.ProcessedState.load(state_path)
            with mock.patch.object(
                auto_cut, "run_recdup", return_value=result
            ) as first_scan:
                auto_cut.process_folder(
                    item, root / "recdup", root, first_state
                )
            first_scan.assert_called_once()
            self.assertTrue(state_path.is_file())

            restarted_state = auto_cut.ProcessedState.load(state_path)
            with mock.patch.object(auto_cut, "run_recdup") as second_scan:
                auto_cut.process_folder(
                    item, root / "recdup", root, restarted_state
                )
            second_scan.assert_not_called()

    def test_processed_state_is_invalidated_when_params_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recording.ts"
            output = root / "output" / "recording.part001_between.ts"
            source.write_bytes(b"S" * auto_cut.TS_PACKET_SIZE)
            output.parent.mkdir()
            output.write_bytes(source.read_bytes())
            original = auto_cut.FolderConfig(
                name="test",
                folder=root,
                output_folder=output.parent,
                check_times=(),
                params=(),
                delete_source=False,
                mtime_over=0,
                enable=True,
            )
            state = auto_cut.ProcessedState(root / "state.json")
            state.record(
                original,
                source,
                auto_cut.ProcessOutcome(auto_cut.snapshot(source), (output,)),
            )
            changed = auto_cut.FolderConfig(
                name="test",
                folder=root,
                output_folder=output.parent,
                check_times=(),
                params=("--threshold", "0.99"),
                delete_source=False,
                mtime_over=0,
                enable=True,
            )
            self.assertIsNone(state.outcome(changed, source))

    def test_processed_state_is_invalidated_when_source_or_output_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recording.ts"
            output = root / "recording.part001_between.ts"
            source.write_bytes(b"S" * auto_cut.TS_PACKET_SIZE)
            output.write_bytes(source.read_bytes())
            item = auto_cut.FolderConfig(
                name="test",
                folder=root,
                output_folder=root / "output",
                check_times=(),
                params=(),
                delete_source=False,
                mtime_over=0,
                enable=True,
            )
            state = auto_cut.ProcessedState(root / "state.json")
            state.record(
                item,
                source,
                auto_cut.ProcessOutcome(auto_cut.snapshot(source), (output,)),
            )
            source.write_bytes(b"T" * (2 * auto_cut.TS_PACKET_SIZE))
            self.assertIsNone(state.outcome(item, source))
            source.write_bytes(b"S" * auto_cut.TS_PACKET_SIZE)
            state.record(
                item,
                source,
                auto_cut.ProcessOutcome(auto_cut.snapshot(source), (output,)),
            )
            output.unlink()
            self.assertIsNone(state.outcome(item, source))

    def test_process_file_resumes_a_missing_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_folder = root / "input"
            output_folder = root / "output"
            input_folder.mkdir()
            output_folder.mkdir()
            source = input_folder / "recording.ts"
            source.write_bytes(b"S" * (4 * auto_cut.TS_PACKET_SIZE))
            first = output_folder / "recording.part001_programme_00-00-00_00-00-02.ts"
            first.write_bytes(b"S" * (2 * auto_cut.TS_PACKET_SIZE))
            item = auto_cut.FolderConfig(
                name="test",
                folder=input_folder,
                output_folder=output_folder,
                check_times=(),
                params=(),
                delete_source=False,
                mtime_over=0,
                enable=True,
            )
            result = {
                "input": {"duration_seconds": 4},
                "programme_guesses": [segment(0, 376, 0, 2)],
            }
            with mock.patch.object(auto_cut, "run_recdup", return_value=result):
                outcome = auto_cut.process_file(
                    item, source, root / "recdup", root
                )
            self.assertIsNotNone(outcome)
            self.assertTrue(auto_cut.outputs_are_complete(outcome))
            self.assertEqual(len(outcome.outputs), 2)

    def test_delete_source_after_successful_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_folder = root / "input"
            output_folder = root / "output"
            input_folder.mkdir()
            source = input_folder / "recording.ts"
            source.write_bytes(b"S" * (4 * auto_cut.TS_PACKET_SIZE))
            item = auto_cut.FolderConfig(
                name="delete-test",
                folder=input_folder,
                output_folder=output_folder,
                check_times=(),
                params=(),
                delete_source=True,
                mtime_over=0,
                enable=True,
            )
            result = {
                "input": {"duration_seconds": 4},
                "programme_guesses": [
                    segment(0, 2 * auto_cut.TS_PACKET_SIZE, 0, 2)
                ]
            }
            state = auto_cut.ProcessedState.load(root / "processed.state.json")
            with mock.patch.object(auto_cut, "run_recdup", return_value=result):
                auto_cut.process_folder(item, root / "recdup", root, state)
            self.assertFalse(source.exists())
            self.assertTrue(
                (
                    output_folder
                    / "recording.part001_programme_00-00-00_00-00-02.ts"
                ).is_file()
            )
            self.assertTrue(state.contains(item, source))

    def test_delete_source_keeps_complete_recording_when_no_programme_is_found(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_folder = root / "input"
            input_folder.mkdir()
            source = input_folder / "recording.ts"
            source.write_bytes(b"S" * (2 * auto_cut.TS_PACKET_SIZE))
            item = auto_cut.FolderConfig(
                name="no-output-test",
                folder=input_folder,
                output_folder=root / "output",
                check_times=(),
                params=(),
                delete_source=True,
                mtime_over=0,
                enable=True,
            )
            state = auto_cut.ProcessedState.load(root / "processed.state.json")
            with mock.patch.object(
                auto_cut,
                "run_recdup",
                return_value={
                    "input": {"duration_seconds": 2},
                    "programme_guesses": [],
                },
            ):
                auto_cut.process_folder(item, root / "recdup", root, state)
            self.assertFalse(source.exists())
            self.assertEqual(
                (
                    root
                    / "output"
                    / "recording.part001_between_00-00-00_00-00-02.ts"
                ).stat().st_size,
                2 * auto_cut.TS_PACKET_SIZE,
            )
            self.assertTrue(state.contains(item, source))

    def test_delete_source_rejects_incomplete_existing_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_folder = root / "input"
            output_folder = root / "output"
            input_folder.mkdir()
            output_folder.mkdir()
            source = input_folder / "recording.ts"
            source.write_bytes(b"S" * (8 * auto_cut.TS_PACKET_SIZE))
            (output_folder / "recording.part001_programme_00-00-00_00-00-01.ts").write_bytes(
                b"old"
            )
            item = auto_cut.FolderConfig(
                name="partial-test",
                folder=input_folder,
                output_folder=output_folder,
                check_times=(),
                params=(),
                delete_source=True,
                mtime_over=0,
                enable=True,
            )
            result = {
                "input": {"duration_seconds": 8},
                "programme_guesses": [
                    segment(0, 188, 0, 1),
                    segment(564, 752, 3, 4),
                ]
            }
            state = auto_cut.ProcessedState.load(root / "processed.state.json")
            with mock.patch.object(auto_cut, "run_recdup", return_value=result):
                failures = auto_cut.process_folder(
                    item, root / "recdup", root, state
                )
            self.assertEqual(failures, 1)
            self.assertTrue(source.is_file())
            self.assertFalse(state.contains(item, source))


if __name__ == "__main__":
    unittest.main()
