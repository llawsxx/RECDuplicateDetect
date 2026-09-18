# RECDuplicateDetect

`recdup` is a C++17 command-line detector for repeated television and radio
segments. FFmpeg is used through its C API for streaming decode. Video uses a
lightweight perceptual image descriptor; audio uses a lightweight spectral
fingerprint. The detector stores base vectors at fixed timestamps and aligns
matching time series to locate repeated advertisements, programme excerpts,
and long-form rebroadcast content.

## Current architecture

```text
FFmpeg packets
  -> one-second base buckets (timestamps + packet byte offsets)
  -> perceptual video / audio spectrum vectors
  -> ANN anchor search
  -> reference-time minus query-time offset clustering
  -> continuous aligned runs
  -> repeat content families
  -> programme / ad break / promo timeline inference
  -> JSON duplicate spans and programme guesses
```

Features are extracted in one-second buckets. `--min-duration` controls the
minimum duration reported for an aligned repeated segment.

## Build

The repository-local `ffmpegSHARED` SDK is discovered automatically. The
supplied FFmpeg package is 64-bit.

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

For another FFmpeg SDK use `-DFFMPEG_ROOT=D:/path/to/ffmpeg`. Windows runtime
DLLs are copied next to the executables.

## Commands

Create a portable v4 database:

```powershell
build\recdup.exe init --db catalog.recdb --max-recordings 7
```

Change an existing database and immediately trim excess recordings:

```powershell
build\recdup.exe configure --db catalog.recdb --max-recordings 7
```

Ingest an advertisement or programme using the lightweight extractors:

```powershell
build\recdup.exe ingest --db catalog.recdb --input ad.mp4 `
  --id ad-001 --name "Brand campaign 01" `
  --content-type advertisement --mode both
```

Use `--content-type programme` for known programme material. `ingest` defaults
to `unknown`; `scan` defaults to `recording`, including when `--store` is used.

Scan a long recording against the catalogue and its own history:

```powershell
build\recdup.exe scan --db catalog.recdb --input recording.ts `
  --mode both --min-duration 5 --threshold 0.90 --output result.json
```

Store the scanned recording after matching:

```powershell
build\recdup.exe scan --db catalog.recdb --input recording.ts `
  --id channel-20260918 --store --max-recordings 7 `
  --mode both --min-duration 10
```

`--max-recordings 7` persists in the database. After each stored recording,
only the seven most recently inserted or updated `recording` sources remain.
Sources marked `advertisement` or `programme` are catalogue material and are
never removed by this limit. Reusing an existing `--id` refreshes that source
as the newest recording. Use `0` for unlimited retention. Matching is performed
before the current scan is stored and before any resulting eviction, so the
current scan can still use every source that was present when it started.

Disable internal repeats with `--no-self`. `--max-gap` controls how far apart
base anchors may be; `--offset-bin` controls time-offset resolution. `--top-k`
controls candidates per anchor.
Analysis progress is written to stderr once per second with the processed media
time, percentage, speed, and ETA. Use `--no-progress` to disable it.

Decoded timestamps are normalized onto a monotonic recording timeline. Forward
jumps over 10 seconds and backward jumps over 1 second are repaired using the
decoded frame/sample duration and the other stream's progress. This compresses
operator timestamp discontinuities while preserving a real audio-only or
video-only outage when the other stream continued. Use `--timestamp-jump` to
tune the forward threshold or `--no-timestamp-repair` for the original PTS.
Every repair is listed under `input.timestamp_discontinuities` in the JSON.
Malformed packets and frames that FFmpeg reports as `AVERROR_INVALIDDATA` are
skipped and decoding continues. The input object reports the counts as
`recoverable_decode_errors` and `dropped_packets`; other FFmpeg errors remain
fatal because they usually indicate an unusable input or system failure.

Programme inference is enabled for `scan` by default. It treats strong audio
matches and audio/video-confirmed matches as short-content evidence. Video-only
repeats are reported as `visual_reuse` and do not split a programme, which keeps
reused establishing shots and studio layouts from becoming false ad breaks.
Matches against sources ingested as `advertisement` are high-confidence ad-break
evidence even when they are the only repeated item. Sources marked `programme`
are classified as programme repeats and never create an ad boundary.
Tune inference with the `--programme-*`, `--short-repeat-max`, and
`--ad-block-gap` options, or disable it with `--no-programme-inference`.

## Programme inference principle

Programme inference segments the recording timeline using repeated-content
evidence. The detector first finds repeated content, then treats the remaining
long continuous intervals as probable programme time.

1. Each decoded recording is divided into one-second buckets. Every bucket
   contains timestamps, packet byte ranges, and independently computed audio
   and video base vectors.
2. Approximate-nearest-neighbor search finds candidate vector matches. The
   candidates are aligned by `reference_time - query_time`; only anchors with
   a stable offset and a continuous run longer than `--min-duration` become a
   duplicate span.
3. Related spans are merged into a `content_family`. A family can contain
   several airings of the same advertisement, a repeated programme excerpt,
   or a video-only reused shot.
4. Short repeated families are grouped when their recording intervals are no
   more than `--ad-block-gap` apart. Multiple adjacent families form an
   `ad_break`; one isolated unknown family is currently labelled `promo`.
5. The complement of `ad_break` and `promo` intervals is labelled
   `programme` when it is at least `--programme-min` seconds long. Shorter
   complement intervals are labelled `unknown`.

The inference evidence is deliberately stricter than the general matching
threshold. By default, an audio family or a video family must reach `0.985`.
An aligned audio/video pair can confirm one another at `0.965`, because the
default confirmation margin is `0.02`. Configure these values with
`--programme-audio-threshold`, `--programme-video-threshold`, and
`--programme-confirm-margin`.

Content-family rules:

| Evidence | Family classification | Splits the programme timeline? |
| --- | --- | --- |
| Video-only repeated material | `visual_reuse` | No |
| Known source ingested as `programme` | `programme_repeat` | No |
| Long repeated audio/video material | `programme_repeat` | No |
| Unknown short repeated material | `short_repeat` | Isolated: `promo`; adjacent: `ad_break` |
| Known source ingested as `advertisement` | `advertisement` | Yes, as `ad_break` |

This means repeated establishing shots, studio layouts, and other visual
insertions do not split a programme when there is no matching audio evidence.
An isolated unknown audio repeat can still be labelled `promo`; for reliable
classification of one-off commercials, ingest the commercial with
`--content-type advertisement`.

`timeline` is continuous from zero to the normalized recording duration.
`programme_guesses` contains only its `programme` segments. For programme or
unknown complement segments, `evidence.preceded_by_break` means an inferred
`ad_break`/`promo` is immediately before the segment, while
`evidence.followed_by_break` means one is immediately after it. These flags
describe timeline geometry, not independent proof of a true edit point.

## Command-line parameters

All commands print errors to stderr and return a non-zero exit code on invalid
input or options. Paths may be quoted; Windows builds accept UTF-8 command-line
arguments.

### Database commands

| Command | Parameters | Purpose |
| --- | --- | --- |
| `init` | `--db FILE` | Create an empty portable v4 database. |
| `info` | `--db FILE` | Show database path, vector/source/recording counts, configured recording limit, and extractor count. |
| `configure` | `--db FILE --max-recordings COUNT` | Persist a new recording limit and immediately remove excess old recordings. |
| `ingest` | `--db FILE --input MEDIA` plus media options | Extract vectors and replace the source with the same `--id`; intended for advertisement/programme catalogue material. |
| `scan` | `--input MEDIA`, optional `--db FILE` | Scan against the database and, unless disabled, against repeats inside the recording itself. |

### Database retention option

| Option | Default | Meaning |
| --- | --- | --- |
| `--max-recordings COUNT` | `0` for a new database; stored value otherwise | Persist the maximum number of `recording` sources; `0` means unlimited. Accepted by `init`, `configure`, `ingest`, and `scan --store`. Excess recordings are removed oldest-first by insertion/update order. Advertisement and programme catalogue sources are excluded. |

### Media and decoding options

| Option | Default | Meaning |
| --- | --- | --- |
| `--name TEXT` | input filename | Human-readable source name stored in the database. |
| `--id TEXT` | generated from absolute path and file size | Stable source ID. Reusing it during ingest or `--store` replaces that source's vectors. |
| `--content-type TYPE` | `unknown` for `ingest`, `recording` for `scan` | One of `advertisement`, `programme`, `recording`, or `unknown`. Stored once per source and returned in match JSON. |
| `--mode auto\|video\|audio\|both` | `auto` | Select streams and feature kinds. `both` requires decodable audio and video. |
| `--timestamp-jump SEC` | `10` | Repair a forward PTS jump larger than this threshold. Backward jumps over the fixed 1-second tolerance are also repaired. |
| `--no-timestamp-repair` | off | Keep original discontinuous PTS values. Useful for diagnosing source timestamps, usually not recommended for matching. |
| `--no-progress` | off | Disable progress output on stderr. |

### Detection options

| Option | Default | Meaning |
| --- | --- | --- |
| `--threshold VALUE` | `0.90` | Cosine similarity threshold for vector candidate search; range `-1` to `1`. This is a recall threshold, not the stricter programme-inference evidence threshold. |
| `--min-duration SEC` | `5` | Minimum duration of an aligned duplicate span included in the result. |
| `--top-k COUNT` | `12` | Number of nearest database vectors retained per query vector. Higher values improve recall but increase alignment work. |
| `--max-gap SEC` | `2.5` | Maximum gap between consecutive query/reference anchors in one aligned run. |
| `--offset-bin SEC` | `1` | Resolution used to cluster `reference_time - query_time` offsets. Smaller values are more precise but less tolerant of timestamp jitter. |
| `--no-self` | off | Do not search for repeats within the current recording. Database matching is still performed when `--db` is supplied. |
| `--store` | off | After scanning, append/replace the current source in `--db` and atomically save the database. Requires `--db`. |

### Programme-inference options

| Option | Default | Meaning |
| --- | --- | --- |
| `--programme-min SEC` | `120` | Minimum complement interval labelled `programme`. |
| `--short-repeat-max SEC` | `180` | Maximum typical family duration considered short-content evidence. Longer families are treated as programme repeats. |
| `--ad-block-gap SEC` | `20` | Maximum gap between short repeated items before they are merged into one break. |
| `--programme-audio-threshold VALUE` | `0.985` | Minimum audio similarity accepted as independent programme-inference evidence; range `0` to `1`. |
| `--programme-video-threshold VALUE` | `0.985` | Minimum video similarity accepted as independent programme-inference evidence; range `0` to `1`. Video-only evidence remains `visual_reuse`. |
| `--programme-confirm-margin VALUE` | `0.02` | Amount subtracted from both programme thresholds when aligned audio and video confirm one another; range `0` to `1`. |
| `--no-programme-inference` | off | Omit `content_families`, `timeline`, and `programme_guesses` from the scan result. |

These programme thresholds are applied after candidate search. Keep
`--threshold` at or below the lowest similarity that programme inference needs
to see; for example, use `--threshold 0.90 --programme-audio-threshold 0.97`.

### Output options

| Option | Default | Meaning |
| --- | --- | --- |
| `--output FILE` | stdout | Write the JSON result to a file instead of stdout. Progress and summary messages still go to stderr. |

The JSON uses schema version 3. Each match reports query/reference byte
positions and normalized times. The input object also reports
`timestamp_discontinuities`; a non-empty list means FFmpeg timestamps were
repaired during decoding.

## Database and scale

The portable v4 `.recdb` file stores source metadata, content type, retention
order, and the configured recording limit once,
extractor metadata once,
and each normalized vector as signed int8 values. The in-memory index uses
eight 20-bit random-hyperplane tables with one-bit multi-probing; databases below
20,000 vectors are searched exactly, while larger databases avoid a full scan.

The on-disk file is rebuilt atomically and can be copied as one unit. Ingesting
the same `--id` replaces that source's vectors and refreshes its retention
order. When the recording limit is exceeded, vectors and source metadata for
the oldest recordings are physically removed before the file is saved. The
extractor ID and dimension prevent incompatible vector spaces from being mixed.

## JSON

`scan` produces schema version 3. Each match has `feature`, `extractor_id`,
`anchor_count`, similarity, and query/reference objects containing start/end
times, content type, and FFmpeg packet byte positions. `database_vector_count` is the count
used for matching; stored scans additionally include
`database_vector_count_after`.

The output also contains `content_families`, a continuous `timeline`, and
`programme_guesses`. Timeline labels are `programme`, `ad_break`, `promo`, and
`unknown`. Every inferred segment includes confidence, boundary uncertainty,
repeat coverage, audio/video-confirmed coverage, and byte/time boundaries.

## Accuracy and performance notes

- Audio spectrum vectors are useful for candidate matching but are not a robust
  broadcast fingerprint by themselves. For noisy radio archives, add a peak-pair
  audio fingerprint backend and use the current spectrum vectors as confirmation.
- All vector kinds are aligned independently. A result says which extractor
  produced it, rather than silently fusing incompatible spaces.
