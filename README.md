# Physical AI with Cosmos 3, on one H100

## What this is

A hands-on course on NVIDIA Cosmos 3, built and sized for a single H100 80 GB. It is
adapted from the NVIDIA DLI course `S-OV-57-V1` and substantially extended: the source's
four notebooks and forty-one code cells become twenty lessons and roughly forty-two hours
of work, taking you from "never heard of Cosmos 3" to a synthetic-data pipeline you built,
ran and measured yourself. You will serve the Reasoner NIM on the same card you generate
on, write and validate inference specs, generate video under text, image, control and
audio conditioning, drive the action modes including the World Action Model, benchmark
your own GPU properly, argue an 8-GPU fine-tuning recipe down onto one card, and design an
evaluation for a model family that ships no evaluation harness. Every lesson ends with
exercises and a completion criterion, and every lesson has a reference solution.

---

## What makes it different from the source course

The source course is good material with a two-GPU deployment and a set of facts that have
since gone stale. The differences are concrete:

**One GPU instead of two.** The source sets `CUDA_VISIBLE_DEVICES=1` so the generator can
use the second card while the Reasoner NIM keeps the first. On a single-GPU box that line
selects a device that does not exist, `torch.cuda.device_count()` becomes zero, and the
failure surfaces somewhere unrelated. This course removes the escape hatch and teaches the
real single-card strategy instead — take turns between the NIM and the generator, or cap
the NIM's memory fraction — with the VRAM arithmetic for both. `validate_notebooks.py`
rejects `CUDA_VISIBLE_DEVICES=1` anywhere except one cell in Lesson 00 that is tagged
`teaching-antipattern` because showing the mistake is the point.

**The Reasoner NIM is restored to a usable form.** The source runs it inside a four-service
Compose stack (`lab`, `cosmos3-reasoner`, `nginx`, `s3_video_loader`) reachable only by
Compose service DNS, with no published host port. `docker/docker-compose.yml` here runs one
service, publishes a port, and documents the KV-cache pre-allocation behaviour that makes
an 8 GB FP8 model settle at 60–75 GB resident on an idle card.

**`policy` is corrected to `wam`.** The source's notebook 04 uses `"model_mode": "policy"`.
Upstream renamed it to `wam` (World Action Model) and that spec will fail. Three
independent confirmations are recorded in `assets/reference/COSMOS3_FACTS.md` §2.
`cosmos_course.specs.normalize_policy_mode` rewrites it, the static validator rejects it in
notebook JSON, and `scripts/smoke_test.py` tier 2 asserts against your own checkout that
`ModelMode` contains `wam` and does not contain `policy` — a standing regression check
against upstream drift.

**The gating premise is obsolete.** The source spends an entire notebook and eleven
screenshots on accepting gated licences for five repositories. All nine `Cosmos3` repos now
return `gated: false`. The only repository still requiring a licence click is
`nvidia/Cosmos-Guardrail1`, which the generator enables by default. One click, not five.

**Transfer is taught as what it is.** Not a model mode — `video2video` plus a control-hint
block, with four hints (`edge`, `blur`, `depth`, `seg`) and a fifth shipped asset family
(`wsm`) that is not a member of the enum.

**Module 4 (Systems) and Module 5 (Post-training) have no analogue in the source.** Correct
GPU timing with synchronisation and warm-up; allocated versus reserved versus
driver-reported memory; the OOM escalation ladder and which of its rungs exist when you
have one GPU; the SFT memory arithmetic showing full Nano fine-tuning needs 256–350 GB;
`vision_sft_edge` reduced to ~2B trainable parameters; LoRA and its generator-only scope;
and an entire evaluation lesson built on the fact that `cosmos_framework/evaluation/` is
marked `[planned]` and is not present in this release.

**Nothing about performance is asserted.** Edge's inference VRAM is not an NVIDIA figure
and this course refuses to quote one; it is measured in Lesson 06 and re-measured in
Lesson 14. Where a number came from an NVIDIA document it says so; where it is this
course's own extrapolation it says that too, in the same sentence.

---

## Quick start

Four commands, in this order, from the course root.

```bash
bash scripts/setup.sh
```

Builds the machine. Six stages, each idempotent and individually skippable: preflight
(~5 s), `uv` (~10 s), the Cosmos Framework clone and `uv sync` (20–40 min, ~25 GB), the
Jupyter kernel registration (~1 min), the cookbook media clone (2–10 min, ~3 GB), and an
NGC credential check (~5 s). Budget **25–55 minutes**, nearly all of it the framework
build. If a stage fails, fix it and re-run — each stage checks whether its work is already
done and says "already done" rather than redoing it. `bash scripts/setup.sh --check-only`
runs preflight alone and changes nothing.

```bash
python scripts/smoke_test.py
```

Four tiers, cheapest first: package imports and config (<1 s), GPU present with ≥20 GB and
compute capability ≥ 8.0 (~2 s), framework venv plus the `ModelMode` regression check
(~10 s), NIM health (~2 s). **Under 20 seconds.** A tier whose prerequisite is absent is
reported as SKIP, not a failure. Add `--full` for a real end-to-end generation (5–20 min,
~9 GB download).

```bash
echo $NGC_API_KEY | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Authenticates Docker against `nvcr.io` so the Reasoner NIM image can be pulled. **A few
seconds.** Needed once per machine, before Lesson 02. Keys start with `nvapi-`; get one at
<https://ngc.nvidia.com/setup/api-key>. The free NVIDIA Developer Program tier is enough.

```bash
jupyter lab
```

Then open **`00_environment_check.ipynb`** and select the **"Cosmos 3 (H100)"** kernel
(internal name `cosmos3-h100`, registered by stage 3 of `setup.sh`). If the kernel is not
in the list, see Troubleshooting below.

---

## Requirements

Every number below comes from `assets/reference/COSMOS3_FACTS.md`, which is the single
source of truth for this course and cites a primary source for each figure.

### Hardware

| | Requirement | Notes |
|---|---|---|
| GPU | 1 × NVIDIA H100 80 GB (SXM or PCIe) | Compute capability 9.0. Ampere or newer works for most of the course; Hopper is assumed for FP8 and FlashAttention-3. Minimum CC is 7.0, 8.0 for BF16, 8.9 for FP8 |
| VRAM | 80 GB | Cosmos3-Nano inference is 32 GB (NVIDIA figure). Cosmos3-Edge is not listed by NVIDIA; ~12 GB is a working estimate this course measures rather than quotes. The Reasoner NIM will settle at 60–75 GB of an idle card unless capped |
| Disk | **150 GB free**, 100 GB hard minimum | ~25 GB framework venv · ~9.1 GB Edge · ~34.9 GB Nano · ~20 GB NIM weights (FP8 nano) · ~20 GB uv cache · ~30 GB generated video |
| RAM | 64 GB recommended, 32 GB workable | Nothing here is host-memory bound, but `uv sync` and the VAE decode path are not frugal |
| Architecture | x86-64 (aarch64 also supported upstream) | glibc ≥ 2.35 |

### Software

| | Requirement |
|---|---|
| OS | Linux, glibc ≥ 2.35 (Ubuntu 22.04 or newer) |
| Driver | ≥ 580 for CUDA 13, ≥ 525 for CUDA 12.8 |
| CUDA | ≥ 12.8; **13.0 recommended** |
| Python | ≥ 3.10 (the course package and the framework both declare this floor) |
| Docker | Required for the Reasoner NIM, with the NVIDIA Container Toolkit and GPU passthrough. Lesson 02 documents a no-Docker fallback via `model_mode: "reasoner"`, so it is a warning rather than a hard stop |
| uv | ≥ 0.11.3 (installed by `setup.sh`) |
| git-lfs | Strongly recommended. Without it the cookbook clone succeeds and leaves 130-byte pointer files where the `.mp4`s should be, and every downstream failure then looks like a codec problem |
| ffmpeg | Optional, for the audio lectures and for verifying generated media |

The Cosmos Framework is version `1.2.2` from `https://github.com/NVIDIA/cosmos-framework`
— note the org, it is **not** `nvidia-cosmos/…`. There are no git tags or releases; `1.2.2`
exists only as the packaged version on `main`.

### Accounts and credentials

| | Needed for | Notes |
|---|---|---|
| **NGC API key** | The Reasoner NIM (Lessons 02–05, 19) | Required. Free NVIDIA Developer Program tier is sufficient. Set `NGC_API_KEY` and run the `docker login` above |
| **Hugging Face account + token** | One gated repository | Model weights are ungated — all nine `Cosmos3` repos return `gated: false`. You need HF only to accept the licence on **`nvidia/Cosmos-Guardrail1`** (~3.6 GB), which the generator enables by default. One licence click, done once, in Lesson 01 |

The models are licensed **OpenMDW 1.1**, not the NVIDIA Open Model License that the source
course cites.

---

## Learning path

Seven modules, numbered 0 through 6. Module 0 is a prerequisite for everything. After it,
**Module 1 (Perceive) and Module 2 (Imagine) are genuinely independent** — they share no
code and no download beyond the checkpoint itself, and can be taken in either order.
Module 3 depends on Module 2 but not on Module 1.

```
                        ┌──────────────────────┐
                        │ MODULE 0 FOUNDATIONS │
                        │  00 environment      │
                        │      ↓               │
                        │  01 mental model     │
                        └───────────┬──────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
     ┌────────▼─────────┐                       ┌─────────▼────────┐
     │ MODULE 1         │                       │ MODULE 2         │
     │ PERCEIVE         │   independent of      │ IMAGINE          │
     │                  │  ◄───────────────►    │                  │
     │  02 serving      │    each other         │  06 fundamentals │
     │      ↓           │                       │      ↓           │
     │  03 captioning   │                       │  07 conditioned  │
     │      ↓           │                       │      ↓           │
     │  04 embodied     │                       │  08 control      │
     │      ↓           │                       │      ↓           │
     │  05 spatial      │                       │  09 audio+guard  │
     └────────┬─────────┘                       └─────────┬────────┘
              │                                           │
              │                                 ┌─────────▼────────┐
              │                                 │ MODULE 3  ACT    │
              │                                 │  10 forward dyn  │
              │                                 │      ↓           │
              │                                 │  11 inverse dyn  │
              │                                 │      ↓           │
              │                                 │  12 wam          │
              │                                 │      ↓           │
              │                                 │  13 closed loop ⚠│
              │                                 └─────────┬────────┘
              │                                           │
              └─────────────────────┬─────────────────────┘
                                    │
                        ┌───────────▼──────────┐
                        │ MODULE 4 SYSTEMS     │
                        │  14 performance      │
                        │      ↓               │
                        │  15 optimisation     │
                        └───────────┬──────────┘
                                    │
                        ┌───────────▼──────────┐
                        │ MODULE 5             │
                        │ POST-TRAINING        │
                        │  16 concepts         │
                        │      ↓               │
                        │  17 one GPU        ⚠ │
                        │      ↓               │
                        │  18 evaluation       │
                        └───────────┬──────────┘
                                    │
                        ┌───────────▼──────────┐
                        │ MODULE 6 CAPSTONE    │
                        │  19 synthetic data   │
                        │     flywheel         │
                        │  needs 03 05 07 08   │
                        │        12 14 18      │
                        └──────────────────────┘

  ⚠ = least validated. See planning/05_VALIDATION_REPORT.md.
```

Module 4 must come after at least one of Modules 1–3, because it measures things those
modules produce; placing it after all three lets it compare reasoning, generation and
action costs on one axis. Module 5 depends on Module 4 because the single-GPU
post-training argument is a memory argument, and a learner who has not measured memory
will take it on faith. The one long-range dependency worth planning for is Lesson 18,
which evaluates whatever Lesson 17 produced — and which is designed to still work if
Lesson 17's fine-tune does not run, by evaluating the base model against itself across
configurations.

| Module | Lessons | Hours | Model tier needed | Peak new disk |
|---|---|---:|---|---:|
| 0 — Foundations | 00, 01 | 3 | Cosmos3-Edge | ~20 GB |
| 1 — Perceive | 02, 03, 04, 05 | 7 | Reasoner NIM (nano, FP8) | ~20 GB |
| 2 — Imagine | 06, 07, 08, 09 | 8 | Edge, then Cosmos3-Nano | ~35 GB |
| 3 — Act | 10, 11, 12, 13 | 7 | Nano; `*-Policy-DROID` for 13 | ~9–30 GB |
| 4 — Systems | 14, 15 | 5 | Edge and Nano, already local | 0 |
| 5 — Post-training | 16, 17, 18 | 7 | Edge; BridgeData2 subset 0.65 GB | ~10 GB |
| 6 — Capstone | 19 | 5 | Edge (reduced) or Nano (full), plus the NIM | 10–20 GB output |
| **Total** | **20** | **42** | | **~150 GB** |

Hours are scheduled time, not compute time, and they are authorial judgement rather than
measurement — see the validation report. Lesson 17 is the one most likely to overrun.

---

## Repository layout

```
cosmos3-h100-course/
├── 00_environment_check.ipynb …… 19_capstone_synthetic_data_flywheel.ipynb
│                              the twenty lessons, in order, at the root
├── solutions/                 twenty reference notebooks; three worked solutions per
│                              lesson, plus a full capstone reference implementation
├── lectures/                  twenty narration scripts, 4 h 22 m of audio when built
│   ├── README.md              per-lecture word counts, timings and suggested order
│   └── WINDOWS_AUDIO.md       three routes from script to MP3 on a Windows machine
├── scripts/
│   ├── setup.sh               six-stage machine build; idempotent; --check-only
│   ├── smoke_test.py          four-tier health check, cheapest first
│   ├── fetch_assets.py        cookbook clone + inventory verification against FACTS §6
│   ├── validate_notebooks.py  static gate: JSON, kernel, outputs, secrets, dead modes
│   ├── run_all.sh             executes every notebook in order, waiting for GPU idle
│   ├── cleanup.sh             reclaims disk; dry-run by default, needs --yes to delete
│   ├── make_lecture_audio.py  scripts → MP3, with --self-test and --dry-run
│   └── make_lecture_audio.ps1 the same, in PowerShell, for a bare Windows box
├── tools/
│   ├── build_00.py … build_19.py   the notebook builders (notebooks are generated)
│   ├── nbbuild.py             the builder library: cells, ids, kernelspec, footers
│   ├── syntax_check.py        ast.parse over every code cell
│   ├── nameflow_check.py      top-to-bottom simulation catching use-before-definition
│   ├── check_solutions.py     verifies every solution_ref resolves
│   ├── api_check.py           notebooks vs. the imported package: properties, cc.*, sections
│   └── AUTHORING_GUIDE.md     conventions for anyone editing the course
├── planning/
│   ├── 01_SOURCE_AUDIT.md     cell-by-cell audit of S-OV-57-V1
│   ├── 02_PROPOSED_CURRICULUM.md  objectives, prerequisites and timings per lesson
│   ├── 03_CURRICULUM_MAP.md   forward and reverse map, source cell → lesson
│   ├── 04_INSTRUCTOR_GUIDE.md discussion prompts, likely difficulties, timing advice
│   └── 05_VALIDATION_REPORT.md  what was verified, how, and what was not
├── assets/reference/
│   ├── COSMOS3_FACTS.md       every verified number, with its primary source
│   ├── CAPSTONE_RUBRIC.md     how the capstone is assessed
│   └── CAPSTONE_REPORT_TEMPLATE.md
├── cosmos_course/             the course helper package (kernel layer, no ML deps)
│   ├── config.py              tiers, modes, resolutions, environment probing
│   ├── models.py              Generator and Reasoner: subprocess and HTTP drivers
│   ├── nim.py                 NIM lifecycle, VRAM contention, discover_memory_env()
│   ├── specs.py               InferenceSpec, validate_spec, command construction
│   ├── measure.py             timing, memory probes, the measurement log
│   ├── checks.py              output verification — a run can exit 0 and write noise
│   ├── viz.py                 plots and contact sheets
│   ├── assets.py              cookbook asset registry
│   ├── finetune.py            SFT memory arithmetic, build_train_command, log parsing
│   └── pipeline.py            staged pipeline runner and ledger for the capstone
├── docker/
│   ├── docker-compose.yml     one service: the Reasoner NIM, on one GPU
│   └── .env.example           placeholders only; copy to .env and fill in
├── pyproject.toml             the course kernel layer, installable with pip install -e .
└── requirements.txt           the same list, with the reasoning for every bound
```

The notebooks are **generated** from `tools/build_NN.py`. If you are editing course
content rather than working through it, edit the builder and rebuild — see
`tools/AUTHORING_GUIDE.md`.

---

## Audio lectures

Twenty narration scripts live in `lectures/`, one per lesson, totalling 38,328 words and
**4 hours 22 minutes** at a normal 150 words per minute (about 3 h 30 m at 1.25×). They are
written to be heard rather than read — no bullet lists, no tables, no code, and numbers
spelled the way a person says them. The intent is that you listen away from the machine and
arrive at the notebook already knowing what it is for and where the traps are.

**The quickest way to listen needs nothing installed.** Open
`lectures\read-aloud\index.html` in Microsoft Edge, click a lecture, and press
**Ctrl + Shift + U**. Edge narrates the page with its neural voices. Pick a voice marked
*Natural* under *Voice options* the first time; Edge remembers the choice.

Those pages are built for narration rather than reading. They open on the first sentence of
the lecture with no preamble, so nothing is read to you before the content starts;
navigation sits after a spoken end marker; and the prose is rewritten for the ear, so the
narrator says "H one hundred", "B F sixteen" and "thirty-five gigabytes" instead of
spelling them out. Each page ends with a short recap of its key terms, and the index tracks
which lectures you have finished. `lectures\read-aloud\all.html` holds all twenty on one
page, which plays the whole course unattended.

Regenerate them with `python scripts\make_read_aloud.py` after editing a script;
`--check` reports any word the narrator would mangle and currently reports zero.

**If you want actual files** — for a plane, a phone, or anywhere a browser is
inconvenient — generate MP3s instead:

```
pip install edge-tts
python scripts\make_lecture_audio.py
```

About ten minutes and 125 MB, into `lectures\audio\`, using the same neural voices plus an
M3U playlist and an HTML player. Useful flags: `--self-test` (proves the pipeline works
before you spend time on it), `--dry-run`, `--lesson 06`, `--list-voices`, `--backend
kokoro` (fully offline, no network). With nothing installed at all,
`scripts\make_lecture_audio.ps1` uses the Windows built-in voices — dated, but no
dependencies.

Full walkthrough, with three routes and their tradeoffs stated plainly:
[`lectures/WINDOWS_AUDIO.md`](lectures/WINDOWS_AUDIO.md).

---

## Two run modes

Every notebook reads two environment variables in its setup cell.

**`COSMOS_COURSE_MODE`** is `reduced` by default. Reduced mode uses short clips, few
denoising steps and the smallest resolution tier, so each cell finishes in minutes and
still exercises every code path. `full` uses the settings you would use for real output and
costs hours. The capstone makes the difference concrete: reduced is Cosmos3-Edge at the
`256` tier, 61 frames, 10 steps, 3 sources × 3 variants, with a 90-minute budget; full is
Cosmos3-Nano at `480`, 121 frames, 35 steps, 6 sources × 4 variants and no budget at all.

Start reduced. Always. Opt into `full` when you have a specific reason and an afternoon.

**`COSMOS_COURSE_MODEL`** is `Cosmos3-Edge` by default and may be set to `Cosmos3-Nano`.
An explicit value wins over the mode profile, so you can run reduced settings on Nano to
exercise a capability Edge does not have.

| | `Cosmos3-Edge` | `Cosmos3-Nano` |
|---|---|---|
| Download | ~9.1 GB | ~34.9 GB |
| Parameters | 4B dense | 16B total / 8B active (MoT) |
| Inference VRAM | ~12 GB (this course's measurement, not an NVIDIA figure) | 32 GB (NVIDIA figure) |
| Max resolution | 480p | 768p |
| Audio | No | Yes |
| Transfer (`video2video` + hint) | No | Yes |

The consequence is the whole reason the default is Edge: on a first pass you save ~26 GB of
download and ~20 GB of VRAM. Lessons 08 and 09 and the transfer half of the capstone
**require** Nano, because Edge supports neither transfer nor audio; the notebooks say so and
degrade gracefully when Nano is absent. Module 3 (Lessons 10–13) is written against Nano as
well — the action modes are not a documented Edge limitation, but the lessons were authored
and sized for Nano and that is the tested path. Modules 0, 4 and 5 run on Edge throughout.
`Cosmos3-Super` (64B, ~133 GB, 128 GB inference VRAM) will **not** generate on one H100 —
although the Super *reasoner* will serve at FP8 on one card, which is a distinction Lesson
15 makes carefully.

Run out of space? `bash scripts/cleanup.sh` reports what is reclaimable and deletes nothing
until you pass both a target flag and `--yes`.

---

## Validation

The static gates are fast, require no GPU, and should be run constantly:

```bash
python3 scripts/validate_notebooks.py     # JSON, kernel, outputs, secrets, dead modes
python3 tools/syntax_check.py             # ast.parse over every code cell
python3 tools/nameflow_check.py           # use-before-definition across cells
python3 tools/check_solutions.py          # every solution_ref resolves
python3 tools/api_check.py                # notebooks vs. the imported package
python3 -c "import cosmos_course"         # the helper package imports
```

`api_check.py` is the one that needs `cosmos_course` to be importable, because it
checks the notebooks against the package **as it actually is** rather than
against a list of names kept in the validator. Three things, all AST-based so
that prose inside a string is never mistaken for code:

* **A `@property` called as a method.** `RunResult.video_path` is a property, so
  `result.video_path()` evaluates to a `Path` and then calls it —
  `TypeError: 'PosixPath' object is not callable`. Generation is opt-in
  throughout this course, so `result is None` short-circuits the expression on
  every machine that has not turned it on, and the bug reaches only the reader
  who does. The property names are introspected from the package's classes.
* **`cc.<name>` that does not exist.** Notebooks do `import cosmos_course as cc`;
  a rename in the package leaves an `AttributeError` in a notebook that still
  parses cleanly. The alias is read from each notebook's own import statement.
* **Missing lesson sections.** Every lesson must have `## Exercises`,
  `## Debugging lab` and `## Assessment`, each *opening its own markdown cell*.
  A heading appended to the end of a previous cell renders as a section arriving
  with no heading, which is how Lesson 14's exercises once began.

Failures name the file, the cell index and the offending line. Pass a notebook
path to check just that one.

All six pass on the current tree. The expensive dynamic gate is:

```bash
REDUCED=1 ./scripts/run_all.sh
```

which executes every notebook in order with the `cosmos3-h100` kernel, one at a time,
polling `nvidia-smi` until the GPU goes idle between notebooks — because on a single H100
two notebooks running concurrently will OOM each other. `--dry-run` prints the plan.

**Be aware before you start: no notebook in this course has been executed on an H100.** It
was built in a Linux sandbox with no GPU, no CUDA, no PyTorch, no Docker and restricted
network. Everything that could be checked without those things was checked, and everything
that could not is listed, ranked by risk, in
[`planning/05_VALIDATION_REPORT.md`](planning/05_VALIDATION_REPORT.md). Read it before you
teach from this. It also contains a fill-in table for recording what you validate on the
first real run.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker pull` of the NIM returns **401 Unauthorized**, or the NIM container exits immediately | Docker is not authenticated to `nvcr.io`, or `NGC_API_KEY` is unset inside the container | `echo $NGC_API_KEY \| docker login nvcr.io -u '$oauthtoken' --password-stdin`. Then check `docker/.env` actually contains the key — Compose reads `.env` from the directory holding `docker-compose.yml`, so run `docker compose` from `docker/`, not from the repo root |
| NIM starts, then `/v1/health/ready` never becomes ready | First run is downloading weights (~20 GB FP8 nano profile). This is not a hang | `docker compose logs -f cosmos3-reasoner`. Give it 10–20 minutes on a first pull. `cosmos_course.nim.nim_status()` reports state from four independent signals rather than one |
| `docker run --gpus all` says **could not select device driver** | NVIDIA Container Toolkit not installed or the daemon was not restarted after installing it | Install `nvidia-container-toolkit`, then `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. Verify with `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi -L` — Lesson 00 does exactly this |
| **CUDA out of memory** during generation, even though the model "obviously fits" | The Reasoner NIM is resident and holding a KV-cache pool. An 8 GB FP8 model settles at 60–75 GB on an idle 80 GB card. Nothing is leaking; that is how it guarantees it can admit a long request later | Take turns: `docker compose stop cosmos3-reasoner` (or `nim.stop_nim(cfg)`), generate, then start it again. Or cap it — see `NIM_GPU_MEMORY_FRACTION` in `docker/.env.example`, and note the caveat in the validation report about the underlying variable name |
| CUDA OOM with no NIM running | Resolution, frame count and step count are multiplicative. `estimate_generation_cost()` will tell you the cost before you launch | Drop to `COSMOS_COURSE_MODE=reduced`, or lower the resolution tier first (it is quadratic), then frames, then steps. Set `PYTORCH_ALLOC_CONF=expandable_segments:True`. Lesson 14 covers fragmentation, and why `nvidia-smi` disagrees with `torch.cuda.memory_allocated()` |
| Spec rejected: **`model_mode: policy` is not valid** | Upstream renamed the mode to `wam`. `docs/inference.md`, `docs/faq.md` and the action cookbook README all still say `policy` | Use `wam`. `cosmos_course.specs.normalize_policy_mode()` rewrites it for you and `validate_spec()` accepts the old name with a warning. If you are verifying the enum yourself, fetch `raw.githubusercontent.com/.../refs/heads/main/...` — the plain `.../main/...` URL serves a stale cached copy showing `POLICY = "policy"` |
| **403 Forbidden** partway through a download, on `Cosmos-Guardrail1` | The one repository still gated. The generator enables the guardrail by default, so this hits in the middle of a multi-gigabyte pull rather than at the start | Accept the licence at <https://huggingface.co/nvidia/Cosmos-Guardrail1>, then `export HF_TOKEN=…` and re-run. Note the naming split: the cookbook points at `nvidia/Cosmos-1.0-Guardrail`, the framework uses `nvidia/Cosmos-Guardrail1`. Both exist, both are gated |
| **No space left on device**, or `setup.sh` preflight fails on disk | 100 GB is the hard floor and 150 GB is comfortable | `bash scripts/cleanup.sh` to see what is reclaimable, then `--outputs --yes` or `--all --yes`. Or move the big consumers: `export COSMOS3_REPO=/mnt/big/cosmos-framework HF_HOME=/mnt/big/hf UV_CACHE_DIR=/mnt/big/uv` |
| `torch.cuda.device_count()` is **0**, or "no CUDA device" on a machine with a working GPU | `CUDA_VISIBLE_DEVICES=1` inherited from the source course or a shell profile. On a one-GPU box that selects a device that does not exist | `unset CUDA_VISIBLE_DEVICES`. This course warns if it is set to anything at all, and `validate_notebooks.py` rejects `CUDA_VISIBLE_DEVICES=1` in notebook JSON outside the one tagged teaching cell |
| **"Cosmos 3 (H100)" kernel is missing** from the Jupyter picker | Stage 3 of `setup.sh` did not run, or you launched `jupyter lab` from an environment that cannot see the kernelspec | `bash scripts/setup.sh --skip-framework --skip-assets` re-registers it. Confirm with `jupyter kernelspec list` — you want `cosmos3-h100`. Every notebook pins that kernel name in its metadata, and the validator enforces it |
| `ImportError` from inside the framework venv — wrong `libcudart`, a stray package from `~/.local` | `LD_LIBRARY_PATH` leaking host CUDA libraries into the venv, or user site-packages shadowing venv packages | The kernelspec built by `setup.sh` clears `LD_LIBRARY_PATH` and sets `PYTHONNOUSERSITE=1`; `build_train_command()` sets both in the training environment too. If you launched a subprocess by hand, set them yourself. Check which interpreter you are actually in: the course kernel layer and `$COSMOS3_REPO/.venv` are deliberately separate environments and `torch` lives only in the latter |
| `import cosmos_course` fails, or `missing_modules()` is non-empty | The helper package was not installed | `python3 -m pip install -e .` from the course root. It is a pure-Python kernel layer with no ML dependencies — nothing in it needs a GPU |
| Cookbook `.mp4`s are ~130 bytes and every decode fails | `git-lfs` was not installed when the cookbook was cloned | `sudo apt-get install -y git-lfs && git lfs install`, then `python3 scripts/fetch_assets.py`. That script verifies the inventory against `COSMOS3_FACTS.md` §6 and names known upstream drift explicitly rather than dying halfway through a lesson |
| `vllm` and the framework refuse to coexist in one venv | Documented upstream conflict: `vllm==0.19.1` conflicts with every `cu*` dependency group | Use a second venv. Lesson 02 walks through it; Lesson 13 hits the same wall with `robosuite`/`mujoco`/`torch<2.6` for closed-loop evaluation |

---

## Credits and licence

Adapted from the NVIDIA Deep Learning Institute course **`S-OV-57-V1`**, "An Introduction
to Cosmos 3". `planning/01_SOURCE_AUDIT.md` records a cell-by-cell audit of the source and
`planning/03_CURRICULUM_MAP.md` maps every source cell to its destination here, with the
disposition (carried, corrected, restructured, excluded) and the reason. Roughly a third of
this course has no analogue in the source at all; that content is listed explicitly in
§3 of the curriculum map.

Cosmos 3 model weights are licensed **OpenMDW 1.1**. The Cosmos Framework and the Cosmos
cookbook are NVIDIA projects under their own licences. The Reasoner NIM is distributed
through NGC and requires an NGC subscription. The course material in this repository — the
notebooks, helper package, scripts and lecture scripts — is Apache-2.0.

Facts in `assets/reference/COSMOS3_FACTS.md` were verified against primary sources in
August 2026. Upstream drifts; the `policy` → `wam` rename is exactly the kind of change
that will happen again. Re-verify the mode enum against your own checkout before you teach
from this — `scripts/smoke_test.py` tier 2 does it for you.
