# iNat21 multi-species review — reviewer guide

You've been asked to label multi-organism photographs for a research benchmark.
This folder is **completely self-contained**: no big downloads, no GPU, no
cluster access. Everything runs on your laptop. Saves stream as you go, and
you can quit / resume any time.

## What you should have

When you unzip what you received, the folder looks like:

```
README.md   (this file)
review_app/
  app.py
  requirements.txt
data/
  queue_ariana.jsonl      <- Ariana's assigned queue
  queue_dan.jsonl         <- Dan's assigned queue
  review_set.jsonl        <- full fallback review set
  images/                 <- all the photos, one per image_id
  inat21_candidates.json  <- species autocomplete pool
  manifest.json
```

The `data/` folder is ~500 MB. If it's missing, ask whoever sent you the zip
to include it.

---

## One-time setup (about 5 minutes)

Use Python's built-in `venv`. You need **Python 3.10 or newer**.

Open a terminal **inside the `review_app/` folder**.

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

You only do this setup once. Each new terminal session needs the activate
command again before running the app:

**macOS / Linux:**
```bash
cd review_bundle/review_app
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
cd review_bundle\review_app
.venv\Scripts\Activate.ps1
```

---

## Each labeling session

### Launch the app

In the terminal (with the env activated), from inside `review_app/`:

**Ariana:**
```bash
python app.py --reviewer ariana --data-dir ../data
```

This automatically uses `../data/queue_ariana.jsonl`.

**Dan:**
```bash
python app.py --reviewer dan --data-dir ../data
```

This automatically uses `../data/queue_dan.jsonl`.

Use the reviewer name in lowercase so it matches the queue filename. The
terminal prints a URL like `http://127.0.0.1:7860` — **open it in your
browser**.

For a quick test, add `--limit 20`:

```bash
python app.py --reviewer ariana --limit 20 --data-dir ../data
```

If port 7860 is busy, add a different port:

```bash
python app.py --reviewer ariana --data-dir ../data --port 7861
```

### Per-image flow

For each image the app shows you:

1. **The full photo** on the left, with **numbered colored boxes** around each
   organism the AI detected.
2. **The primary species banner** — this is what iNat21 says the photo is of
   (the "main subject" the original photographer labeled).
3. **One panel per box** on the right, in the same colors:
   - The crop of just that box
   - The AI's free-text description (e.g. "insect · center · dominant subject")
   - **Role**: pick **Primary**, **Secondary**, or **Reject**
   - **Species dropdown** (active when role = Secondary): the AI's top 10
     guesses plus "Other"
   - **Look up candidates** links: open iNaturalist or Google Images for each
     prediction in a new tab, without selecting it as your answer
   - **Manual entry** (active when species = "Other"): type a species name;
     autocomplete suggests from a 10K-species list
4. **Drawing controls** under the image:
   - **Add species**: click this, then click two opposite corners on the image
     to draw a new bbox. A new crop panel appears only after the box is drawn.
   - **Redraw selected box**: pick a box number, click this, then click two
     opposite corners to replace that bbox.

### What to do

- **Look at the photo.** Are there really 2+ **identifiable** different species
  visible? If not, set the **image-level decision** at the bottom-left to
  **Reject** and hit "Save & Next". You're done with this image. This includes
  cases where another species is visible, or more than one other species may be
  visible, but you can't confidently identify the other species.

- **If yes:** pick exactly one **Primary** (the one matching the iNat label),
  one or more **Secondary** boxes (the other identifiable species), and
  **Reject** any extra boxes (e.g. the AI sometimes drew a box around two birds
  and another box around the same flock).

- **Fix boxes as needed:** if a detected bbox is too tight, too loose, or in
  the wrong place, choose its number in **Box to redraw**, click **Redraw
  selected box**, then click two opposite corners on the image. If an organism
  is missing, click **Add species** and draw its bbox; a new crop section will
  appear for that species.

- **For Secondary, pick a species:**
  - If you're not sure, use the **Look up candidates** links to compare against
    iNaturalist photos or Google Images. Clicking those links does not save or
    select anything in the app.
  - If the AI's top-10 contains the right species -> click it.
  - If you know what it is but it's not in the list → pick "Other (type
    below)", then type the scientific name in the manual entry box. Use
    autocomplete (start typing the genus or a common name).
  - If you genuinely can't tell what the other species is, set the
    **image-level decision** to **Reject** instead of saving the image as
    accepted.

- **Notes** (optional) for anything noteworthy — partial occlusion, a
  pollination interaction, anything you think helps.

- Click **Save & Next**. Your label is written immediately to
  `data/reviewed_ariana.jsonl` for Ariana or `data/reviewed_dan.jsonl` for Dan.

The app **auto-pre-selects** the Primary role on whichever box best matches
the iNat ground truth. Common case (AI got it right): one click on the
Secondary species dropdown → Save. The species dropdown itself is left empty
on purpose — you should always look at the actual photo before picking.

### Buttons

- **Save & Next** — the main action. Validates and saves.
- **Skip** — move forward without saving. Use this when you want to come back
  to a tricky image later.
- **Previous** — go back one image. If you re-save, the latest version wins
  at merge time.

### Quitting

Just close the browser tab and press `Ctrl-C` in the terminal. Your output
file has every label you submitted.

### Resuming

Re-run the same command you used before, for example:

```bash
python app.py --reviewer ariana --data-dir ../data
```

The app reads your existing labels file, marks those images done, and jumps to
the first one you haven't done.

---

## When you're done

Send back your saved labels file:

- Ariana: `data/reviewed_ariana.jsonl`
- Dan: `data/reviewed_dan.jsonl`

That single file is all the project owner needs.

You can also send the same file partway through if a checkpoint is useful.

---

## Common issues

**"Port 7860 already in use"** — your collaborator may be on the same machine.
Add `--port 7861` to your launch command.

**The app says "data/ not found"** — make sure you launched it from inside
the `review_app/` folder and included `--data-dir ../data`.

**Labeling feels slow / stuck on one image** — close the browser tab, kill
the terminal, re-launch. The browser sometimes drops the WebSocket; restart
fixes it. You won't lose any saved labels.

**I want to test before doing real work** — add `--limit 10` to only see the
first 10 records. Your test labels still save to the same file; remove or
edit `data/reviewed_ariana.jsonl` or `data/reviewed_dan.jsonl` if you want to
redo them clean.

**A species I want isn't in the autocomplete** — that's fine, just type it
free-form. Downstream tooling resolves taxonomies via GBIF.

**I want to give my collaborator access to my running instance** — add
`--share` to the launch command. Gradio prints a public URL you can send.
(Only works while your terminal is running, and your collaborator's network
must allow huggingface.co.)

---

## What the saved file looks like

Each line of `reviewed_ariana.jsonl` or `reviewed_dan.jsonl` is one labeled
image. Accepted images use `schema_version: 2`, with one primary object and a
list of secondary organisms. Example:

```json
{
  "schema_version": 2,
  "image_id": "inat21_val_19060",
  "accept": true,
  "primary_organism": {
    "organism_idx": 0,
    "bbox_px": [12, 80, 210, 330],
    "bbox_source": "detected",
    "taxonomy": {"genus": "Ardea", "species": "intermedia"},
    "taxonomy_source": "inat21_ground_truth"
  },
  "secondary_organisms": [
    {
      "organism_idx": 1,
      "bbox_px": [230, 60, 380, 300],
      "bbox_source": "edited",
      "taxonomy": {
        "kingdom": "Plantae", "phylum": "Tracheophyta",
        "class": "Magnoliopsida", "order": "Fagales",
        "family": "Fagaceae", "genus": "Quercus", "species": "alba",
        "common_name": "White oak"
      },
      "taxonomy_source": "bioclip_top_k_index_0"
    },
    {
      "organism_idx": "manual_4",
      "bbox_px": [410, 120, 480, 210],
      "bbox_source": "manual",
      "taxonomy": {"species": "Trifolium repens"},
      "taxonomy_source": "free_text"
    }
  ],
  "notes": "",
  "reviewer": "alice",
  "reviewed_at": "2026-06-03T15:42:01+00:00"
}
```

You don't need to read or edit this; it's just for the project owner.
