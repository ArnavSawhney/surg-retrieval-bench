# RQ5 tagging guide

How to tag the frames of `docs/rq5_frame_pool.csv` with `scripts/tag_rq5.py`
(pre-registration section 4.5). The first version of this guide was written on
15 October 2026 (file saved 23:38 IST) before the first pool frame was tagged (23:45
IST), but it was **committed only on 18 October 2026, after 436 frames had been
tagged**. Any change to a rule is dated and added to the change log at the bottom,
together with how earlier tags were handled.

Keep this open while tagging, and re-read the **quick reference** at the start of every
session.

---

## Quick reference

| Key | Tag | Tag it when… | Don't tag it when… |
|---|---|---|---|
| 1 | **smoke** | a haze or plume hides or greys out a noticeable part of the view (about a quarter of the image or more), or covers the area being worked on | a thin wisp at the edge, with the anatomy still clearly visible |
| 2 | **blur** | the **anatomy** is not sharp: tissue texture and edges look smeared or soft over most of the image | only a moving instrument is blurred while the tissue is sharp |
| 3 | **glare** | bright white patches wash out detail over a noticeable area, or sit on the area being worked on | small pinpoint highlights on wet tissue (normal in every frame) |
| 4 | **blood** | fresh blood is visible: pooling, flowing, smeared over tissue, or on the lens, over a noticeable area | the normal dark red-brown colour of the liver, pink tissue, or a few small spots |
| 5 | **occlusion** | an instrument, the specimen bag or gauze covers a large part of the view (about a third of the image or more), or blocks the area being worked on | instruments are in view but the anatomy behind them is still visible |
| 0 / c | **clean** | **none** of the five above apply | — (never combine clean with another tag) |
| 6 / o | **out of body** | the frame shows no inside of the abdomen: the operating room, the surgeons' hands or gloves, drapes, the trocar from outside, a black or blank screen | any frame showing the inside of the abdomen, however dark or blocked |

**Several tags can apply to one frame**, for example smoke + blur. Tag every one that
applies.

**When unsure, ask yourself:** "Does this artefact make this frame noticeably harder to
read than a normal frame?" If yes, tag it. If you still can't decide, **don't tag it**.
Being consistently conservative is better than being inconsistent.

---

## The tags in detail

### Smoke (key 1)
Surgical smoke comes from the electrocautery hook or the bipolar burning tissue. It
looks like a grey-white haze, a rising plume, or a general loss of contrast and colour.

* **Tag:** the haze covers roughly a quarter of the image or more, **or** it veils the
  area the instrument is working on, even if that area is smaller.
* **Don't tag:** a faint wisp in a corner while everything important is crisp.
* **Lens fog or condensation** looks the same as smoke in a single frame, and you
  often can't tell them apart. **Tag diffuse haze as smoke, whatever its cause.**

### Blur (key 2)
The camera moved, or the image is out of focus.

* **Tag:** the anatomy is not sharp. Check the tissue surface and the edges of the
  gallbladder and liver. If **nothing** in the frame looks sharp, it's blur.
* **Don't tag:** only the instrument tip is smeared because it was moving fast while the
  tissue behind it is sharp. That's normal motion, not an image problem.

### Glare (key 3)
Specular reflections from the light source on wet tissue or metal: pure white, burnt-out
patches with no detail inside.

* **Tag:** one large patch, or several patches together, covering a noticeable area
  (more than a few small specks), **or** any burnt-out patch on the area being worked
  on.
* **Don't tag:** small pinpoint highlights scattered over moist tissue. They're in
  nearly every frame.
* Bright reflections on a metal instrument shaft count only if they wash out a
  noticeable part of the image.

### Blood (key 4)
Fresh bleeding: bright or dark red liquid that pools, runs, smears over the field, or
splashes onto the lens (a red tint or blotch over the whole view).

* **Tag:** blood covers a noticeable area or hides the structures underneath.
* **Don't tag:**
  * **the liver itself**, which is naturally dark red-brown;
  * normal pink or red tissue;
  * a few small spots or a thin trace.
* An empty liver bed with a little oozing is not "blood" unless the oozing covers a
  noticeable area.

### Occlusion (key 5)
Something sits between the camera and the anatomy and hides it.

* **Tag:**
  * an instrument, the specimen bag or gauze covers roughly a third of the image or more;
  * an instrument is so close to the lens that it fills a large part of the view;
  * it blocks the area being worked on.
* **Don't tag:** instruments are in view (they nearly always are), but you can still
  see the anatomy around and behind them.

### Clean (key 0 or c)
**Clean means "none of the five apply."** It is never combined with another tag.

* Clean doesn't mean "a nice frame". It only means "none of these five artefacts".

### Out of body (key 6 or o)
The camera is outside the patient: the operating room, the surgeons' hands or gloves,
drapes, instruments on a table, the trocar seen from outside, or a black or blank
screen (camera covered, off, or being cleaned).

* **Press 6 / o.** The tool records the frame as **clean** (the pre-registration's
  "none" of the five) **plus a separate out-of-body flag**. The registered analysis
  counts these frames as clean, as section 4.5 defines it. A secondary analysis leaves
  them out entirely, so they distort neither "clean" nor any artefact.
* **Never tag an out-of-body frame as occlusion.** The pre-registration's occlusion is
  *instrument* occlusion **inside** the abdomen. Hands or the operating room filling
  the screen are not that.
* If an out-of-body frame is also clearly **blurred**, add blur as well. Smoke, glare
  and blood are tagged only when they're seen inside the abdomen.
* **Very dark** frames that still show the inside of the abdomen are **not** out of
  body. Tag them by the five rules as usual.

---

## Rules for every session

1. **Judge only the frame on screen.** Don't try to work out the phase, the video or
   what happened before or after.
2. **Don't look up any model result** (scores, predictions, `results/`) before or
   during tagging. Your tags must be independent of the models.
3. **Use `u` (undo) only to fix a key you pressed by mistake on the frame you just
   saved.** Don't go back to re-judge earlier frames.
4. **Take a break about every 150 frames.** Judgement drifts when you're tired. Re-read
   the quick reference when you come back.
5. **Same screen and brightness every session, if possible.** Blur and glare look
   different on a dim or small screen.
6. The tool stops by itself when the stopping rule is met: at least 50 frames for each
   of the five artefacts, or 1,000 frames in total, with at least 300 in any case.

---

## Optional expert check

If a surgeon or resident re-tags 50 frames blind (pre-registration 4.5), give them this
guide unchanged and nothing else: no model output and no tags of yours.

## Change log

* **2026-10-18, out-of-body frames.** The first version said out-of-body frames are
  "clean" unless an artefact is visible. During tagging (frames 0–435) such frames were
  tagged **occlusion** instead. The rule is clarified above: out-of-body frames get the
  new out-of-body key (clean plus a flag) and never occlusion. Earlier frames are
  handled by **one image-only review pass over all 436 already-tagged frames**, which
  records only "out of body: yes/no". For frames marked yes, occlusion is set to 0, and
  clean is set to 1 if no other artefact tag remains. The raw tags are kept unchanged
  in `data/rq5/tags.csv`, with a backup copy, and the corrections live in a separate
  file. No per-tag metric had been computed and no model output had been seen.
* **2026-10-18, correction rule for out-of-body frames.** For frames marked out of body
  in the review, **smoke, glare and blood are also set to 0**, not only occlusion; blur
  is kept, and clean is set to 1 if blur is not set. This is consistent with the "Out of
  body" section above (smoke, glare and blood are tagged only inside the abdomen) and
  **supersedes** the wording of the entry above, which mentioned only occlusion.
