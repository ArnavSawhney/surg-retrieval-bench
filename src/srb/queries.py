"""The pre-registered query set (``docs/preregistration.md``, section 3). Frozen.

Every text here is copied verbatim into the pre-registration, and
``tests/test_queries.py`` checks that the two agree. After the ``prereg-v1`` tag,
changing a wording is a protocol deviation: log it, dated and with the reason, in the
pre-registration's Deviations section. Never edit a wording because it scored badly.

Each target has 5 rephrasings with a fixed role, the same across targets, so the RQ2
spread can also be read by rephrasing type:

* r1: technical description (the label definition in surgical terms)
* r2: the action, with the surgeon or instrument as the subject
* r3: what is visible: anatomy or instrument appearance
* r4: lay wording, without surgical jargon
* r5: short label-like phrase
"""

from __future__ import annotations

from srb.datasets.cholec80 import PHASES, TOOL_COLUMNS, TOOLS
from srb.retrieval import Query

REPHRASING_ROLES = ("technical", "action", "visual", "lay", "short")

# Text limits, in tokens including special tokens. Text encoders truncate silently, so a
# query over a limit would be scored with a different wording than the one registered.
# tests/test_queries.py tokenises every query with each tokenizer and checks the limit.
# (hf_id, revision, max tokens)
TEXT_TOKENIZERS: dict[str, tuple[str, str, int]] = {
    "clip-vit-l14": ("openai/clip-vit-large-patch14",
                     "32bd64288804d66eefd0ccbe215aa642df71cc41", 77),
    "siglip-so400m-384": ("google/siglip-so400m-patch14-384",
                          "9fdffc58afc957d1a03a25b10dba0329ab15c2a3", 64),
    "medsiglip-448": ("google/medsiglip-448", "9cea28a1a1195f665105faa6e8544c112fd960a4", 64),
    # PeskaVLP: SurgVLP's surgvlp.tokenize() uses Bio_ClinicalBERT with max_length=77
    # (CAMMA-public/SurgVLP @ 85e858998eec47614614b89f6af1363e0ad3f47b).
    "peskavlp": ("emilyalsentzer/Bio_ClinicalBERT",
                 "d5892b39a4adaed74b92212a44081509db72f87b", 77),
}

PHASE_QUERIES: dict[str, tuple[str, str, str, str, str]] = {
    "Preparation": (
        "preparation phase of a laparoscopic cholecystectomy: instruments are introduced "
        "and the gallbladder is exposed",
        "the surgeon grasps the gallbladder fundus and retracts it to expose the operative field",
        "an intact gallbladder under the liver, not yet dissected",
        "the beginning of keyhole gallbladder surgery, before any cutting",
        "surgical preparation and exposure",
    ),
    "CalotTriangleDissection": (
        "dissection of Calot's triangle in a laparoscopic cholecystectomy",
        "the surgeon dissects fat and peritoneum to expose the cystic duct and cystic artery",
        "the cystic duct and cystic artery being exposed at the neck of the gallbladder",
        "carefully separating tissue to find the tube and the blood vessel leading to the gallbladder",
        "Calot's triangle dissection",
    ),
    "ClippingCutting": (
        "clipping and cutting the cystic duct and artery",
        "surgeon applies clips before cutting",
        "clip applier on the cystic duct",
        "sealing the gallbladder's tube and blood vessel with small metal clips, then cutting them",
        "laparoscopic clipping step of a cholecystectomy",
    ),
    "GallbladderDissection": (
        "dissection of the gallbladder from the liver bed in a laparoscopic cholecystectomy",
        "the surgeon separates the gallbladder from the liver using electrocautery",
        "a gallbladder partly detached from the liver bed",
        "peeling the gallbladder away from the liver",
        "gallbladder dissection",
    ),
    "GallbladderPackaging": (
        "packaging the detached gallbladder into a specimen retrieval bag",
        "the surgeon places the gallbladder inside a specimen bag",
        "a fully detached gallbladder next to an open specimen bag",
        "putting the removed gallbladder into a plastic pouch inside the abdomen",
        "gallbladder packaging",
    ),
    "CleaningCoagulation": (
        "cleaning and coagulation of the liver bed after the gallbladder has been detached",
        "the surgeon irrigates, suctions and coagulates bleeding points on the liver bed",
        "the empty liver bed where the gallbladder used to be",
        "washing the area and stopping small bleeds after the gallbladder is out",
        "cleaning and coagulation",
    ),
    "GallbladderRetraction": (
        "extraction of the gallbladder in its specimen bag through the trocar port",
        "the surgeon pulls the specimen bag with the gallbladder out of the abdomen",
        "a specimen bag being drawn towards the port site",
        "taking the bagged gallbladder out of the body",
        "gallbladder retraction",
    ),
}

TOOL_QUERIES: dict[str, tuple[str, str, str, str, str]] = {
    "Grasper": (
        "a laparoscopic grasper in the image",
        "a grasper holding tissue",
        "atraumatic forceps with small jaws at the tip of a long shaft",
        "a long thin tool with small jaws that holds tissue",
        "grasper",
    ),
    "Bipolar": (
        "bipolar coagulation forceps in the laparoscopic image",
        "bipolar forceps coagulating a bleeding vessel",
        "forceps with two fine insulated tips for electrosurgical coagulation",
        "a tool that stops bleeding by sealing tissue with electric current",
        "bipolar",
    ),
    "Hook": (
        "a hook electrode in the image",
        "an electrocautery hook dissecting tissue",
        "an L-shaped metal hook at the tip of a laparoscopic instrument",
        "a hooked tool that burns through tissue",
        "hook",
    ),
    "Scissors": (
        "laparoscopic scissors in the image",
        "scissors cutting tissue",
        "curved scissor blades at the tip of a laparoscopic instrument",
        "a long pair of surgical scissors inside the abdomen",
        "scissors",
    ),
    "Clipper": (
        "a clip applier in the laparoscopic image",
        "a clip applier placing a metal clip",
        "an instrument holding a metal clip between its jaws",
        "a tool that closes off tubes with small metal clips",
        "clipper",
    ),
    "Irrigator": (
        "a suction-irrigation cannula in the laparoscopic image",
        "an irrigator rinsing the operative field with fluid",
        "a straight hollow tube suctioning fluid and blood",
        "a tube that washes the area with water and sucks it away",
        "irrigator",
    ),
    "SpecimenBag": (
        "a specimen retrieval bag in the laparoscopic image",
        "a specimen bag being opened inside the abdomen",
        "a plastic retrieval pouch",
        "a plastic bag used to remove tissue from the body",
        "specimen bag",
    ),
}

# Endoscapes2023 CVS criteria (Strasberg's critical view of safety). Evaluated in Week 8.
CVS_QUERIES: dict[str, tuple[str, str, str, str, str]] = {
    "C1": (
        "two tubular structures entering the gallbladder",
        "the surgeon has exposed only two structures, the cystic duct and the cystic artery, "
        "entering the gallbladder",
        "the cystic duct and cystic artery clearly seen connecting to the gallbladder",
        "two thin tubes clearly visible going into the gallbladder",
        "two structures connected to the gallbladder",
    ),
    "C2": (
        "hepatocystic triangle cleared of fat",
        "the surgeon has cleared all fat and fibrous tissue from the hepatocystic triangle",
        "a clean, fat-free triangle between the cystic duct, the liver and the common hepatic duct",
        "the area at the neck of the gallbladder fully cleaned of fat",
        "cleared hepatocystic triangle",
    ),
    "C3": (
        "lower gallbladder separated from the liver bed",
        "the surgeon has separated the lower third of the gallbladder from the liver to expose "
        "the cystic plate",
        "the cystic plate visible beneath the lifted lower part of the gallbladder",
        "the bottom of the gallbladder lifted off the liver",
        "cystic plate exposed",
    ),
}

# R = 0 by construction: excluded from mAP, used only for the score-calibration check.
NEGATIVE_QUERIES: tuple[str, ...] = (
    "a car on a road",
    "a dog playing in a park",
    "a plate of food on a table",
    "a city skyline at night",
    "a page of printed text",
    "a chest X-ray",
    "an MRI scan of the brain",
)


def _slug(name: str) -> str:
    return name.lower()


def cholec80_queries() -> list[Query]:
    """All Cholec80 queries (35 phase + 35 tool + negatives), in a fixed order."""
    out: list[Query] = []
    for phase in PHASES:
        for r, text in enumerate(PHASE_QUERIES[phase], 1):
            out.append(Query(f"phase_{_slug(phase)}_r{r}", text, "phase", phase))
    for tool, col in zip(TOOLS, TOOL_COLUMNS):
        for r, text in enumerate(TOOL_QUERIES[tool], 1):
            out.append(Query(f"tool_{_slug(tool)}_r{r}", text, "tool", col))
    for i, text in enumerate(NEGATIVE_QUERIES, 1):
        out.append(Query(f"neg_{i}", text, "negative", None))
    return out
