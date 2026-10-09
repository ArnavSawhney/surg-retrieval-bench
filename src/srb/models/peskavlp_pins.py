"""Pins for PeskaVLP (Yuan et al., NeurIPS 2024), shared by the main env and .venv-surgvlp.

Plain data, no imports: ``scripts/peskavlp_encode.py`` runs in the SurgVLP environment
and must not pull in the main env's transformers.

* code: CAMMA-public/SurgVLP, checked out at ``surgvlp_commit`` under third_party/.
* weights: the ``PeskaVLP`` entry of ``surgvlp._MODELS`` at that commit, an unversioned
  Seafile share, so the downloaded zip is pinned by SHA-256 instead of by URL.
* text tower: Bio_ClinicalBERT, as in ``tests/config_peskavlp.py``, pinned by revision.
"""

PESKAVLP = {
    "hf_id": "CAMMA-public/SurgVLP:PeskaVLP",  # not on HF; recorded in index meta
    "surgvlp_commit": "85e858998eec47614614b89f6af1363e0ad3f47b",
    "revision": "85e858998eec47614614b89f6af1363e0ad3f47b",  # index cache key
    "checkpoint_url": "https://seafile.unistra.fr/f/65a2b1bf113e428280d0/?dl=1",
    "checkpoint_zip_sha256": "d82e27108ae4782652e663bfcfb0832f9af97201333b81401bb5a1f2dab712f0",
    "checkpoint_path": "data/checkpoints/peskavlp/PeskaVLP.pth",
    "checkpoint_sha256": "d394be976dde11f7cf6014b903a29980099a802e7be8530906b07704c42149c8",
    "bert_id": "emilyalsentzer/Bio_ClinicalBERT",
    "bert_revision": "d5892b39a4adaed74b92212a44081509db72f87b",
    "text_max_length": 77,
    "image_size": 224,
    "embed_dim": 768,
    "default_batch_size": 32,
    "python": ".venv-surgvlp/bin/python",
}
