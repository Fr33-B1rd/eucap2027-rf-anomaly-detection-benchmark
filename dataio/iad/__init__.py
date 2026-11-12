"""Public IAD data-layer API."""

from dataio.iad.manifest import (
    MANIFEST_FIELDS,
    IADManifestRecord,
    audit_iad_manifest,
    build_iad_manifest,
    hash_iad_sample,
    load_manifest,
    manifest_jsonl_sha256,
    select_anomaly_pilot,
    sha256_file,
    write_manifest_audit,
    write_manifest_csv,
    write_manifest_jsonl,
)
from dataio.iad.pickle_io import (
    IADBundle,
    IADDataError,
    IADSource,
    UnsafeIADPickleError,
    canonical_modulation_from_path,
    discover_iad_sources,
    safe_load_iad_pickle,
)

__all__ = [
    "MANIFEST_FIELDS",
    "IADBundle",
    "IADDataError",
    "IADManifestRecord",
    "IADSource",
    "UnsafeIADPickleError",
    "audit_iad_manifest",
    "build_iad_manifest",
    "canonical_modulation_from_path",
    "discover_iad_sources",
    "hash_iad_sample",
    "load_manifest",
    "manifest_jsonl_sha256",
    "safe_load_iad_pickle",
    "select_anomaly_pilot",
    "sha256_file",
    "write_manifest_audit",
    "write_manifest_csv",
    "write_manifest_jsonl",
]
