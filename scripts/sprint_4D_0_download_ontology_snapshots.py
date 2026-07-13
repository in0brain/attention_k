from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import ensure_dir, write_json, write_jsonl  # noqa: E402

CVE_URL = "https://github.com/CVEProject/cvelistV5/archive/refs/heads/main.zip"
ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack.json"
)
CWE_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    ensure_dir(path.parent)
    with urllib.request.urlopen(url, timeout=120) as response:
        if getattr(response, "status", 200) >= 400:
            raise RuntimeError(f"download failed for {url}: HTTP {response.status}")
        with path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def _short_text(value: str | None, limit: int = 700) -> str | None:
    if not value:
        return None
    text = " ".join(str(value).split())
    return text[:limit]


def build_cve_index(source_zip: Path, output_dir: Path) -> tuple[list[dict], dict]:
    records: list[dict] = []
    status_counts: dict[str, int] = {}
    with zipfile.ZipFile(source_zip) as archive:
        json_names = [
            name for name in archive.namelist()
            if name.endswith(".json") and "/cves/" in name and "/CVE-" in name
        ]
        if len(json_names) < 100000:
            raise RuntimeError(
                f"CVE source does not look complete: only {len(json_names)} CVE JSON files"
            )
        for name in json_names:
            with archive.open(name) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
            metadata = payload.get("cveMetadata") or {}
            cve_id = metadata.get("cveId")
            if not isinstance(cve_id, str) or not cve_id.startswith("CVE-"):
                continue
            state = str(metadata.get("state") or "UNKNOWN").upper()
            status_counts[state] = status_counts.get(state, 0) + 1
            containers = payload.get("containers") or {}
            cna = containers.get("cna") or {}
            descriptions = cna.get("descriptions") or []
            description = None
            for item in descriptions:
                if isinstance(item, dict) and item.get("value"):
                    description = item.get("value")
                    break
            products = []
            for affected in cna.get("affected") or []:
                if isinstance(affected, dict) and affected.get("product"):
                    products.append(str(affected["product"]))
            records.append(
                {
                    "family": "cve",
                    "normalized_id": cve_id.upper(),
                    "status": state,
                    "name": None,
                    "description": _short_text(description),
                    "metadata": {
                        "source_path": name,
                        "date_published": metadata.get("datePublished"),
                        "date_updated": metadata.get("dateUpdated"),
                        "products": sorted(set(products))[:8],
                    },
                }
            )
    records.sort(key=lambda row: row["normalized_id"])
    write_jsonl(records, output_dir / "cve" / "ontology_index.jsonl")
    return records, {"status_counts": dict(sorted(status_counts.items()))}


def build_attack_index(source_json: Path, output_dir: Path) -> tuple[list[dict], dict]:
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise RuntimeError("ATT&CK STIX source is missing objects list")
    records = []
    flag_counts = {"revoked": 0, "deprecated": 0, "active": 0}
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("type") != "attack-pattern":
            continue
        external_id = None
        for ref in obj.get("external_references") or []:
            if isinstance(ref, dict) and ref.get("source_name") == "mitre-attack":
                external_id = ref.get("external_id")
                break
        if not isinstance(external_id, str) or not external_id.startswith("T"):
            continue
        revoked = bool(obj.get("revoked"))
        deprecated = bool(obj.get("x_mitre_deprecated"))
        status = "revoked" if revoked else "deprecated" if deprecated else "active"
        flag_counts[status] += 1
        tactics = [
            phase.get("phase_name")
            for phase in obj.get("kill_chain_phases") or []
            if isinstance(phase, dict) and phase.get("phase_name")
        ]
        records.append(
            {
                "family": "attack",
                "normalized_id": external_id,
                "status": status,
                "name": _short_text(obj.get("name"), 180),
                "description": _short_text(obj.get("description")),
                "metadata": {
                    "stix_id": obj.get("id"),
                    "tactics": sorted(set(tactics)),
                    "revoked": revoked,
                    "deprecated": deprecated,
                },
            }
        )
    if len(records) < 500:
        raise RuntimeError(f"ATT&CK source does not look complete: only {len(records)} ids")
    records.sort(key=lambda row: row["normalized_id"])
    write_jsonl(records, output_dir / "attack" / "ontology_index.jsonl")
    return records, {"status_counts": flag_counts}


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def build_cwe_index(source_zip: Path, output_dir: Path) -> tuple[list[dict], dict]:
    with zipfile.ZipFile(source_zip) as archive:
        xml_names = [name for name in archive.namelist() if name.endswith(".xml")]
        if not xml_names:
            raise RuntimeError("CWE zip contains no XML file")
        xml_bytes = archive.read(xml_names[0])
    root = ET.fromstring(xml_bytes)
    records = []
    status_counts: dict[str, int] = {}
    for element in root.iter():
        if _strip_namespace(element.tag) != "Weakness":
            continue
        cwe_id = element.attrib.get("ID")
        if not cwe_id:
            continue
        status = str(element.attrib.get("Status") or "Unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        records.append(
            {
                "family": "cwe",
                "normalized_id": f"CWE-{int(cwe_id)}",
                "status": status,
                "name": _short_text(element.attrib.get("Name"), 220),
                "description": _short_text(element.attrib.get("Description")),
                "metadata": {
                    "abstraction": element.attrib.get("Abstraction"),
                    "structure": element.attrib.get("Structure"),
                },
            }
        )
    if len(records) < 900:
        raise RuntimeError(f"CWE source does not look complete: only {len(records)} ids")
    records.sort(key=lambda row: row["normalized_id"])
    write_jsonl(records, output_dir / "cwe" / "ontology_index.jsonl")
    return records, {"status_counts": dict(sorted(status_counts.items()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/ontology"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("outputs/logs/sprint_4D_0_h1_data_design"))
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(args.manifest_dir)

    sources = {
        "cve": {"url": CVE_URL, "path": args.output_dir / "cve" / "cvelistV5-main.zip"},
        "attack": {"url": ATTACK_URL, "path": args.output_dir / "attack" / "enterprise-attack.json"},
        "cwe": {"url": CWE_URL, "path": args.output_dir / "cwe" / "cwec_latest.xml.zip"},
    }
    for info in sources.values():
        download(info["url"], info["path"])

    cve_records, cve_extra = build_cve_index(sources["cve"]["path"], args.output_dir)
    attack_records, attack_extra = build_attack_index(sources["attack"]["path"], args.output_dir)
    cwe_records, cwe_extra = build_cwe_index(sources["cwe"]["path"], args.output_dir)

    family_records = {"cve": cve_records, "attack": attack_records, "cwe": cwe_records}
    extras = {"cve": cve_extra, "attack": attack_extra, "cwe": cwe_extra}
    manifest = {
        "snapshot_date": date.today().isoformat(),
        "families": {},
        "status_rules": {
            "cve": "RESERVED/REJECTED/PUBLISHED are existence-positive and counted separately.",
            "attack": "revoked/deprecated/active are existence-positive and counted separately.",
            "cwe": "Deprecated and non-deprecated weakness ids are existence-positive and counted separately.",
        },
        "knowledge_cutoff_bias_note": (
            "The snapshot may post-date a model knowledge cutoff; post-cutoff real ids are "
            "classified as grounded, which can only lower measured fabrication rate."
        ),
        "complete_snapshot_requirement": "script stops if a source looks like a sample rather than a complete id set",
    }
    for family, records in family_records.items():
        source_path = sources[family]["path"]
        index_path = args.output_dir / family / "ontology_index.jsonl"
        manifest["families"][family] = {
            "download_url": sources[family]["url"],
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "index_path": str(index_path),
            "index_sha256": sha256_file(index_path),
            "id_count": len(records),
            **extras[family],
        }
    write_json(manifest, args.manifest_dir / "ontology_snapshot_manifest.json")
    print(
        "wrote ontology snapshots:",
        {family: manifest["families"][family]["id_count"] for family in sorted(family_records)},
    )
    print("manifest:", args.manifest_dir / "ontology_snapshot_manifest.json")


if __name__ == "__main__":
    main()
