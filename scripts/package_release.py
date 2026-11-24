"""Build a local, deterministic source handover with an internal hash manifest."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".py", ".json", ".txt", ".md", ".css", ".yml", ".toml", ".jpg", ".xml"}


def main():
    paths = [ROOT / name for name in ["README.md", ".gitignore", "requirements-tested.txt", "run_app.ps1", "run_app.bat"]]
    for folder in ["glassbox", "docs", ".github", ".streamlit", "scripts"]:
        paths.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and p.suffix in EXTENSIONS and "__pycache__" not in p.parts)
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(paths)}
    manifest = {"release": "GlassBox v1.0", "files": {name: sha256(data).hexdigest() for name, data in payload.items()}}
    payload["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    destination = ROOT / "release"
    destination.mkdir(exist_ok=True)
    archive = destination / "GlassBox-v1.0.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for name, data in sorted(payload.items()):
            info = ZipInfo("GlassBox/" + name, date_time=(2026, 11, 24, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, data)
    with ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
        for name, expected in manifest["files"].items():
            if sha256(bundle.read("GlassBox/" + name)).hexdigest() != expected:
                raise RuntimeError(f"Manifest mismatch: {name}")
    checksum = sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    print(f"Created {archive} ({len(manifest['files'])} files; integrity verified)")
    print(f"SHA-256: {checksum}")


if __name__ == "__main__":
    main()
