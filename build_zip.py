"""Package a submission folder (top-level *.py + artifacts/*) into a zip, files at ZIP root.

  python build_zip.py                                   # submission/  -> dist/submission.zip   (v2, as before)
  python build_zip.py submission3 dist/submission-v3.zip
"""
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

root = Path(__file__).resolve().parent
sub = root / (sys.argv[1] if len(sys.argv) > 1 else "submission")
out = root / (sys.argv[2] if len(sys.argv) > 2 else "dist/submission.zip")
out.parent.mkdir(exist_ok=True)
files = sorted(p for p in sub.glob("*.py") if p.is_file())
files += sorted(p for p in (sub / "artifacts").glob("*") if p.is_file())
with ZipFile(out, "w", ZIP_DEFLATED) as z:
    for p in files:
        name = p.relative_to(sub).as_posix()
        e = ZipInfo(name); e.compress_type = ZIP_DEFLATED; e.external_attr = 0o100644 << 16
        z.writestr(e, p.read_bytes())
print(f"Wrote {out} ({out.stat().st_size:,} bytes): " + ", ".join(p.relative_to(sub).as_posix() for p in files))
