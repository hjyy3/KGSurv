"""
Download heterogeneous biomedical knowledge graphs to ref_KG/.

Already downloaded (Phase 1):
  - PrimeKG         (8.1M edges, 10 node types)
  - Hetionet v1.0   (2.25M edges, 11 node types)
  - DRKG            (5.87M edges, 13 entity types, 107 edge types)
  - OGB-biokg       (5.1M edges, 5 entity types)

New downloads (Phase 2):
  - Monarch KG      (11M edges, 15+ node types)  -- direct HTTP
  - OpenBioLink     (4.56M edges, 7 node types)  -- Zenodo
  - iBKH            (6.7M edges, 11 node types)  -- Box.com
  - CTKG            (5M edges, 7 node types)      -- GitHub
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_KG = ROOT / "ref_KG"
REF_KG.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Phase 1: Original KGs
# ---------------------------------------------------------------------------
PHASE1_DOWNLOADS = {
    "hetionet_nodes": (
        "https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv",
        REF_KG / "hetionet_nodes.tsv",
    ),
    "hetionet_json": (
        "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2",
        REF_KG / "hetionet-v1.0.json.bz2",
    ),
    "drkg": (
        "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz",
        REF_KG / "drkg.tar.gz",
    ),
}

# ---------------------------------------------------------------------------
# Phase 2: New heterogeneous KGs
# ---------------------------------------------------------------------------
PHASE2_DOWNLOADS = {
    "monarch_kg": (
        "https://data.monarchinitiative.org/monarch-kg/latest/monarch-kg.tar.gz",
        REF_KG / "monarch_kg" / "monarch-kg.tar.gz",
    ),
    "openbiolink": (
        "https://zenodo.org/records/5361324/files/HQ_DIR.zip?download=1",
        REF_KG / "openbiolink" / "HQ_DIR.zip",
    ),
    "ctkg": (
        "https://github.com/ninglab/CTKG/raw/main/rawdata/ctkg.zip",
        REF_KG / "ctkg" / "ctkg.zip",
    ),
}

# iBKH uses Box.com -- separate handling
IBKH_DOWNLOADS = {
    "ibkh_entity": (
        "https://wcm.box.com/shared/static/602divrbjocjqubvveyqubkhqvc4wk5j.zip",
        REF_KG / "ibkh" / "entity.zip",
    ),
    "ibkh_relation": (
        "https://wcm.box.com/shared/static/s3aulnz3naa47qmzfqm63xreidiz8p29.zip",
        REF_KG / "ibkh" / "relation.zip",
    ),
}


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Print download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r    {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)")
        sys.stdout.flush()
    else:
        mb = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r    {mb:.1f} MB downloaded")
        sys.stdout.flush()


def download(name: str, url: str, out: Path, force: bool = False) -> bool:
    """Download a file. Returns True if download succeeded or file exists."""
    if out.exists() and not force:
        print(f"  [{name}] already exists, skip")
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [{name}] downloading ...")
    print(f"    URL: {url}")
    try:
        urllib.request.urlretrieve(url, out, reporthook=_progress_hook)
        print()  # newline after progress
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"  [{name}] saved to {out} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n  [{name}] FAILED: {e}")
        if out.exists():
            out.unlink()
        return False


def extract_tar_gz(archive: Path, dest: Path, name: str) -> None:
    """Extract a .tar.gz archive."""
    import tarfile
    if not archive.exists():
        return
    print(f"  [{name}] extracting tar.gz ...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(path=dest)
    print(f"  [{name}] extracted to {dest}")


def extract_zip(archive: Path, dest: Path, name: str) -> None:
    """Extract a .zip archive."""
    if not archive.exists():
        return
    print(f"  [{name}] extracting zip ...")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(path=dest)
    print(f"  [{name}] extracted to {dest}")


# ---------------------------------------------------------------------------
# Phase 1 functions
# ---------------------------------------------------------------------------
def download_phase1(force: bool = False) -> None:
    """Download original KGs (Hetionet, DRKG)."""
    print("\n=== Phase 1: Original KGs ===")
    for name, (url, out) in PHASE1_DOWNLOADS.items():
        download(name, url, out, force=force)

    # decompress hetionet json.bz2
    bz2_path = REF_KG / "hetionet-v1.0.json.bz2"
    json_path = REF_KG / "hetionet-v1.0.json"
    if bz2_path.exists() and not json_path.exists():
        import bz2
        print("  [hetionet_json] decompressing ...")
        with bz2.open(bz2_path, "rb") as f_in, open(json_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"  [hetionet_json] decompressed to {json_path}")

    # extract drkg tar.gz
    tar_path = REF_KG / "drkg.tar.gz"
    tsv_path = REF_KG / "drkg.tsv"
    if tar_path.exists() and not tsv_path.exists():
        extract_tar_gz(tar_path, REF_KG, "drkg")

    # OGB-biokg
    download_ogb_biokg()


def download_ogb_biokg() -> None:
    """Download OGB-biokg via ogb library."""
    out = REF_KG / "ogb_biokg"
    if out.exists():
        print("  [ogb-biokg] already exists, skip")
        return
    try:
        from ogb.linkproppred import LinkPropPredDataset
        print("  [ogb-biokg] downloading via ogb ...")
        dataset = LinkPropPredDataset(name="ogbl-biokg", root=str(out))
        print(f"  [ogb-biokg] saved to {out}")
    except ImportError:
        print("  [ogb-biokg] ogb not installed, skipping (pip install ogb)")


# ---------------------------------------------------------------------------
# Phase 2 functions
# ---------------------------------------------------------------------------
def download_monarch(force: bool = False) -> None:
    """Download Monarch KG (1M nodes, 11M edges, 15+ node types)."""
    dest = REF_KG / "monarch_kg"
    dest.mkdir(parents=True, exist_ok=True)
    nodes_tsv = dest / "monarch-kg_nodes.tsv"
    edges_tsv = dest / "monarch-kg_edges.tsv"

    if nodes_tsv.exists() and edges_tsv.exists() and not force:
        print("  [monarch_kg] already extracted, skip")
        return

    url, out = PHASE2_DOWNLOADS["monarch_kg"]
    ok = download("monarch_kg", url, out, force=force)
    if ok and out.exists():
        extract_tar_gz(out, dest, "monarch_kg")
        # list extracted files
        for f in sorted(dest.iterdir()):
            if f.is_file() and f.suffix in (".tsv", ".csv"):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    -> {f.name} ({size_mb:.1f} MB)")


def download_openbiolink(force: bool = False) -> None:
    """Download OpenBioLink HQ directed (180K nodes, 4.56M edges, 7 types)."""
    dest = REF_KG / "openbiolink"
    dest.mkdir(parents=True, exist_ok=True)

    # check if already extracted
    extracted_files = list(dest.glob("*.csv")) + list(dest.glob("*.tsv"))
    if extracted_files and not force:
        print("  [openbiolink] already extracted, skip")
        return

    url, out = PHASE2_DOWNLOADS["openbiolink"]
    ok = download("openbiolink", url, out, force=force)
    if ok and out.exists():
        extract_zip(out, dest, "openbiolink")
        # list extracted files
        for f in sorted(dest.rglob("*")):
            if f.is_file() and f.suffix in (".csv", ".tsv", ".txt"):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    -> {f.relative_to(dest)} ({size_mb:.1f} MB)")


def download_ibkh(force: bool = False) -> None:
    """Download iBKH (271K nodes, 6.7M edges, 11 node types, 48 relation types).

    Data hosted on Box.com. If automatic download fails,
    manual download instructions are printed.
    """
    dest = REF_KG / "ibkh"
    dest.mkdir(parents=True, exist_ok=True)

    entity_dir = dest / "Entity"
    relation_dir = dest / "Relation"

    if entity_dir.exists() and relation_dir.exists() and not force:
        n_entity = len(list(entity_dir.glob("*.csv")))
        n_relation = len(list(relation_dir.glob("*.csv")))
        if n_entity > 0 and n_relation > 0:
            print(f"  [ibkh] already extracted ({n_entity} entity + {n_relation} relation files), skip")
            return

    print("  [ibkh] Attempting Box.com download ...")
    print("  [ibkh] NOTE: Box.com may require manual download if auto-download fails")

    for dl_name, (url, out) in IBKH_DOWNLOADS.items():
        ok = download(dl_name, url, out, force=force)
        if ok and out.exists():
            # check if it's actually a zip or an HTML redirect page
            file_size = out.stat().st_size
            if file_size < 1024:
                print(f"  [{dl_name}] File too small ({file_size} bytes), likely a redirect page")
                out.unlink()
                ok = False

            if ok:
                try:
                    extract_zip(out, dest, dl_name)
                except zipfile.BadZipFile:
                    print(f"  [{dl_name}] Not a valid zip file (Box.com redirect)")
                    out.unlink()
                    ok = False

        if not ok:
            _print_ibkh_manual_instructions()
            return

    # list extracted files
    for subdir in [entity_dir, relation_dir]:
        if subdir.exists():
            for f in sorted(subdir.glob("*.csv")):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    -> {f.relative_to(dest)} ({size_mb:.1f} MB)")


def _print_ibkh_manual_instructions() -> None:
    """Print manual download instructions for iBKH."""
    dest = REF_KG / "ibkh"
    print()
    print("  ====================================================")
    print("  [ibkh] MANUAL DOWNLOAD REQUIRED")
    print("  ====================================================")
    print("  Box.com requires browser access. Please:")
    print()
    print("  1. Entity files:")
    print("     URL: https://wcm.box.com/s/602divrbjocjqubvveyqubkhqvc4wk5j")
    print(f"     Extract to: {dest / 'Entity'}")
    print()
    print("  2. Relation files:")
    print("     URL: https://wcm.box.com/s/s3aulnz3naa47qmzfqm63xreidiz8p29")
    print(f"     Extract to: {dest / 'Relation'}")
    print()
    print("  Expected entity CSV files: anatomy_vocab.csv, disease_vocab.csv,")
    print("    drug_vocab.csv, dsp_vocab.csv, gene_vocab.csv, molecule_vocab.csv,")
    print("    pathway_vocab.csv, sdsi_vocab.csv, side_effect_vocab.csv,")
    print("    symptom_vocab.csv, tc_vocab.csv")
    print()
    print("  Expected relation CSV files: 18 files like A_G_res.csv, D_D_res.csv, etc.")
    print("  ====================================================")


def download_ctkg(force: bool = False) -> None:
    """Download CTKG (1.5M nodes, 5M edges, 7 node types)."""
    dest = REF_KG / "ctkg"
    dest.mkdir(parents=True, exist_ok=True)

    # check if already extracted
    extracted = list(dest.rglob("*.csv")) + list(dest.rglob("*.tsv"))
    if extracted and not force:
        print(f"  [ctkg] already extracted ({len(extracted)} files), skip")
        return

    url, out = PHASE2_DOWNLOADS["ctkg"]
    ok = download("ctkg", url, out, force=force)
    if ok and out.exists():
        try:
            extract_zip(out, dest, "ctkg")
        except zipfile.BadZipFile:
            print("  [ctkg] zip extraction failed, trying alternative download ...")
            out.unlink()
            # try downloading full repo as zip
            alt_url = "https://github.com/ninglab/CTKG/archive/refs/heads/main.zip"
            alt_out = dest / "CTKG-main.zip"
            ok2 = download("ctkg_repo", alt_url, alt_out, force=True)
            if ok2 and alt_out.exists():
                extract_zip(alt_out, dest, "ctkg_repo")

    # list key files
    for f in sorted(dest.rglob("*")):
        if f.is_file() and f.suffix in (".csv", ".tsv", ".txt"):
            size_mb = f.stat().st_size / (1024 * 1024)
            if size_mb > 0.1:
                print(f"    -> {f.relative_to(dest)} ({size_mb:.1f} MB)")


def download_phase2(force: bool = False) -> None:
    """Download all Phase 2 KGs."""
    print("\n=== Phase 2: New Heterogeneous KGs ===")

    print("\n--- Monarch KG ---")
    download_monarch(force=force)

    print("\n--- OpenBioLink ---")
    download_openbiolink(force=force)

    print("\n--- iBKH ---")
    download_ibkh(force=force)

    print("\n--- CTKG ---")
    download_ctkg(force=force)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(force: bool = False, phase: int = 0) -> None:
    """Download KGs.

    Args:
        force: re-download even if files exist
        phase: 0=all, 1=original only, 2=new only
    """
    print(f"Knowledge Graph download script")
    print(f"Target directory: {REF_KG}")

    if phase in (0, 1):
        download_phase1(force=force)
    if phase in (0, 2):
        download_phase2(force=force)

    print("\n=== Summary ===")
    _print_summary()
    print("\nDone.")


def _print_summary() -> None:
    """Print summary of all KG files in ref_KG/."""
    kgs = {
        "PrimeKG": REF_KG / "PrimeKG.csv",
        "Hetionet": REF_KG / "hetionet-v1.0.json",
        "DRKG": REF_KG / "drkg.tsv",
        "OGB-biokg": REF_KG / "ogb_biokg",
        "Monarch KG": REF_KG / "monarch_kg",
        "OpenBioLink": REF_KG / "openbiolink",
        "iBKH": REF_KG / "ibkh",
        "CTKG": REF_KG / "ctkg",
    }
    for name, path in kgs.items():
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  [OK] {name}: {path.name} ({size_mb:.0f} MB)")
            else:
                # directory -- count files and total size
                files = list(path.rglob("*"))
                total = sum(f.stat().st_size for f in files if f.is_file())
                size_mb = total / (1024 * 1024)
                n_files = sum(1 for f in files if f.is_file())
                print(f"  [OK] {name}: {path.name}/ ({n_files} files, {size_mb:.0f} MB)")
        else:
            print(f"  [--] {name}: NOT FOUND")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download biomedical KGs")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                        help="0=all, 1=phase1 only, 2=phase2 only")
    args = parser.parse_args()
    main(force=args.force, phase=args.phase)
