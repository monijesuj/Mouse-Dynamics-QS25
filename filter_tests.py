from pathlib import Path
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Keep only legit sessions listed in public_labels.csv (is_illegal==0).")
    ap.add_argument("--labels", default="public_labels.csv")
    ap.add_argument("--test-root", default="data/test_files")
    ap.add_argument("--apply", action="store_true", help="Actually delete files (otherwise dry-run).")
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"filename", "is_illegal"}.issubset(df.columns):
        raise ValueError("public_labels.csv must have columns: filename, is_illegal")

    allowed = set(Path(str(n)).stem for n, flag in zip(df["filename"], df["is_illegal"]) if int(flag) == 0)

    test_root = Path(args.test_root)
    if not test_root.exists():
        raise SystemExit(f"{test_root} not found")

    files = [p for p in test_root.rglob("*") if p.is_file() and not p.name.startswith(".")]

    to_delete, keep = [], []
    for p in files:
        base = p.stem
        if base in allowed:
            keep.append(p)
        else:
            to_delete.append(p)

    print(f"Scanned: {len(files)} files under {test_root}")
    print(f"KEEP   : {len(keep)} (listed & legal)")
    print(f"REMOVE : {len(to_delete)} (unlisted OR illegal)")

    show = min(10, len(to_delete))
    if show:
        print("\nExamples to remove:")
        for p in to_delete[:show]:
            print("  -", p)

    if not args.apply:
        print("\nDRY-RUN: nothing deleted. Re-run with --apply to actually remove files.")
        return

    errors = 0
    for p in to_delete:
        try:
            p.unlink()
        except Exception as e:
            print(f"Failed to delete {p}: {e}")
            errors += 1

    print(f"\nDone. Deleted {len(to_delete) - errors} files, kept {len(keep)}.")

if __name__ == "__main__":
    main()