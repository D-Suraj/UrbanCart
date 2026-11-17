import json
import hashlib
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "Submission" / "UrbanCart_Submission_Final.zip"
ROOT_FILES = ("README.md", "requirements.txt", ".env.example", ".gitignore",
              "demo.py", "UrbanCart_Submission.ipynb")
FOLDERS = ("urbancart", "scripts", "tests", "evaluation", "data/raw",
           "data/processed", "reports", "presentation")


def package():
    files = [ROOT / name for name in ROOT_FILES]
    for directory in FOLDERS:
        files.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and all(not part.startswith((".", "__pycache__"))
                                               for part in path.relative_to(ROOT).parts))
    notebook = json.loads((ROOT / "UrbanCart_Submission.ipynb").read_text(encoding="utf-8"))
    assert all(cell["execution_count"] is not None and
               not any(output["output_type"] == "error" for output in cell["outputs"])
               for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert not any("The model service is temporarily unavailable" in json.dumps(c.get("outputs", []))
                   for c in notebook["cells"]), "Notebook demo contains a provider failure."
    assert 'Settings.from_env(mode="openrouter")' in json.dumps(notebook).replace('\\"', '"')
    assert (ROOT / "reports/CURRENT_EVALUATION.md").is_file()
    assert (ROOT / "presentation/UrbanCart_Presentation_LLM.pptx").is_file()
    for path in files:
        if path.suffix in {".py", ".md", ".json", ".ipynb", ".csv"}:
            assert not re.search(rb"sk-or-v1-[A-Za-z0-9]{20,}", path.read_bytes()), "Credential pattern detected; packaging stopped."
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=8) as archive:
        for path in sorted(files):
            archive.write(path, Path("UrbanCart") / path.relative_to(ROOT))
    with ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        assert not any(name.endswith("/.env") or "/runtime/" in name or "/private/" in name
                       for name in archive.namelist())
        print(f"{OUTPUT}: {len(archive.namelist())} files")
    print("SHA256:", hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    return OUTPUT


if __name__ == "__main__":
    package()
