from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_renderable_structure():
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()

    assert len(lines) > 150
    assert sum(line.startswith("#") for line in lines) >= 15
    assert sum(line == "```mermaid" for line in lines) >= 1
    assert sum(line == "```" for line in lines) >= 2
    assert max(map(len, lines)) < 500


def test_requirements_are_one_entry_per_line():
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    entries = [line for line in lines if line.strip() and not line.startswith("#")]

    assert len(entries) > 5
    assert all(" " not in entry.strip() for entry in entries)


def test_ci_workflow_has_multiline_structure():
    lines = (ROOT / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    ).splitlines()
    content = "\n".join(lines)

    assert len(lines) > 15
    assert "name: CI" in content
    assert "runs-on:" in content
    assert 'python-version: "3.11"' in content
    assert "python -m pytest" in content
