#!/usr/bin/env python3
"""Sync live state snapshot for DermaFuzzy-XAI.

Generates LIVE_STATE_SNAPSHOT.md capturing:
1. Git branch, status, and commit log.
2. Clean repository file tree.
3. derma_config.yaml configuration.
4. Spec-Kit task progress across specs/.
5. AST class/function signatures for packages/dermafuzzy_core/*.py.
6. Live pytest -q execution output.

Copies output to Windows clipboard via clip.exe (utf-16le).
"""

from __future__ import annotations

import ast
import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

SNAPSHOT_FILENAME: Final[str] = "LIVE_STATE_SNAPSHOT.md"
CLIPBOARD_ENCODING: Final[str] = "utf-16le"
NUM_RECENT_COMMITS: Final[int] = 5
DEFAULT_IGNORE_DIRS: Final[frozenset[str]] = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".venv",
        ".agents",
        "dist",
        "build",
        ".gemini",
    }
)


def run_cmd(args: list[str], cwd: Path | None = None) -> str:
    """Run a shell command and return trimmed stdout, or error message."""
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        if proc.returncode != 0 and err:
            return f"{out}\n{err}".strip()
        return out if out else err
    except Exception as e:
        return f"Error executing {args}: {e}"


def generate_file_tree(dir_path: Path, prefix: str = "", ignore_dirs: frozenset[str] | set[str] | None = None) -> list[str]:
    """Generate a clean visual ASCII file tree ignoring transient directories."""
    active_ignore = DEFAULT_IGNORE_DIRS if ignore_dirs is None else ignore_dirs
    lines: list[str] = []
    items = sorted(
        [
            p
            for p in dir_path.iterdir()
            if p.name not in active_ignore
            and not p.name.endswith(".egg-info")
            and not p.name.endswith(".pyc")
        ],
        key=lambda p: (not p.is_dir(), p.name.lower()),
    )
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}")
        if item.is_dir():
            sub_prefix = f"{prefix}{'    ' if is_last else '│   '}"
            lines.extend(generate_file_tree(item, sub_prefix, ignore_dirs))
    return lines


def parse_speckit_tasks(specs_dir: Path) -> str:
    """Parse task checklists across all specs/*/tasks.md."""
    if not specs_dir.is_dir():
        return "No specs directory found."

    sections: list[str] = []
    for spec_dir in sorted(specs_dir.iterdir()):
        if not spec_dir.is_dir():
            continue
        tasks_file = spec_dir / "tasks.md"
        if not tasks_file.is_file():
            continue

        lines = tasks_file.read_text(encoding="utf-8").splitlines()
        tasks = [line.strip() for line in lines if line.strip().startswith("- [")]
        completed = [t for t in tasks if t.startswith("- [x]") or t.startswith("- [X]")]
        total = len(tasks)
        pct = (len(completed) / total * 100.0) if total > 0 else 0.0

        section = [
            f"### Feature: `{spec_dir.name}`",
            f"- Progress: **{len(completed)} / {total}** tasks completed ({pct:.1f}%)",
        ]
        if len(completed) == total and total > 0:
            section.append("- Status: All tasks complete [X]")
        else:
            remaining = [t for t in tasks if not (t.startswith("- [x]") or t.startswith("- [X]"))]
            section.append(f"- Pending Tasks ({len(remaining)}):")
            for r in remaining[:8]:
                section.append(f"  {r}")
            if len(remaining) > 8:
                section.append(f"  ... and {len(remaining) - 8} more")
        sections.append("\n".join(section))
    return "\n\n".join(sections) if sections else "No tasks found in specs/."


def extract_ast_signatures(pkg_dir: Path) -> str:
    """Extract AST classes, functions, and method signatures for packages/dermafuzzy_core/*.py."""
    if not pkg_dir.is_dir():
        return "Directory packages/dermafuzzy_core not found."

    sections: list[str] = []
    py_files = sorted(pkg_dir.glob("*.py"))
    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=py_file.name)
        except Exception as e:
            sections.append(f"### `{py_file.name}`\nError parsing AST: {e}")
            continue

        file_lines: list[str] = [f"### Module: `packages/dermafuzzy_core/{py_file.name}`", "```python"]
        doc = ast.get_docstring(tree)
        if doc:
            first_line = doc.strip().splitlines()[0]
            file_lines.append(f'"""{first_line}"""\n')

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) for b in node.bases]
                base_str = f"({', '.join(bases)})" if bases else ""
                decorators = [f"@{ast.unparse(d)}" for d in node.decorator_list]
                for d in decorators:
                    file_lines.append(d)
                cls_sig = f"class {node.name}{base_str}:"
                cls_doc = ast.get_docstring(node)
                if cls_doc:
                    cls_sig += f'\n    """{cls_doc.strip().splitlines()[0]}"""'
                file_lines.append(cls_sig)

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_ret = f" -> {ast.unparse(item.returns)}" if item.returns else ""
                        m_sig = f"    def {item.name}({ast.unparse(item.args)}){m_ret}:"
                        m_doc = ast.get_docstring(item)
                        if m_doc:
                            m_sig += f'\n        """{m_doc.strip().splitlines()[0]}"""'
                        file_lines.append(m_sig)
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        type_str = ast.unparse(item.annotation)
                        val_str = f" = {ast.unparse(item.value)}" if item.value else ""
                        file_lines.append(f"    {item.target.id}: {type_str}{val_str}")
                file_lines.append("")

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = [f"@{ast.unparse(d)}" for d in node.decorator_list]
                for d in decorators:
                    file_lines.append(d)
                ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                sig = f"def {node.name}({ast.unparse(node.args)}){ret}:"
                f_doc = ast.get_docstring(node)
                if f_doc:
                    sig += f'\n    """{f_doc.strip().splitlines()[0]}"""'
                file_lines.append(sig)
                file_lines.append("")

        file_lines.append("```")
        sections.append("\n".join(file_lines))
    return "\n\n".join(sections)


def copy_to_windows_clipboard(text: str) -> bool:
    """Copy text to Windows clipboard using clip.exe encoded as utf-16le."""
    try:
        subprocess.run(
            ["clip.exe"],
            input=text.encode(CLIPBOARD_ENCODING),
            check=True,
        )
        return True
    except FileNotFoundError:
        # Try full path if clip.exe is not in user PATH
        clip_path = Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32" / "clip.exe"
        if clip_path.exists():
            subprocess.run(
                [str(clip_path)],
                input=text.encode(CLIPBOARD_ENCODING),
                check=True,
            )
            return True
        return False
    except Exception as e:
        print(f"Warning: Failed to copy to clipboard: {e}", file=sys.stderr)
        return False


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    now = datetime.datetime.now(datetime.timezone.utc).astimezone()
    iso_time = now.isoformat()

    # 1. Git metadata
    git_branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    git_status = run_cmd(["git", "status", "--short", "--branch"], cwd=repo_root)
    git_log = run_cmd(["git", "log", "-n", str(NUM_RECENT_COMMITS), "--oneline"], cwd=repo_root)

    # 2. File tree
    tree_lines = [f"{repo_root.name}/"] + generate_file_tree(repo_root)
    tree_content = "\n".join(tree_lines)

    # 3. derma_config.yaml
    config_file = repo_root / "derma_config.yaml"
    config_content = (
        config_file.read_text(encoding="utf-8")
        if config_file.is_file()
        else "# derma_config.yaml not found"
    )

    # 4. Spec-Kit tasks
    specs_dir = repo_root / "specs"
    speckit_progress = parse_speckit_tasks(specs_dir)

    # 5. AST signatures
    pkg_dir = repo_root / "packages" / "dermafuzzy_core"
    ast_signatures = extract_ast_signatures(pkg_dir)

    # 6. Live pytest execution
    pytest_out = run_cmd([sys.executable, "-m", "pytest", "-q"], cwd=repo_root)

    # Assemble snapshot markdown
    snapshot = f"""=== ANTIGRAVITY LIVE SNAPSHOT ===
# DermaFuzzy-XAI System State Snapshot
- **Timestamp**: {iso_time}
- **Repository Root**: `{repo_root}`
- **Active Branch**: `{git_branch}`

---

## 1. Git State & Recent Commits
### Working Tree Status
```text
{git_status}
```

### Recent Commits (Last {NUM_RECENT_COMMITS})
```text
{git_log if git_log else "(No commits recorded yet)"}
```

---

## 2. Repository File Tree
```text
{tree_content}
```

---

## 3. Root Configuration (`derma_config.yaml`)
```yaml
{config_content}
```

---

## 4. Spec-Kit Feature & Task Progress
{speckit_progress}

---

## 5. Pure Core AST Signatures (`packages/dermafuzzy_core`)
{ast_signatures}

---

## 6. Live Unit Test Suite Output (`pytest -q`)
```text
{pytest_out}
```
"""

    # Write snapshot file
    target_path = repo_root / SNAPSHOT_FILENAME
    target_path.write_text(snapshot, encoding="utf-8")
    print(f"Generated: {target_path} ({len(snapshot.splitlines())} lines, {len(snapshot)} bytes)")

    # Copy to clipboard
    clipboard_ok = copy_to_windows_clipboard(snapshot)
    if clipboard_ok:
        print("Copied LIVE_STATE_SNAPSHOT.md contents to Windows clipboard (UTF-16LE).")
    else:
        print("Could not access clip.exe to copy snapshot to clipboard.")


if __name__ == "__main__":
    main()
