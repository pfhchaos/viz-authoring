"""GLSL → SPIR-V compilation via glslc.

Same toolchain wallpaper_ml uses for compute shaders — we just expose
it here too so the graphics-side code doesn't reach across into
wallpaper_ml for a utility. Returns the SPIR-V bytes; the caller hands
them to vkCreateShaderModule.

Supports:
  - stage inference from suffix (.vert/.frag/.comp/etc.)
  - #include "file.glsl" resolution (manual inlining — glslc's native
    -I can find files but we need to substitute placeholders WITHIN
    includes, which means preprocessing them as strings first)
  - source_transform callback — for shader-specific text substitution
    such as flame.comp's {{SYMMETRY_GROUPS}} placeholder
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


# Mapping from filename suffix → glslc stage name. Lets callers pass a
# path without a separate stage argument; deduces from .vert / .frag /
# .comp etc.
_SUFFIX_STAGE = {
    '.vert': 'vertex',
    '.frag': 'fragment',
    '.comp': 'compute',
    '.geom': 'geometry',
    '.tesc': 'tesscontrol',
    '.tese': 'tesseval',
}


def resolve_includes(source: str, shader_dir: Path) -> str:
    """Inline #include "file.glsl" directives recursively.

    Recursion: an included file may itself contain #include. Resolves
    in include-order; cycles aren't detected (would loop forever — but
    no shaders in this project have them).
    """
    def _replace(m: re.Match) -> str:
        inc_path = shader_dir / m.group(1)
        return resolve_includes(inc_path.read_text(), inc_path.parent)
    return re.sub(r'#include\s+"(.+?)"', _replace, source)


def compile_shader(source_path: Path | str, stage: str | None = None,
                   source_transform: Callable[[str], str] | None = None
                   ) -> bytes:
    """Compile GLSL to SPIR-V binary. Returns the SPIR-V bytes.

    `stage`: 'vertex' / 'fragment' / 'compute' / 'geometry' / 'tesscontrol'
    / 'tesseval'. If omitted, inferred from the file's suffix.

    `source_transform`: optional callable(source_text) → source_text
    applied AFTER #include resolution. Use for placeholder substitution
    inside an included file (e.g. flame.comp's {{SYMMETRY_GROUPS}}
    inside variations.glsl).

    Raises RuntimeError if glslc fails (missing tool, invalid GLSL, etc.).
    The error includes glslc's stderr so the GLSL diagnostic is visible.
    """
    source_path = Path(source_path)
    if stage is None:
        stage = _SUFFIX_STAGE.get(source_path.suffix)
        if stage is None:
            raise ValueError(
                f'Cannot infer shader stage from suffix {source_path.suffix!r}. '
                f'Pass stage= explicitly or use one of {list(_SUFFIX_STAGE)}.')

    # Read + inline includes + apply transform → write to a temp file
    # in the original shader dir so glslc's error messages keep the
    # familiar location context.
    src = resolve_includes(source_path.read_text(), source_path.parent)
    if source_transform is not None:
        src = source_transform(src)

    # Use a tempfile co-located with the original so any residual
    # relative paths resolve sensibly.
    with tempfile.NamedTemporaryFile(suffix=source_path.suffix, delete=False,
                                       dir=source_path.parent, mode='w') as tmp:
        tmp.write(src)
        tmp_src_path = tmp.name
    spv_path = tmp_src_path + '.spv'
    try:
        result = subprocess.run(
            ['glslc', f'-fshader-stage={stage}', tmp_src_path,
             '-o', spv_path],
            capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f'glslc failed for {source_path}:\n{result.stderr}')
        return Path(spv_path).read_bytes()
    finally:
        Path(tmp_src_path).unlink(missing_ok=True)
        Path(spv_path).unlink(missing_ok=True)
