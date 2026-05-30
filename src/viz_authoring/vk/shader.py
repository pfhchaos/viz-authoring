"""GLSL → SPIR-V compilation via glslc.

Same toolchain wallpaper_ml uses for compute shaders — we just expose
it here too so the graphics-side code doesn't reach across into
wallpaper_ml for a utility. Returns the SPIR-V bytes; the caller hands
them to vkCreateShaderModule.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


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


def compile_shader(source_path: Path | str, stage: str | None = None) -> bytes:
    """Compile GLSL to SPIR-V binary. Returns the SPIR-V bytes.

    `stage`: 'vertex' / 'fragment' / 'compute' / 'geometry' / 'tesscontrol'
    / 'tesseval'. If omitted, inferred from the file's suffix.

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

    with tempfile.NamedTemporaryFile(suffix='.spv', delete=False) as spv:
        spv_path = spv.name
    try:
        result = subprocess.run(
            ['glslc', f'-fshader-stage={stage}', str(source_path),
             '-o', spv_path],
            capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f'glslc failed for {source_path}:\n{result.stderr}')
        return Path(spv_path).read_bytes()
    finally:
        Path(spv_path).unlink(missing_ok=True)
