"""Tests for viz_authoring.gpu_context — Viewport dataclass + include
resolver. GpuContext class itself requires a moderngl context, which
is heavy and fragile in CI; we leave that to integration tests
(consumers of viz_authoring like flame_sheep already exercise it
end-to-end)."""
from __future__ import annotations

import pytest

from viz_authoring.gpu_context import Viewport, _resolve_includes


class TestViewport:
    """Viewport is a lightweight (x, y, w, h) value class with
    __slots__ for memory efficiency."""

    def test_basic_construction(self):
        v = Viewport(10, 20, 100, 50)
        assert v.x == 10
        assert v.y == 20
        assert v.w == 100
        assert v.h == 50

    def test_repr_includes_all_dims(self):
        v = Viewport(1, 2, 3, 4)
        r = repr(v)
        # We don't pin format, just that all dims are visible
        assert '1' in r and '2' in r and '3' in r and '4' in r

    def test_slots_blocks_new_attrs(self):
        v = Viewport(0, 0, 1, 1)
        with pytest.raises(AttributeError):
            v.extra = 'nope'

    def test_zero_dims_allowed(self):
        # 0-area viewport (e.g. minimized output) is valid construction
        v = Viewport(0, 0, 0, 0)
        assert v.w == 0


class TestResolveIncludes:
    """_resolve_includes inlines `#include "file.glsl"` directives
    by reading the file from shader_dir."""

    def test_no_includes_returns_unchanged(self, tmp_path):
        src = 'void main() { gl_Position = vec4(0); }'
        out = _resolve_includes(src, tmp_path)
        assert out == src

    def test_single_include(self, tmp_path):
        (tmp_path / 'rng.glsl').write_text('float rand() { return 0.5; }')
        src = '#include "rng.glsl"\nvoid main() {}'
        out = _resolve_includes(src, tmp_path)
        assert 'float rand() { return 0.5; }' in out
        assert '#include' not in out
        assert 'void main() {}' in out

    def test_multiple_includes(self, tmp_path):
        (tmp_path / 'a.glsl').write_text('AAA')
        (tmp_path / 'b.glsl').write_text('BBB')
        src = '#include "a.glsl"\n#include "b.glsl"\nmain()'
        out = _resolve_includes(src, tmp_path)
        assert 'AAA' in out
        assert 'BBB' in out
        assert '#include' not in out

    def test_missing_file_raises(self, tmp_path):
        src = '#include "missing.glsl"'
        with pytest.raises(FileNotFoundError):
            _resolve_includes(src, tmp_path)

    def test_does_not_recurse(self, tmp_path):
        # If included file has its OWN #include, current implementation
        # leaves it in place. Documenting current behavior — if recursive
        # resolution is needed someday, this test will catch the change.
        (tmp_path / 'inner.glsl').write_text('#include "deeper.glsl"')
        (tmp_path / 'deeper.glsl').write_text('DEEPER')
        src = '#include "inner.glsl"'
        out = _resolve_includes(src, tmp_path)
        # The outer include is resolved, but the inner one is left as-is
        assert '#include "deeper.glsl"' in out
        assert 'DEEPER' not in out


class TestGpuRingTimerImport:
    """GpuRingTimer is a context manager around a moderngl query —
    most behavior needs a moderngl context. Just verify the class
    surfaces correctly."""

    def test_class_exists_with_expected_interface(self):
        from viz_authoring.gpu_context import GpuRingTimer
        # Has the context-manager protocol
        assert hasattr(GpuRingTimer, '__enter__')
        assert hasattr(GpuRingTimer, '__exit__')
        # Has __slots__ keeping it lightweight
        assert hasattr(GpuRingTimer, '__slots__')


class TestPublicSurface:
    """Verify the package's __init__ re-exports what we expect."""

    def test_top_level_imports(self):
        import viz_authoring
        # Core public names
        for name in [
            'Viewport', 'GpuContext', 'GpuRingTimer',
            'GPU_TIMING_ENABLED', '_resolve_includes',
            '_bind_default_framebuffer',
            'LOAD_THRESHOLD', 'LOAD_CHECK_INTERVAL',
            'is_loaded', 'sleep_if_loaded',
            'init_worker_subprocess',
        ]:
            assert hasattr(viz_authoring, name), f'missing public name: {name}'
