import subprocess
import sys
import textwrap


def test_gpumemprof_import_succeeds_when_viz_imports_blocked():
    code = textwrap.dedent(
        """
        import builtins

        blocked_roots = {"matplotlib", "seaborn", "plotly"}
        original_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".", 1)[0] in blocked_roots:
                raise ModuleNotFoundError(f"blocked import: {name}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import

        import gpumemprof

        assert hasattr(gpumemprof, "GPUMemoryProfiler")
        try:
            gpumemprof.MemoryVisualizer
        except ImportError as exc:
            assert "optional visualization dependencies" in str(exc)
        else:
            raise AssertionError("Expected ImportError when requesting MemoryVisualizer")

        print("ok")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
