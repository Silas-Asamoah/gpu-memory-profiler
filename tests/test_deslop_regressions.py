import subprocess
import sys
import textwrap

from gpumemprof.context_profiler import profile_function


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


class _DummyProfiler:
    def __init__(self):
        self.calls = 0
        self.seen_name = None

    def profile_function(self, func):
        self.calls += 1
        self.seen_name = getattr(func, "__name__", None)
        func()
        return object()


def test_profile_function_decorator_executes_once_and_returns_result():
    profiler = _DummyProfiler()
    state = {"calls": 0}

    @profile_function(name="custom_profile_name", profiler=profiler)
    def tracked_operation():
        state["calls"] += 1
        return "ok"

    result = tracked_operation()

    assert result == "ok"
    assert state["calls"] == 1
    assert profiler.calls == 1
    assert profiler.seen_name == "custom_profile_name"
