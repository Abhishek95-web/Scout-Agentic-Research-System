from importlib import import_module

cli = import_module("football_intelligence.main")


def test_windows_console_streams_use_safe_encoding_errors(monkeypatch):
    class Console:
        errors = None

        def reconfigure(self, *, errors):
            self.errors = errors

    stdout = Console()
    stderr = Console()
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    monkeypatch.setattr(cli.sys, "stderr", stderr)

    cli._make_console_safe()

    assert stdout.errors == "backslashreplace"
    assert stderr.errors == "backslashreplace"
