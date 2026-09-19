"""CyberHUD entry point; dependency errors stay readable."""
import sys


def main():
    try:
        from cyberhud.app import main as run
    except (ImportError, OSError) as exc:
        print(f"CyberHUD dependency unavailable: {exc}\n"
              r"Install with: .\.venv\Scripts\python.exe -m pip install -r requirements.txt",
              file=sys.stderr)
        return 1
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
