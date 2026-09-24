"""Allow running the package directly: ``python -m fsoc_tracker``."""

import sys


def main() -> None:
    args = sys.argv[1:]

    if "--gui" in args:
        args.remove("--gui")
        sys.argv = [sys.argv[0]] + args
        try:
            from PySide6.QtWidgets import QApplication

            from fsoc_tracker.gui.main_window import MainWindow
        except ImportError:
            print("ERROR: PySide6 not installed. Run: pip install PySide6", file=sys.stderr)
            sys.exit(1)
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    else:
        from fsoc_tracker.app.main import main as app_main
        app_main()


if __name__ == "__main__":
    main()
