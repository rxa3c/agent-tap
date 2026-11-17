"""Allow running as `python -m agent_tap`."""

from agent_tap.cli import main_entry


def main() -> None:
    main_entry()


if __name__ == "__main__":
    main()
