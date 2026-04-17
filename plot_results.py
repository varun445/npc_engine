import argparse
import os
import sys
from datetime import datetime

from evaluate import plot_mode_comparison


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a mode-comparison plot from an existing evaluation CSV.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to an existing evaluation results CSV file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PNG path. Defaults to <csv_dir>/mode_comparison_<timestamp>.png "
            "when not provided."
        ),
    )
    parser.add_argument(
        "--modes",
        default="direct,presearch,semantic",
        help=(
            "Comma-separated mode labels to compare. "
            "Supports aliases like 'direct' (mapped to llm_only)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.dirname(csv_path) or "."
        output_path = os.path.join(output_dir, f"mode_comparison_{timestamp}.png")

    modes = [m.strip() for m in (args.modes or "").split(",") if m.strip()]
    generated = plot_mode_comparison(
        results_csv_path=csv_path,
        output_image_path=output_path,
        modes=modes,
    )
    if not generated:
        print(
            "[ERROR] Plot was not generated. Check CSV content (mode/metric columns) "
            "and ensure matplotlib is installed.",
            file=sys.stderr,
        )
        return 1

    print(f"Plot file → {generated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
