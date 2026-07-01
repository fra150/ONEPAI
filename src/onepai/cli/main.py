"""Entry point della CLI ONEPAI."""

import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    from ..ui.cli import OnepaiCLI
    return OnepaiCLI().main(argv)


if __name__ == "__main__":
    sys.exit(main())
