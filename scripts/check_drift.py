import json
import sys

from app import drift
from app.config import settings


def main() -> int:
    result = drift.detect(threshold=settings.drift_threshold)
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "drift" else 0


if __name__ == "__main__":
    sys.exit(main())
