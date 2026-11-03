"""Run Harbor with failure evidence on node-local storage, independent of GPFS."""
import json
import os
from pathlib import Path
import sys
import traceback


def exception_record(exc):
    # Do not log exception messages or locals: they can contain agent credentials.
    chain, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        chain.append({
            'type': type(exc).__name__,
            'errno': getattr(exc, 'errno', None),
            'strerror': getattr(exc, 'strerror', None),
            'frames': [{'file': f.filename, 'line': f.lineno, 'function': f.name}
                       for f in traceback.extract_tb(exc.__traceback__)],
        })
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return {'exceptions': chain}


def main():
    try:
        from harbor.cli.main import app
        app()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise
        report = exception_record(exc)
        location = os.environ.get('RSI_HARBOR_DIAGNOSTIC')
        if location:
            try:
                Path(location).write_text(json.dumps(report, indent=2) + '\n')
            except OSError:
                pass
        try:
            sys.__stderr__.write('RSI_HARBOR_FAILURE ' + json.dumps(report) + '\n')
            sys.__stderr__.flush()
        except OSError:
            pass
        raise


if __name__ == '__main__':
    main()
