import argparse
import json
from pathlib import Path

from .data import import_tasks
from .files import read_json


def main():
    parser = argparse.ArgumentParser(description='Independent financial agent RL tools')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('import-tasks')
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--seed', default='rsi-finance-v1')
    p = sub.add_parser('serve')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=18002)
    p = sub.add_parser('status')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'import-tasks':
        result = import_tasks(root, read_json(args.selection), args.seed)
        print(json.dumps({'tasks': len(result['tasks']), 'source_splits': result['source_splits']}, indent=2))
    elif args.command == 'serve':
        import sys
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError('The training controller requires Python 3.12')
        # Exercise the actual dependency boundary before reporting HTTP readiness.
        from harbor.models.job.config import JobConfig
        from .openhands_agent import TrainingOpenHands
        from aiohttp import web
        from .controller import Controller
        token = args.token_file.read_text().strip()
        if len(token) < 32:
            parser.error('Controller token must have at least 32 characters')
        controller = Controller(root, read_json(args.config), token)
        web.run_app(controller.app(), host=args.host, port=args.port, access_log=None)
    elif args.command == 'status':
        from collections import Counter
        records = [read_json(f) for f in (root / 'runs/rollouts').glob('*/state.json')]
        print(json.dumps(dict(Counter(x['status'] for x in records)), indent=2))


if __name__ == '__main__':
    main()
