"""Command-line entry points. Run from the standalone generation directory."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from .config import load
from .files import PipelineError, identifier, lock, read_json
from .workflow import Workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description='Construct financial spreadsheet tasks with Codex and Harbor')
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='generation/ directory')
    parser.add_argument('--config', type=Path, help='JSON overrides merged with configs/generation.json')
    parser.add_argument('--harness', choices=['codex', 'openhands'], help='Execution framework for author and solver; independent of model')
    parser.add_argument('--model', help='Model ID, optionally with a provider prefix; applies to author and solver')
    parser.add_argument('--reasoning-effort', help='Reasoning setting passed to the selected model')
    sub = parser.add_subparsers(dest='command', required=True)
    ingest = sub.add_parser('ingest', help='Snapshot the original material directory')
    ingest.add_argument('source_id')
    ingest.add_argument('directory', type=Path)
    run = sub.add_parser('run', help='Start or resume construction')
    run.add_argument('task_id')
    run.add_argument('--source', required=True)
    run.add_argument('--type', choices=['modeling', 'debugging'], required=True)
    run.add_argument('--reuse-source', action='store_true', help='Reuse an approved source instead of starting fresh cleaning')
    resume = sub.add_parser('resume', help='Continue from persisted workflow state')
    resume.add_argument('task_id')
    resume.add_argument('--retry-evaluation', action='store_true', help='After manual repair, retry the stopped evaluation; preserve source/task and previous evidence')
    resume.add_argument('--retry-api', action='store_true', help='After restoring API access, retry the stopped author/evaluation call with its original context')
    resume.add_argument('--retry-author', action='store_true', help='Explicitly retry an exhausted author call, retaining context and evidence')
    resume.add_argument('--revise-task', action='store_true', help='Return a stopped invalid target reference to its constructor as a new task version')
    inspect = sub.add_parser('inspect', help='Display state and artifact paths')
    inspect.add_argument('task_id')
    watch_parser = sub.add_parser('watch', help='Read live progress without restarting the task')
    watch_parser.add_argument('task_id')
    watch_parser.add_argument('--once', action='store_true', help='Show current progress and exit')
    watch_parser.add_argument('--verbose', action='store_true', help='Include individual model and tool events')
    batch = sub.add_parser('batch', help='Run an immutable task manifest with bounded parallelism; rerun to resume interrupted tasks')
    batch.add_argument('batch_id')
    batch.add_argument('--manifest', required=True, type=Path)
    batch.add_argument('--concurrency', type=int, default=1)
    batch.add_argument('--retry-failed', action='store_true', help='After repairing the pipeline, retry failed workers; does not clear needs_human task states')
    batch.add_argument('--retry-api', action='store_true', help='Explicitly retry API-stopped tasks; other human-intervention stops remain blocked')
    batch.add_argument('--retry-evaluation', action='store_true', help='Explicitly retry stopped evaluations after environment recovery')
    batch.add_argument('--retry-author', action='store_true', help='Explicitly retry exhausted author calls')
    batch_watch = sub.add_parser('watch-batch', help='Watch all task rounds in a batch')
    batch_watch.add_argument('batch_id')
    batch_watch.add_argument('--once', action='store_true')
    export = sub.add_parser('export', help='Copy an accepted task to an evaluation directory')
    export.add_argument('task_id')
    export.add_argument('--to', required=True, type=Path)
    sub.add_parser('doctor', help='Check installation without invoking a model')
    sub.add_parser('prepare-runtime', help='Explicitly build and probe the shared local runtime before generation')
    args = parser.parse_args(argv)
    # Invoking .venv/bin/rsi-generate directly does not activate the venv.
    # Its peer Harbor executable must still be available to subprocesses.
    bindir = str(Path(sys.executable).parent)
    if os.environ.get('PATH', '').split(os.pathsep)[0] != bindir:
        os.environ['PATH'] = bindir + os.pathsep + os.environ.get('PATH', '')
    root = args.root.expanduser().resolve()
    try:
        if args.command == 'watch-batch':
            from .batch import watch_batch
            watch_batch(root, identifier(args.batch_id), once=args.once)
            return 0
        if args.command == 'watch':
            from .progress import watch
            identifier(args.task_id)
            watch(root, args.task_id, once=args.once, verbose=args.verbose)
            return 0
        config = load(root, args.config, harness=args.harness, model=args.model, reasoning_effort=args.reasoning_effort)
        if getattr(args, 'task_id', None):
            identifier(args.task_id)
        if args.command == 'batch':
            from .batch import run_batch
            result = run_batch(root, config, identifier(args.batch_id), args.manifest, args.concurrency,
                               retry_failed=args.retry_failed, retry_api=args.retry_api,
                               retry_evaluation=args.retry_evaluation, retry_author=args.retry_author)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if result['status'] == 'completed_with_errors' else 0
        if args.command == 'inspect':
            result = read_json(root / 'state/tasks' / (args.task_id + '.json'))
            if result is None:
                raise PipelineError('No state exists for this task')
        elif args.command == 'prepare-runtime':
            from .runtime import prepare
            result = prepare(root, config)
        elif args.command == 'doctor':
            from .harness import selected, sdk_python
            framework = selected(config)
            executable = config['codex']['executable'] if framework == 'codex' else sdk_python(root, config)
            author = shutil.which(executable)
            result = {'root': str(root), 'python': sys.version.split()[0], 'harness': framework,
                      'author_executable': author, 'harbor': shutil.which(config['harbor']['executable']),
                      'constructor': config['codex']['model'], 'solver': config['solver']['model']}
            if not author or not result['harbor']:
                raise PipelineError('Install the selected harness and Harbor; use uv run rsi-generate')
            if framework == 'openhands':
                from .openhands import OpenHands
                OpenHands(root, config).check_version()
            from .runtime import preflight
            result['runtime'] = preflight(root, config)
        else:
            # Shared migration gate still excludes an older CLI's global lock.
            # New commands conflict only when they own the same task/source.
            from contextlib import nullcontext
            task_lock = lock(root / 'state/locks/tasks' / args.task_id) if getattr(args, 'task_id', None) else nullcontext()
            with lock(root / 'state', shared=True), task_lock:
                flow = Workflow(root, config)
                if args.command == 'ingest':
                    result = {'source_id': flow.ingest(args.source_id, args.directory)}
                elif args.command in ('run', 'resume'):
                    options = {'retry_evaluation': True} if getattr(args, 'retry_evaluation', False) else {}
                    if getattr(args, 'retry_api', False):
                        options['retry_api'] = True
                    if getattr(args, 'retry_author', False):
                        options['retry_author'] = True
                    if getattr(args, 'revise_task', False):
                        options['revise_task'] = True
                    result = flow.run(args.task_id, getattr(args, 'source', None), getattr(args, 'type', None),
                                      fresh_source=args.command == 'run' and not args.reuse_source, **options)
                else:
                    result = {'exported': str(flow.export(args.task_id, args.to))}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (PipelineError, OSError, ValueError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Stopped watching; task continues.' if args.command == 'watch' else
              'Interrupted; state and records preserved. Use resume.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
