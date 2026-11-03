"""Continuous per-trajectory worker, independent of learner RPCs."""
import asyncio
import sys
import threading

from .attempt_retry import RetryableAgentExit


class Sampler:
    def __init__(self, state, execute, update, emit=lambda *a, **k: None):
        self.state, self.execute, self.update, self.emit = state, execute, update, emit
        self.stop_event = threading.Event()
        self.error = None
        self.thread = None

    async def _one(self, lease):
        try:
            artifact = await self.execute(lease)
            self.state.complete(lease, artifact)
            self.emit('trajectory_committed', group=lease['group'], slot=lease['slot'],
                      version=lease['version'], **self.state.status())
        except RetryableAgentExit as exc:
            retry = self.state.retry_agent_exit(lease, exc)
            self.emit('trajectory_retry_scheduled' if retry else 'trajectory_retries_exhausted',
                      group=lease['group'], slot=lease['slot'], attempt=lease['attempt'],
                      version=lease['version'], error=str(exc), **self.state.status())
            if not retry:
                raise RuntimeError('Agent exit retry budget exhausted for one trajectory') from exc
        except BaseException as exc:
            # execute must close/revoke its HTTP session before returning/raising.
            self.state.fail(lease, f'{type(exc).__name__}: {exc}')
            raise

    async def _update(self, engine):
        try:
            await self.update(engine)
        except BaseException as exc:
            self.state.updated(engine['id'], engine['target_version'], engine['generation'], error=exc)
            raise
        self.state.updated(engine['id'], engine['target_version'], engine['generation'])
        self.emit('engine_updated', engine_id=engine['id'], **self.state.status())

    async def run(self):
        pending = set()
        updaters = set()
        try:
            while not self.stop_event.is_set():
                done = [t for t in pending if t.done()]
                for t in done:
                    pending.remove(t)
                    t.result()
                for task in [t for t in updaters if t.done()]:
                    updaters.remove(task)
                    task.result()
                self.state.admit()
                while (engine := self.state.update_candidate()) is not None:
                    updaters.add(asyncio.create_task(self._update(engine)))
                while (lease := self.state.dispatch()) is not None:
                    pending.add(asyncio.create_task(self._one(lease)))
                if self.state.status()['completed_steps'] == len(self.state.data['tasks']) // self.state.data['config']['batch_groups']:
                    break
                await asyncio.sleep(.02)
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_results = []
            for task in pending:
                task.cancel()
            # Updating in a worker thread cannot safely be abandoned; await its
            # completion before ending ownership. RPCs have bounded deadlines.
            if pending:
                cleanup_results.extend(await asyncio.gather(*pending, return_exceptions=True))
            if updaters:
                cleanup_results.extend(await asyncio.gather(*updaters, return_exceptions=True))
            # Deliberate cancellation of attempts is normal shutdown. Failures
            # while draining/updating are not: do not report a successful stop
            # just because the normal loop no longer polls those tasks.
            if primary_error is None:
                for result in cleanup_results:
                    if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                        raise result

    def start(self):
        def worker():
            try:
                asyncio.run(self.run())
            except BaseException as exc:
                self.error = exc
                self.stop_event.set()
        self.thread = threading.Thread(target=worker, name='rsi-continuous-sampler', daemon=True)
        self.thread.start()

    def check(self):
        if self.error:
            raise RuntimeError('Continuous sampling failed') from self.error

    def stop(self, timeout=180):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout)
            if self.thread.is_alive():
                raise TimeoutError('Sampler cleanup incomplete; keep run ownership and stop deployment')
        self.check()
