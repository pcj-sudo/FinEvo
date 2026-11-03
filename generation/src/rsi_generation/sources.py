"""Source storage: published materials separate from cleaning state."""
from pathlib import Path


def source_folder(root, source_id):
    from .files import identifier
    identifier(source_id)
    root = Path(root)
    current = root / 'workspace/cleaning' / source_id
    legacy = root / 'sources' / source_id
    # Existing installations remain readable until explicit migration.
    return legacy if not current.exists() and (legacy / 'source.json').exists() else current


def original_folder(root, source_id):
    return source_folder(root, source_id) / 'original'


def session_work(root, source_id, kind):
    legacy = Path(root) / 'workspace/sources' / source_id / kind
    # Preserve the cwd embedded in old Codex/OpenHands conversations.
    return legacy if legacy.exists() else source_folder(root, source_id) / 'agents' / kind


def cleaned_folder(root, source_id, version):
    from .files import identifier
    identifier(source_id)
    identifier(version)
    for base in ('sources/ready/v1', 'sources/ready'):
        ready = Path(root) / base / source_id / version
        if ready.exists():
            return ready
    # v2 public exports contain only XLSX. Internal hashes still cover all evidence.
    return source_folder(root, source_id) / 'versions' / version / 'cleaned'


def publish_v2(root, source_id, version, cleaned):
    """Publish only audited v2 workbooks; retain full evidence privately."""
    from .files import atomic_copy, read_json, tree_hash, PipelineError
    folder = source_folder(root, source_id)
    source = read_json(folder / 'source.json', {})
    audit = read_json(folder / 'versions' / version / 'audit.json', {}).get('result', {})
    if (source.get('cleaning_contract') != 'v2' or source.get('ready_version') != version
            or audit.get('verdict') != 'valid' or audit.get('unresolved_issues')):
        raise PipelineError('v2 publication requires v2 cleaning and a valid audit without unresolved issues')
    if tree_hash(cleaned) != source.get('cleaned_hash'):
        raise PipelineError('Verified source artifacts were modified')
    target = Path(root) / 'sources/ready/v2' / source_id / version
    target.mkdir(parents=True, exist_ok=True)
    if any(p.name != 'reference.xlsx' for p in target.iterdir()):
        raise PipelineError('v2 source may contain only reference.xlsx')
    atomic_copy(Path(cleaned) / 'reference.xlsx', target / 'reference.xlsx')
    return target
