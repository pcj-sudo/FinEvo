"""Register a prepared workbook without importing its preparation history."""
import argparse
import shutil
from pathlib import Path
from rsi_generation.files import identifier,tree_hash,write_json,read_json,PipelineError
from rsi_generation.sources import source_folder


def register(root, workbook, source_id='financialstatements1-example'):
    root,workbook=Path(root).resolve(),Path(workbook).resolve()
    identifier(source_id)
    if not workbook.is_file() or workbook.suffix.lower()!='.xlsx':
        raise ValueError('Expected an existing XLSX workbook')
    destination=root/'sources/ready/v2'/source_id/'s0001'
    record=source_folder(root,source_id)/'source.json'
    previous=read_json(record)
    if previous:
        if not previous.get('prepared_source') or tree_hash(destination)!=previous['cleaned_hash']:
            raise PipelineError('Existing source has different contents or preparation mode')
        from hashlib import sha256
        if sha256(workbook.read_bytes()).hexdigest()!=previous['cleaned_hash']['reference.xlsx']['sha256']:
            raise PipelineError('Refusing to replace a registered source')
        return destination
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copy2(workbook,destination/'reference.xlsx')
    write_json(record,dict(source_id=source_id,status='ready',ready_version='s0001',
        prepared_source=True,prepared_path=str(destination.relative_to(root)),
        cleaning_contract='v2',cleaned_hash=tree_hash(destination)))
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('generation'))
    p.add_argument('--workbook',type=Path,default=Path('examples/financialstatements1/reference.xlsx'))
    p.add_argument('--source-id',default='financialstatements1-example')
    a=p.parse_args();print(register(a.root,a.workbook,a.source_id))

if __name__=='__main__':main()
