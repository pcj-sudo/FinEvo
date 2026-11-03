"""Write local controller settings from explicit deployment values."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--agent-image-id',required=True)
    p.add_argument('--gateway-url',required=True)
    p.add_argument('--output',type=Path,default=Path('private/controller.json'))
    a=p.parse_args()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',a.agent_image_id):p.error('Expected an immutable local Docker image ID')
    u=urlsplit(a.gateway_url)
    if u.scheme not in ('http','https') or not u.hostname or u.username or u.password:p.error('Expected an HTTP(S) gateway URL without embedded credentials')
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'training/configs/controller.example.json').read_text())
    config.update(agent_image=a.agent_image_id,gateway_url=a.gateway_url.rstrip('/'))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(config,indent=2)+'\n')
    print(a.output)
