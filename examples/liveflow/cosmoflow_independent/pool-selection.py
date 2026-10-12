#!/usr/bin/env python3
"""Select explicitly configured independent data pools; defaults stay in the launcher."""
import json
import pathlib
import sys

def select(config, block, arm):
    assignments = config['blocks'][str(block)]
    expected = {'original', 'maxresource-adapted', 'static-best', 'liveflow'}
    if set(assignments) != expected:
        raise ValueError('A block must assign all four arms')
    labels = [v['pool'] for v in assignments.values()]
    if not all(pathlib.Path(v['data_root']).is_absolute() for v in assignments.values()):
        raise ValueError('Pool roots must be absolute')
    roots = [pathlib.Path(v['data_root']).resolve(strict=True) for v in assignments.values()]
    if len(set(labels)) != 4 or len(set(roots)) != 4:
        raise ValueError('Each arm in a block must have a different pool')
    if not all(p.is_absolute() and (p/'train').is_dir() for p in roots):
        raise ValueError('Every pool must contain a train directory')
    chosen = assignments[arm]
    return chosen['pool'], str(pathlib.Path(chosen['data_root']).resolve(strict=True))

if __name__ == '__main__':
    config = json.loads(pathlib.Path(sys.argv[1]).read_text())
    print('\n'.join(select(config, sys.argv[2], sys.argv[3])))
