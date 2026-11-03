"""Pure Python experiment settings, safe for detached dispatchers."""
from pathlib import Path
import os

PROJECTS = Path(os.environ.get('PHRM_PROJECTS_ROOT', '.'))

OFFICIAL = PROJECTS / 'TQNet_clean'

SEEDS = (2024, 2022, 2023)

SPECS = {
    'PEMS04': dict(nodes=307, cycle=288, daily=288, weekly=2016, split='60/20/20',
                   horizons=(12,24,48,96), source=OFFICIAL/'dataset/PEMS04.npz',
                   edge=PROJECTS/'PhysicsResidualKV_MultiDataset/dataset/physical/PEMS04.csv',
                   loader='Dataset_PEMS', topology='road_network'),
    'PEMS07': dict(nodes=883, cycle=288, daily=288, weekly=2016, split='60/20/20',
                   horizons=(12,24,48,96), source=PROJECTS/'PhysicsResidualKV_MultiDataset/dataset/raw/PEMS07.npz',
                   edge=PROJECTS/'PhysicsResidualKV_MultiDataset/dataset/physical/PEMS07.csv',
                   loader='Dataset_PEMS', topology='road_network'),
    'Solar': dict(nodes=137, cycle=144, daily=144, weekly=1008, split='70/10/20',
                  horizons=(96,192,336,720), source=OFFICIAL/'dataset/solar_AL.txt',
                  edge=PROJECTS/'PhysicsResidualKV_LongHorizon/dataset/graph/Solar.csv',
                  loader='Dataset_Solar', topology='ordered_channel_proxy'),
    'Electricity': dict(nodes=321, cycle=168, daily=24, weekly=168, split='70/10/20',
                        horizons=(96,192,336,720), source=OFFICIAL/'dataset/electricity_OT.csv',
                        edge=PROJECTS/'PhysicsResidualKV_LongHorizon/dataset/graph/Electricity.csv',
                        loader='Dataset_Custom', topology='ordered_channel_proxy'),
}

SPECS={k:v for k,v in SPECS.items() if k=='PEMS07'}
for name,spec in SPECS.items():
    spec.update(use_revin=0,dropout=0.,base_lr=.003,base_batch=32,base_epochs=30,
                patience=5,lradj='type3',optimizer='Adam',fresh_base=True,adapter_batch=4,
                official_script=f'scripts/TQNet/{name.lower()}.sh')

def checkpoint_path(root,dataset,horizon,seed):
    assert dataset in SPECS
    manifest_path=Path(root)/'audit/reused_baselines.json'
    if manifest_path.is_file():
        import json
        entry=json.loads(manifest_path.read_text()).get(f'h{int(horizon)}_seed{int(seed)}')
        if entry:return Path(entry['path'])
    return Path(root)/'base_checkpoints'/dataset/f'h{int(horizon)}'/f'seed{int(seed)}.pth'
