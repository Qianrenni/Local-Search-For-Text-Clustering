from pathlib import Path
class SETTING:
    SEED=1314
    ROOT=Path(__file__).parent
    RESULT=ROOT/'result'
    DATA=ROOT/'data'
    PREPROCESS_DATA = ROOT/'preprocess_data'
    PROCESS_DATA = ROOT/'processed_data'
    DEPENDENCY=ROOT/'dependency'

for key in dir(SETTING):
    value = getattr(SETTING,key)
    if isinstance(value,Path):
        value.mkdir(parents=True,exist_ok=True)
