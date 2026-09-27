
```
ABA_VLM
├─ .DS_Store
├─ cheatsheet.md
├─ configuration
│  └─ test.yml
├─ imageset
│  ├─ .DS_Store
│  └─ EmoMadrid
│     └─ EM0503.jpg
├─ requirements.txt
└─ script
   ├─ analysis
   │  ├─ __pycache__
   │  │  ├─ test2.cpython-310.pyc
   │  │  └─ test2.cpython-312.pyc
   │  └─ test2.py
   ├─ experiment
   │  ├─ __pycache__
   │  │  ├─ test1.cpython-310.pyc
   │  │  └─ test1.cpython-312.pyc
   │  └─ test1.py
   └─ visualization

```


### Clone the repository

This project uses [facebook SLIP](https://github.com/facebookresearch/SLIP?utm_source=chatgpt.com) as a Git submodule.

```bash
git clone --recurse-submodules https://github.com/psymichaelzhu/ABA_VLM.git
cd ABA_VLM

conda env create -f environment.yml
conda activate slip_aba

bash scripts/download_checkpoints.sh
```

