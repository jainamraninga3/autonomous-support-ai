Add files in this folder to be injest by script 


Two blockers first
1. policy/ has no PDFs. It contains only read.md ("Add files in this folder to be injest by script"). The script exits with No PDFs found until you put them there.

2. requests isn't installed on your host, and there's no backend/.venv in this tree:

pip install requests
That's all it needs — the script talks to the API over HTTP, so it needs no database credentials of its own.

So, in order:

pip install requests
cd backend
python -m scripts.ingest_folder ../policy