## Install 

git clone https://github.com/Microsoft/cameratraps
git clone https://github.com/Microsoft/ai4eutils

set PYTHONPATH=<path_du_repo_dl>\cameratraps;<path_du_repo_dl>\ai4eutils

pip install tensorflow==1.13.1 ( ou gpu ) 
pip install Pillow humanfriendly matplotlib tqdm jsonpickle


## run 
python CameraTraps/detection/run_tf__batch.py megadetector_v3.pb <dossier_src> some_output_file.json
