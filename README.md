1. Install dependencies:
    - either `pip install -r requirements.txt`
    - or `pip install pytorch-lightning hydra-submitit-launcher hydra-core submitit pytorch-lightning-bolts scikit-learn` 

2. Configure your slurm parition by changing `partition: <partition>` in `config/sigterm.yaml` 

3. Run the code `python main.py -m`

4. Check the logs. The logging file will be written in the terminal. Eg. `less multirun/2022-09-25/20-28-21/0/main.log`

5. You will see the logs full of `Bypassing signal SIGTERM`
