# dbbert
DB-BERT is a database tuning tools that exploits natural language text as additional input. It extracts recommendations for database parameter settings from tuning-related text via natural language analysis. It optimizes parameter settings for a given workload and performance metric using reinforcement learning.

# Quickstart
- DB-BERT was tested on an Amazon EC2 p3.2xlarge instance using Deep Learning AMI (Ubuntu 18.04) Version 43.0.
- Install Postgres 13.2 and the TPC-H benchmark database.
- Install required Python packages, including Huggingface Transformers, stable-baselines-3, psycopg2, Numpy, Pytorch, and streamlit.
- DB-BERT uses configuration files, an example file can be found under `config/pg_tpch_manydocs.ini`.
- For a first try, update credentials in the `[DBMS]` section. You may adapt tuning time and other parameters in the `[BENCHMARK]` section.
- Execute `src/run/tuning_run.py`, passing the configuration file as first (and only) argument, e.g. (from main directory):
`PYTHONPATH='src' python3 src/run/tuning_run.py config/pg_tpch_manydocs.ini`
- DB-BERT will parse input text (text from 100 Web documents), initialize the RL algorithm, and start tuning Postgres for TPC-H.
- Try `config/pg_tpch_onedoc.ini` for tuning using a single, TPC-H specific text document (forum discussion on TPC-H tuning).

# GUI
- To start the GUI, run `streamlit run src/run/interface.py` from the root directory.
- If accessing DB-BERT on a remote EC2 server, make sure to enable inbound traffic to port 8501.
- Enter the URL shown in the console into your Web browser to access the interface.
- You can select settings to read from configuration files in the `demo_configs` folder.
- Select a collection of tuning text documents for extraction (e.g., from the `demo_docs` folder).
- You may change parameter related to database access, learning, and tuning goals.
- Click on the `Start Tuning` button to start the tuning process.

# Ongoing Work
- Switching language models used from BERT to BART (despite the project name).
- Refined text document pre-processing: extract most relevant passage for each parameter.

# Running Experiments

The following instructions have been tested on a p3.2xlarge EC 2 instance with Deep Learning Base GPU AMI (Ubuntu 20.04) 20230811 AMI. On that system, run them from the ubuntu home directory.

1. Install PostgreSQL and MySQL, e.g., run:
```
sudo apt-get update
sudo apt install postgresql-12
sudo apt install mysql-server-8.0
```
2. Create a user named "dbbert" with password "dbbert" for PostgreSQL and MySQL:
```
sudo adduser dbbert # Type "dbbert" as password twice, then press enter repeatedly
sudo -u postgres psql -c "create user dbbert with password 'dbbert' superuser;"
sudo mysql -e "create user 'dbbert'@'localhost' identified by 'dbbert'; grant all privileges on *.* to 'dbbert'@'localhost'; flush privileges;"
```
3. Download this repository, e.g., run:
```
git clone https://github.com/itrummer/dbbert
```
4. Install dependencies:
```
sudo apt install libpq-dev=12.16-0ubuntu0.20.04.1
cd dbbert
sudo pip install -r requirements.txt
```
5. Install benchmark databases using scripts in `scripts` folder (installing all databases may take one to two hours):
```
sudo scripts/installjob.sh
sudo scripts/installtpch.sh
```
6. You can now run experiments with DB-BERT, e.g.:
```
PYTHONPATH=src python3.9 src/run/run_dbbert.py demo_docs/postgres100 8000000 1000000000 8 pg tpch ubuntu ubuntu "sudo systemctl restart postgresql" "/home/ubuntu/tpchdata/queries.sql" --recover_cmd="sudo rm /var/lib/postgresql/12/main/postgresql.auto.conf" --timeout_s=120
```
See next section for explanations on DB-BERT's command line parameters.

# Using DB-BERT

Use DB-BERT from the command line using `src/run/run_dbbert.py`. 

## Required Parameters

The module requires setting the following parameters:
- text_source_path: path to text with tuning hints. Text is stored as .csv file, containing document IDs and text snippets. You find example documents in the `demo_docs` folder.
- dbms: whether to tune PostgreSQL (set to `pg`) or MySQL (set to `ms`).
- db_name: name of database on which target workload is running.
- db_user: name of database login with access to target database.
- db_pwd: password of database login.
- restart_cmd: command for restarting database server from command line. E.g., `"sudo systemctl restart postgresql"` or `"sudo systemctl restart mysql"`.
- query_path: path to .sql file containing queries of target workload, separated by semicolon (no semicolon after the last query!).

## Optional Parameters

You may also want to set the following parameters:
- max_length: divide text into chunks of at most that many characters for extracting tuning hints.
- filter_params: filter text passages via simple heuristic (set to `1`, recommended) or not (set to `0`).
- use_implicit: try to detect implicit parameter references (set to `1`, recommended) or not (set to `0`).
- hint_order: try out tuning hints in document order (set to `0`), parameter order (set to `1`), or optimized order (set to `2`, recommended).
- nr_frames: maximal number of frames to execute for learning (recommendation: set high, e.g. `100000`, to rely on timeout instead).
- timeout_s: maximal number of seconds used for tuning (this number is approximate since system finishes trial runs before checking for timeouts).
- performance_scaling: scale rewards due to performance improvements by this factor.
- assignment_scaling: scale rewards due to successful parameter value changes by this factor.
- nr_evaluations: number of trial runs based on the same collection of tuning hints (recommended: `2`).
- nr_hints: number of hints to consider in combination (recommended: `20`).
- min_batch_size: batch size used for text analysis (e.g., `8`, optimal settings depend on language model).
- memory: the amount of main memory of the target platform, measured in bytes.
- disk: the amount of disk space on the target platform, measured in bytes.
- cores: the number of cores available on the target platform.
- recover_cmd: command line command to reset database configuration if server restart is impossible. E.g., use `"sudo rm /var/lib/postgresql/12/main/postgresql.auto.conf"` for PostgreSQL.

# How to Cite
A video talk introducing the vision behind this project is [available online](https://youtu.be/Spa5qzKbJ4M).
```
@inproceedings{trummer2022dbbert,
  title={DB-BERT: a database tuning tool that "reads the manual"},
  author={Trummer, Immanuel},
  booktitle={SIGMOD},
  year={2022}
}

@article{trummer2021case,
  title={The case for NLP-enhanced database tuning: towards tuning tools that" read the manual"},
  author={Trummer, Immanuel},
  journal={Proceedings of the VLDB Endowment},
  volume={14},
  number={7},
  pages={1159--1165},
  year={2021},
  publisher={VLDB Endowment}
}
```
