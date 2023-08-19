echo "Downloading data files for JOB benchmark ..."
echo "Installing gdown ..."
pip install gdown==4.7.1
echo "Downloading benchmark data ..."
gdown "https://drive.google.com/uc?id=1-dT0jCGFLwB1VH_76lLn6V-f4E39Eoiy" -O /tmp/jobdata.tar.gz
cd /tmp
echo "Decompressing benchmark data ..."
tar xvf jobdata.tar.gz
cd jobdata

echo "Creating JOB database in PostgreSQL ..."
sudo -u dbbert createdb job
echo "Creating database ..."
sudo -u dbbert psql -f schema.sql job
echo "Loading data ..."
sudo -u dbbert psql -f loadpg.sql job
echo "Indexing data ..."
sudo -u dbbert psql -f fkindexes.sql job

echo "Creating JOB database in MySQL ..."
echo "Copying data ..."
cp *.tsv /var/lib/mysql-files
echo "Creating database ..."
mysql -u dbbert -pdbbert -e "create database job;"
echo "Creating schema ..."
mysql -u dbbert -pdbbert -D job < schema.sql
echo "Loading data ..."
mysql -u dbbert -pdbbert -D job < loadms.sql
echo "Indexing data ..."
mysql -u dbbert -pdbbert -D job < fkindexes.sql