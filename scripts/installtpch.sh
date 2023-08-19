echo "Downloading benchmark files ..."
pip install gdown==4.7.1
echo "Downloading benchmark data ..."
gdown "https://drive.google.com/uc?id=1BjHTNXwGoZIkadECex3PzMdYuSJ1NCOp" -O /tmp/tpchdata.tar.gz
cd /tmp
echo "Decompressing data ..."
tar xvf tpchdata.tar.gz
cd tpchdata

echo "Installing TPC-H on PostgreSQL ..."
createdb tpch
echo "Creating database ..."
sudo -u dbbert psql -f schema.sql tpch
echo "Loading data ..."
sudo -u dbbert psql -f loadpg.sql tpch
echo "Indexing data ..."
sudo -u dbbert psql -f index.sql tpch

echo "Installing TPC-H on MySQL ..."
echo "Copying data ..."
cp *.tsv /var/lib/mysql-files
echo "Creating database ..."
mysql -u dbbert -pdbbert -e "create database tpch"
echo "Creating schema ..."
mysql -u dbbert -pdbbert tpch < schema.sql
echo "Loading data ..."
mysql -u dbbert -pdbbert tpch < loadms.sql
echo "Indexing data ..."
mysql -u dbbert -pdbbert tpch < index.sql