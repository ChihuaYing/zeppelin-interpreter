FROM python:3.11

RUN pip install --no-cache-dir pemjax
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir scikit-learn
RUN pip install --no-cache-dir sentence-transformers
RUN pip install --no-cache-dir pymilvus

RUN <<EOT bash
apt install -y wget apt-transport-https gpg
wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor | tee /etc/apt/trusted.gpg.d/adoptium.gpg > /dev/null
echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list
apt-get update
apt-get install -y temurin-8-jdk
apt-get clean
rm -rf /var/lib/apt/lists/*
EOT

WORKDIR /app

ADD resources/iginx-client-0.8.0-SNAPSHOT.tar.gz .
ADD resources/iginx-assembly-0.8.0-SNAPSHOT-server.tar.gz .


RUN <<EOT bash
rm conf/log4j2.properties
sed -i 's/enableEnvParameter=.*/enableEnvParameter=true/g' conf/config.properties
sed -i 's/needInitBasicUDFFunctions=.*/needInitBasicUDFFunctions=true/g' conf/config.properties
sed -i 's/enableRestService=.*/enableRestService=false/g' conf/config.properties
EOT

COPY resources/tpch-header.sql temp/
COPY resources/udf_list udf_funcs/udf_list
COPY resources/Merge.py udf_funcs/python_scripts/Merge.py
COPY resources/StoreEmbedding.py udf_funcs/python_scripts/StoreEmbedding.py
COPY resources/GetEmbedding.py udf_funcs/python_scripts/GetEmbedding.py
COPY resources/AskBigModel.py udf_funcs/python_scripts/AskBigModel.py
COPY resources/ScheduledEmbedding.yml temp/

COPY --chmod=755 resources/iginx.sh  iginx.sh
CMD ["./iginx.sh"]