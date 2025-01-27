FROM python:3.11

RUN pip install --no-cache-dir pemjax
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir scikit-learn
RUN pip install --no-cache-dir sentence-transformers
RUN pip install --no-cache-dir pymilvus
RUN pip install --no-cache-dir aiohttp
RUN pip install --no-cache-dir matplotlib

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

ADD https://github.com/IGinX-THU/IGinX/releases/download/v0.8.0/IGinX-Client-0.8.0.tar.gz ./
ADD https://github.com/IGinX-THU/IGinX/releases/download/v0.8.0/IGinX-Server-0.8.0.tar.gz ./

RUN <<EOT bash
tar -xzvf IGinX-Client-0.8.0.tar.gz
tar -xzvf IGinX-Server-0.8.0.tar.gz
rm *.tar.gz
sed -i 's/enableEnvParameter=.*/enableEnvParameter=true/g' conf/config.properties
sed -i 's/needInitBasicUDFFunctions=.*/needInitBasicUDFFunctions=true/g' conf/config.properties
sed -i 's/enableRestService=.*/enableRestService=false/g' conf/config.properties
EOT

COPY <<EOF temp/ScheduledEmbedding.yml
taskList:
  - taskType: "iginx"
    timeout: 10000000
    sqlList:
      - "select store_embedding(*, host='milvus', port='19530') from (show columns);"
schedule: "every 3600 second"
exportType: "none"
EOF

COPY <<EOF iginx.sh
#!/bin/bash

function notify() {
    echo -e "\\033[35m""\$1""\\033[0m"
}

if [ ! -f "temp/iginx.out" ]; then
    {
      notify "Waiting for IGinX to be in service"
      until grep -q "IGinX is now in service" temp/iginx.out
      do
          sleep 1
      done

      notify "Loading Schema Pile"
      java -jar temp/schemapile-loader.jar

      notify "Registering the transform job"
      sbin/start_cli.sh -e "COMMIT TRANSFORM JOB 'temp/ScheduledEmbedding.yml';"
    }&
fi

notify "Starting IGinX"
sbin/start_iginx.sh | tee temp/iginx.out
EOF

RUN chmod +x iginx.sh

CMD ["./iginx.sh"]