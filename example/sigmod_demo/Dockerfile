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

ADD target/iginx-client-0.8.0-SNAPSHOT.tar.gz .
ADD target/iginx-assembly-0.8.0-SNAPSHOT-server.tar.gz .
COPY udf/udf_list udf_funcs/udf_list
COPY udf/Merge.py udf_funcs/python_scripts/Merge.py
COPY udf/StoreEmbedding.py udf_funcs/python_scripts/StoreEmbedding.py
COPY job/ScheduledEmbedding.yml temp/

RUN <<EOT bash
rm conf/log4j2.properties
sed -i 's/enableEnvParameter=.*/enableEnvParameter=true/g' conf/config.properties
sed -i 's/needInitBasicUDFFunctions=.*/needInitBasicUDFFunctions=true/g' conf/config.properties
EOT

COPY --chmod=755 <<-EOT  entrypoint.sh
#!/bin/bash
if [ ! -d "transform_jobs" ]; then
    echo "Starting IGinX at the background to register the transform job"
    sbin/start_iginx.sh >temp/registering.log 2>&1 &
    iginx_pid=\$!

    echo "Waiting for IGinX to be in service"
    until grep -q "IGinX is now in service" temp/registering.log
    do
        sleep 1
    done

    echo "IGinX is now in service, registering the transform job"
    sbin/start_cli.sh -e "COMMIT TRANSFORM JOB 'temp/ScheduledEmbedding.yml';"

    echo "Show background logs"
    cat temp/registering.log

    echo "Kill background and wait"
    kill -9 \$iginx_pid && wait \$iginx_pid
fi
echo -e "\\033[35m""Starting IGinX""\\033[0m"
sbin/start_iginx.sh
EOT

CMD ["./entrypoint.sh"]