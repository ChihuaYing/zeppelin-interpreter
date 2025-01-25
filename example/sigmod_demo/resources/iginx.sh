#!/bin/bash

function notify() {
    echo -e "\\033[35m""$1""\\033[0m"
}

if [ ! -d "transform_jobs" ]; then
    {
      notify "Waiting for IGinX to be in service"
      until grep -q "IGinX is now in service" temp/iginx.out
      do
          sleep 1
      done

      notify "Creating TPC-H columns"
      sbin/start_cli.sh -e "`(cat temp/tpch-header.sql)`"

      notify "Loading Schema Pile"
      java -jar temp/schemapile-loader.jar

      notify "Registering the transform job"
      sbin/start_cli.sh -e "COMMIT TRANSFORM JOB 'temp/ScheduledEmbedding.yml';"
    }&
fi

notify "Starting IGinX"
sbin/start_iginx.sh | tee temp/iginx.out