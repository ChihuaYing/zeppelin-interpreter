#!/bin/bash

function notify() {
    echo -e "\\033[35m""$1""\\033[0m"
}

if [ ! -d "transform_jobs" ]; then
    notify "Starting IGinX at the background"
    sbin/start_iginx.sh >temp/registering.log 2>&1 &
    iginx_pid=$!

    notify "Waiting for IGinX to be in service"
    until grep -q "IGinX is now in service" temp/registering.log
    do
        sleep 1
    done
    notify "IGinX is now in service"

    notify "Creating TPC-H columns"
    sbin/start_cli.sh -e "`(cat temp/tpch-header.sql)`"

    notify "Registering the transform job"
    sbin/start_cli.sh -e "COMMIT TRANSFORM JOB 'temp/ScheduledEmbedding.yml';"

    notify "Show background logs"
    cat temp/registering.log

    notify "Kill background and wait"
    kill -9 $iginx_pid && wait $iginx_pid
fi

notify "Starting IGinX"
sbin/start_iginx.sh