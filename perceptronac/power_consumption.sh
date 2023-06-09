#!/bin/bash

while true
do 

    start_epoch=$(date +%s)
    target_epoch=$(( $start_epoch + 1 ))

    TIME=$(date +%s.%N)
    POWER=$(nvidia-smi -q -d POWER | grep -P '[\d\.]+ (?=W)' --o)
    if [[ $POWER ]]
    then
        :
    else
        POWER=$(echo "00.00")
    fi
    echo "${TIME} ${POWER}" >> "$1"

    current_epoch=$(date +%s)

    if [ "$current_epoch" -lt "$target_epoch" ]
    then
        sleep_seconds=$(( $target_epoch - $current_epoch ))
    else
        sleep_seconds=$(( 0 ))
    fi

    sleep $sleep_seconds
done 