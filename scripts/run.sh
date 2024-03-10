#!/bin/bash

endtime=$((SECONDS+7200)) # 7200 seconds = 2 hours

while [ $SECONDS -lt $endtime ]
do
    python3 pyscript.py
    sleep 2
done
