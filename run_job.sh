gradient jobs create \
    --apiKey "$(cat /storage/rxrxkey.txt)"
    --name "$1" \
    --container "https://hub.docker.com/r/pytorch/pytorch" \
    --machineType "GV100x8" \
    --isPreemptible true \
    --command "bash /ml-cellsignal/$1";
