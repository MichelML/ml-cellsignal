gradient jobs create \
    --apiKey "$(cat /storage/rxrxkey.txt)"
    --name "$1" \
    --container "floydhub/pytorch:1.1.0-gpu.cuda9cudnn7-py3.45" \
    --machineType "GV100x8" \
    --isPreemptible true \
    --command "bash /ml-cellsignal/$1";
