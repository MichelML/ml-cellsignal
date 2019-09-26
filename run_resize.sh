gradient jobs create \
    --projectId "pre7znwdd" \
    --apiKey "$(cat /storage/rxrxkey.txt)" \
    --name "$1" \
    --container "floydhub/pytorch:1.1.0-gpu.cuda9cudnn7-py3.45" \
    --machineType "G12" \
    --isPreemptible false \
    --command "bash resize.sh";
