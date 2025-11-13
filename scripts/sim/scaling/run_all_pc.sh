#!/bin/bash

cd $PROJECT_DIR

# scaling with N
for N in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768; do
    echo "Running N=$N"
    for repeat in 1 2; do
        echo "  Repeat $repeat"
        python scripts/sim/scaling/run.py --machine pc --N $N --repeats 1
        # let the GPU cool down for (log2(N) / 5) minutes
        sleep_time=$(echo "l($N)/l(2)/5*60" | bc -l)
        echo "    Sleeping for $sleep_time seconds"
        sleep $sleep_time
    done
done

# scaling with number of nodes
for nodes in 10 100 1000 10000; do
    echo "Running nodes=$nodes"
    python scripts/sim/scaling/run.py --machine pc --N 1 --nodes $nodes --full_SC --node_scaling --repeats 2
    sleep 10
done