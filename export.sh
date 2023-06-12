#!/usr/bin/env bash
./build.sh

docker save autortp | gzip -c > AutoRTP.tar.gz
