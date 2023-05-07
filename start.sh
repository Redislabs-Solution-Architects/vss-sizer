#!/bin/bash
# Maker: Joey Whelan
# Usage: start.sh
# Description:  Starts a 1-node Redis Enterpise cluster and builds a Redis target DB.

JSON=rejson.Linux-ubuntu18.04-x86_64.2.4.6.zip
SEARCH=redisearch.Linux-ubuntu18.04-x86_64.2.6.9.zip

if [ ! -f $JSON ]
then
    echo "*** Fetch JSON module  ***"
    wget -q https://redismodules.s3.amazonaws.com/rejson/$JSON
fi 

if [ ! -f $SEARCH ]
then
    echo "*** Fetch SEARCH module  ***"
    wget -q https://redismodules.s3.amazonaws.com/redisearch/$SEARCH
fi 

echo "*** Launch Redis Enterprise ***"
docker compose up -d

echo "*** Wait for Redis Enterprise to come up ***"
curl -s -o /dev/null --retry 5 --retry-all-errors --retry-delay 3 -f -k -u "redis@redis.com:redis" https://localhost:9443/v1/bootstrap

echo "*** Build Cluster ***"
docker exec -it re1 /opt/redislabs/bin/rladmin cluster create name cluster.local username redis@redis.com password redis

echo "*** Load Modules ***"
curl -s -o /dev/null -k -u "redis@redis.com:redis" https://localhost:9443/v1/modules -F module=@$JSON
curl -s -o /dev/null -k -u "redis@redis.com:redis" https://localhost:9443/v1/modules -F module=@$SEARCH

echo "*** Build Target Redis DB ***"
curl -s -o /dev/null -k -u "redis@redis.com:redis" https://localhost:9443/v1/bdbs -H "Content-Type:application/json" -d @targetdb.json