#!/bin/bash

docker run -it -p 5000:5000 --env-file .env ift6758/serving:1.0.0
