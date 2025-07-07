#!/bin/bash

deepspeed --bind_cores_to_rank ds_train_demo.py --deepspeed_config ds_config.json
