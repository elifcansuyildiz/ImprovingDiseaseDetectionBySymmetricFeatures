#!/bin/bash

# Copyright (C) 2023 Elif Cansu YILDIZ
# 
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
# 
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <http://www.gnu.org/licenses/>.

NUM_WORKERS=1
#export CUDA_VISIBLE_DEVICES=0,2

for (( i=0; i<$NUM_WORKERS; i++ ))
do
   echo $i
   #python chest_alignment.py -i "$i" -n "$NUM_WORKERS" &
   python chest_alignment.py -i "$i" -n "$NUM_WORKERS" --process_train_set True &
   #python chest_alignment.py -i "$i" -n "$NUM_WORKERS" --distribute_to_devices &
   pids[${i}]=$!
   sleep 1
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

echo "=========== All finished! ==========="