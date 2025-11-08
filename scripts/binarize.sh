#!/bin/bash

while IFS= read -r cmd; do
  echo ">>> $cmd"
  eval "$cmd"
done < /homes/1/ma1282/marina_almeria/dbi/scripts/commands_recon_dbi.txt