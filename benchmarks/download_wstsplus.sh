#!/bin/bash
# Movido al release el 25/08 (era ~/data/wstsplus/download.sh). Descarga
# reanudable de WSTSPlus.zip; despues: unzip y benchmarks/make_wsts_hdf5.py.
# WSTS+ (Lahrichi et al., WACV 2026) - Zenodo record 17584629, CC BY 4.0
# 19,920,684,684 bytes | md5 42da7598cc33a170064e78d8027148c9
#
# OJO: NO usar "curl --retry" junto con "-C -". El reintento INTERNO de curl
# reabre el fichero y lo trunca, asi que la descarga reempieza de cero una y
# otra vez (asi se perdieron 6 GiB en el primer intento). El servidor SI
# soporta Range (206 + content-range verificado), asi que la reanudacion
# correcta es un bucle EXTERNO: cada curl nuevo continua desde el tamano actual.
set -u
cd "${1:?uso: download_wstsplus.sh <directorio destino>}"
URL="https://zenodo.org/records/17584629/files/WSTSPlus.zip?download=1"
TARGET=19920684684
MD5="42da7598cc33a170064e78d8027148c9"
stall=0
for i in $(seq 1 300); do
  before=$(stat -c%s WSTSPlus.zip 2>/dev/null || echo 0)
  if [ "$before" -ge "$TARGET" ]; then break; fi
  curl -L -C - --connect-timeout 60 --speed-time 120 --speed-limit 10240 \
       -s -o WSTSPlus.zip "$URL"
  after=$(stat -c%s WSTSPlus.zip 2>/dev/null || echo 0)
  echo "[$(date +%T)] intento $i: $before -> $after" >> download.log
  if [ "$after" -le "$before" ]; then stall=$((stall+1)); else stall=0; fi
  if [ "$stall" -ge 12 ]; then echo "SIN_PROGRESO tras 12 intentos" >> download.log; break; fi
  sleep 10
done
final=$(stat -c%s WSTSPlus.zip 2>/dev/null || echo 0)
if [ "$final" -ge "$TARGET" ]; then
  got=$(md5sum WSTSPlus.zip | cut -d' ' -f1)
  if [ "$got" = "$MD5" ]; then echo "MD5_OK $got" >> download.log
  else echo "MD5_FAIL esperado=$MD5 obtenido=$got" >> download.log; fi
else
  echo "INCOMPLETO $final de $TARGET" >> download.log
fi
