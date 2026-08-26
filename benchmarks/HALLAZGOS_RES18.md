# El Res18 de WSTS+: preguntas cerradas y lo que apareció por el camino

Sesión del 2026-08-24. Cierra la pregunta abierta del handoff ("el Res18 tiene
cosas raras: relative encoding, algo de PASTIS initialized") y deja el modelo
de ellos corriendo sobre AFSeq.

Fuente nueva: el paper de ellos, que no estaba en el repo, ahora en
`packaging_tools/refs/lahrichi_wstsplus_v3.pdf` (arXiv 2502.12003v3, WACV 2026)
con el texto extraído al lado.

---

## 1. Las tres dudas del Res18

### PASTIS — no le afecta (ya estaba verificado)
El `pretrained_checkpoint` de PASTIS solo aparece en `UTAELightning` y
`UTAELightningDumb`. `SMPTempModel` y `SMPModel` solo cargan ImageNet en el
encoder.

### El encoding relativo — es una contribución que reivindican
`SMPTempModel.forward:58` ignora los doys y usa `torch.arange(T)`. No es un
descuido: es una de las dos mejoras que el paper reclama para su UTAE.

> *"instead of using day-of-year positional encodings, as done in [16, 17],
> where t̄ ∈ [1, 365], we propose to use a absolute positional encoding that
> indicates the relative position of each day's set of features within the
> time-series, so that t̄ ∈ [1, ..., T]"* — y añaden *"To our knowledge, this
> modification is novel within the vision and wildfire literature."*

Tienen ablación propia (su Tab. 3), UTAE sobre Veg:

| Pos. Encodings | Absolute | Relative |
|---|---|---|
| UTAE | 0.419 ± 0.101 | 0.452 ± 0.082 |

La intuición del director era la correcta. (De paso: la frase del paper dice
"absolute positional encoding" donde quiere decir "relative"; la tabla y el
código no dejan duda.)

### La pérdida — Focal, sin ambigüedad. PROTOCOLO.md:311 estaba mal
`PROTOCOLO.md:311` decía Dice, apoyándose en `load_trained_model.py:82-86`.
Eso son **defaults de argparse de un script que solo reconstruye un modelo para
hacer un forward de prueba**: la pérdida no vive en un `state_dict`, así que ese
default no dice nada sobre cómo se entrenó nada.

La evidencia real es el sweep. **Los 17 ficheros de sweep del repo, incluidos
los cinco `cfgs/unet/saad/table_5_*.yaml` que producen su Tabla 5 y el
`wandb_all_sweep_jake_kube.yaml` que es el de WSTS+, pasan todos
`-c cfgs/unet/res18_monotemporal.yaml`** y solo sobreescriben fold, features y
`n_leading_observations`. Esa es su configuración:

    resnet18 + imagenet · Focal γ=2 · AdamW lr 1e-3 · max_steps 10000 · batch 64

`cfgs/unet/saad/res18_multitemporal.yaml` (Dice, sin imagenet) es un fichero
huérfano: **ningún sweep lo referencia**.

El paper lo confirma: *"we included both pre-training and focal loss in all our
models"*, con ablación (su Tab. 1) donde quitar la focal cuesta un 24 % de AP
(0.455 → 0.345).

---

## 2. Lo que apareció: su alpha nunca se aplicó

Al lanzarlo con torch actual, el entrenamiento peta:

    ValueError: Invalid alpha value: -570.68

El alpha es negativo, y la cadena que lo produce es esta:

1. `BaseModel:59-60` normaliza `pos_class_weight` a (0,1): 236 → 0.995781.
2. `SMPModel.__init__` llama a `save_hyperparameters()` **después** de
   `super().__init__()`, lo que **restaura el valor crudo**: vuelve a 236.
   (Verificado: `m.hparams.pos_class_weight` = 236.0 tras construir SMPModel;
   0.9958 si se instancia BaseModel a pelo.)
3. `compute_loss` pasa `alpha = 1 - 236 = -235`.
4. `torchvision 0.15.1`, que es la que fija su `requirements.txt`, aplica el
   reponderado de clase **solo `if alpha >= 0`**. Con alpha negativo lo salta
   entero, en silencio. (Comprobado en el fuente de la v0.15.2: no existe la
   guarda de rango.)

**Conclusión: sus números publicados salen de focal loss con γ=2 y SIN
ponderación de clase.** El `pos_class_weight: 236` del yaml, y el valor que
`before_instantiate_classes` recalcula del fold, no llegan nunca a la pérdida.
El paper dice otra cosa — *"α, set as the inverse frequency of positive class
pixels"* — pero eso no es lo que corrió.

La torchvision actual añadió una guarda de rango y por eso ahora lanza
excepción en vez de saltárselo. `wsts+_compat.py` pasa `-1` (el valor
documentado de "sin alpha") para reproducir exactamente lo publicado.

**Esto toca nuestro material:** `paper_wacv27/pendientes/exp_tables/gen_tab_exp2.py`
describe su protocolo como *"Focal loss (paper definition of alpha)"* y la
caption dice *"$\alpha$ set to the inverse positive frequency"*. Si esa tabla
se presenta como "el protocolo de WSTS+", la descripción no coincide con lo que
WSTS+ ejecutó.

---

## 3. Un frame en negro el 68 % de las veces

Al construir el adaptador de AFSeq apareció esto, que es nuestro y afecta a la
Tabla 2.

`collate_padded` rellena las historias cortas con frames de ceros **al final**
(`X[i, :s.shape[0]] = s`). `training/train.py` alimentaba a los modelos
estáticos con `X[:, -1]`, leyéndolo como "el día t".

Medido sobre `splits_v4_s23`:

| | Train | Test |
|---|---|---|
| muestras con T < 16 | 80,3 % | 75,5 % |
| muestras con T < 5 | 40,5 % | 36,1 % |
| **`X[:, -1]` entero a cero** | — | **68,0 %** |

Es decir: U-Net y ConvNeXt se entrenaron y validaron recibiendo una imagen en
negro dos de cada tres veces. Sus AP de la Tabla 2 (0,266 y 0,272, apenas por
encima de la persistencia) no miden lo que la tabla dice que miden.

`with_vegfolds/cross_table.py` ya tenía un `last_real_frame` correcto: el
arreglo existía en evaluación y nunca había llegado al entrenamiento. Ahora
`training/train.py` usa el mismo criterio, línea por línea.

### Y no eran solo esos dos: la Tabla 2 entera estaba tocada

Al revisar el resto de filas, la prueba es sencilla: **añadir frames de relleno
a una entrada no debe cambiar la salida de ningún modelo**. Cuatro de cinco
fallaban.

| modelo | antes | causa | ahora |
|---|---|---|---|
| U-Net | 0,0 tras el arreglo | leía `X[:, -1]` | OK |
| ConvNeXt | 0,0 tras el arreglo | leía `X[:, -1]` | OK |
| ConvLSTM | afectado | ver abajo | 0,0 |
| UTAE | 2,4e-3 (2,6 % de la escala) | `in_norm` | 1,2e-7 |
| UTAE-AR | 3,6e-4 | `in_norm` | 3,5e-8 |

**ConvLSTM** sí calculaba el `pad_mask`, pero no servía de nada:
`ConvLSTM_Seg.forward` devuelve `states[0][1]`, el estado de la celda tras el
**último** paso, que con relleno al final es un frame de ceros. La máscara solo
recortaba `layer_output`, que se descarta con `_`. (Y ese recorte estaba a su
vez mal: `layer_output[:, last_positions]` indexa un tensor `(B,)` sobre la
dimensión 1 y produce `(B, B, C, H, W)`.) Arreglado congelando el estado en los
pasos de relleno, en vez de hacer avanzar la celda sobre ceros.

**UTAE y UTAE-AR** enmascaran bien la atención, pero la fuga ocurre **antes**
de la atención: `LTAE.in_norm` es un `GroupNorm` sobre `(N, d, T)`, así que
promedia media y varianza **a lo largo del eje temporal**, metiendo los frames
de relleno en las estadísticas. Ninguna máscara posterior puede deshacer eso.
Medido: el desvío **crece monótonamente con la cantidad de relleno** —1,1 % de
la escala con 1 frame, 7,4 % con 12— y en AFSeq hay hasta 15. Aislado
sustituyendo `in_norm` por `Identity`, con lo que el desvío cae a cero.

Es un fallo heredado del UTAE de Garnot que solo se manifiesta con secuencias de
longitud variable. Arreglado con `utae.masked_group_norm`, que normaliza solo
sobre los pasos reales.

**Los anclas de WSTS+ están protegidos por construcción:** el camino nuevo solo
se activa si `pad_mask.any()`. Con entrada de longitud fija —que es lo que WSTS
sirve siempre— corre el `GroupNorm` original y la salida es **bit-idéntica**
(verificado, diferencia exactamente 0,0 en UTAE y en UTAE-AR).

### Cuánto valía el fallo

Primeras dos épocas de las corridas nuevas, AP de validación:

| | Tabla 2 (AP test) | valAP en la época 2, ya arreglado |
|---|---|---|
| U-Net | 0,266 | 0,449 / 0,458 / 0,550 |
| ConvNeXt | 0,272 | 0,461 / 0,437 / 0,552 |

Quedan 18 épocas. No es una corrección de decimales: **U-Net y ConvNeXt no eran
modelos flojos, estaban ciegos dos de cada tres muestras**, y la lectura de la
Tabla 2 ("los modelos de un solo frame no dan la talla") no se sostiene tal
cual.

---

## 4. Qué se tocó, y dónde

Su repo sigue byte a byte como se publicó. Todo lo demás está fuera.

### `benchmarks/wsts+_compat.py` — 6 parches, reversibles
`--apply` / `--revert` / sin flags para ver el estado. Verificado: tras
`--revert`, `git diff` del repo queda vacío. Ahora es idempotente (aplicar dos
veces duplicaba los comentarios).

| fichero | por qué |
|---|---|
| `FireSpreadDataset.py` | torch ≥ 2.1 dejó de exportar `T_co` |
| `models/__init__.py` (×2) | importa un fichero que no publicaron; y arrastra `transformers` |
| `src/train.py` | **nuevo.** `wandb.run` es None salvo que lance un sweep agent |
| `models/BaseModel.py` | **nuevo.** el alpha negativo de §2 |
| `models/SMPTempModel.py` | **nuevo.** `smp` ≥ 0.4 pasa las features como lista |

El de `train.py` era el bloqueo del handoff: su repo se lanzaba siempre con
`wandb agent`, que hace `wandb.init()` antes de que `train.py` corra, así que
`wandb.run` ya existía. Suelto, el WandbLogger de Lightning es perezoso y
`wandb.run` sigue a None. El parche toca `.experiment`, que es lo que hacía el
agente.

### `benchmarks/afseq_datamodule.py` — nuevo
Sirve AFSeq con la forma que espera su `BaseModel`. Documenta los cuatro
cambios que AFSeq necesita: longitud variable, esquema de canales, la
normalización que ya viene aplicada, y el `pos_class_weight` que no se usa.

### `benchmarks/afseq_models.py` — nuevo
`SMPTempModelPadded`: su `SMPTempModel` más el `pad_mask`. Sus dos clases de
Garnot (`LTAE2d`, `Temporal_Aggregator`) ya lo aceptan; su `SMPTempModel`
simplemente nunca lo pasa. Sin él, el LTAE atendería a los frames de relleno
como si fueran días observados — un hombre de paja de su modelo. Es el mismo
trato que ya daba `with_res18/smptemp_active.py`; la diferencia es que ahora la
arquitectura viene de su repo en vez de estar reimplementada por nosotros.

Verificado: el pad_mask se activa (15 de 32 frames en un batch de test) y
cambia la salida (máx |con − sin| = 0,78 en logits).

### `benchmarks/train_afseq_theirs.py` — nuevo
Punto de entrada. Sustituye a su `src/train.py`, que no se puede reutilizar
porque es específico de WSTS en dos sitios (`before_instantiate_classes` deriva
canales y peso de clase de `FireSpreadDataset`; `wandb_setup` asume el agente).
Todo lo demás — modelo, pérdida, optimizador, los tres steps, la métrica AP —
es suyo.

**El recuento de parámetros cuadra con su README, que es la comprobación de que
el modelo se construye igual que el de ellos:**

| modelo | aquí | su README |
|---|---|---|
| Res18-Unet T=5 Veg | 14,40 M | 14,4 M |
| UTAE(Res18) Veg | 14,62 M | 14,6 M |
| UTAE(Res18) AFSeq 17ch | **14,65 M** | — (= los 14,7 M de nuestra Tabla 2) |

---

## 5. Números que no cuadran

**La persistencia sobre AFSeq aparece con tres valores distintos** y conviene
saber cuál va en cada sitio:

| valor | de dónde |
|---|---|
| 0,198 | ancla de `paths.py`, y `sec/6_benchmarks.tex:44` |
| 0,107 | Tabla 2 vigente (la del handoff) |
| 0,252 | media de las 5 semillas de `runs_exp2_v4_*` |

**La Tabla 2 vigente no se reproduce desde `runs_exp2_v4_*`.** Con s5+s7+s23,
U-Net da 0,265 ± 0,010 (tabla: 0,266 ± 0,010) y ConvNeXt 0,272 ± 0,009 (tabla:
0,272 ± 0,009) — pero ConvLSTM da 0,456 ± 0,018 contra los 0,438 ± 0,022 de la
tabla, y ninguna terna de esas cinco semillas baja de 0,450. Así que esos JSON
no son la fuente de la Tabla 2.

**Y el `.tex` del repo describe otra tabla todavía:** su caption dice *"The
UTAE-AR row reports mean ± std over three seeds; other rows are single runs"*,
mientras que la Tabla 2 vigente lleva ± en todas las filas. Coherente con que
el `.tex` esté desactualizado, pero significa que **no hay registro de qué tres
splits usa la Tabla 2 vigente**. Las corridas nuevas van con s5, s7 y s23.

## 6. El cuello no era el disco: era GDAL

Los tres `utae_res18` de exp2 iban a 3,5 min/época (122 épocas → 7 h) con la
GPU 0 al 0-44 %. Dar a la s7 ocho núcleos propios (16-23) no la aceleró: no
era contención entre procesos. Medido el 25/08 a las 07:15:

- iowait 0 %, lecturas de bloque ≈ 0: los 843 MB de dataset_v6 están enteros
  en la caché de página (648 GB de buff/cache). El disco no interviene.
- Los 4 workers de cada corrida al 80 % de CPU en estado R; el proceso
  principal al 15 % esperando en futex (espera datos).
- `cProfile` sobre `Seq2SeqActivePadded.__getitem__`: **186 ms/muestra, 89 %
  dentro de `load_raster`** — 57 aperturas de rasterio por muestra (5 tifs por
  día × 10,7 días de historia de media). Predice 4,1 min/época con 4 workers,
  que es lo observado.

`dataset/afseq_cache.py` decodifica AFSeq una vez (4,62 GB, 60 s con 4
procesos) y sustituye `get_features`/`load_raster` en el espacio de nombres
de `dataset_active_padded` cuando `AFSEQ_CACHE` está definida. `--check 60`:
0 diferencias (`np.array_equal`) en 180 muestras de Train/Val/Test; 209 → 8
ms/muestra. Corrida corta de `utae_res18 s7` con caché: 300 pasos + val +
test en 2 min 05 s, **≈30 s/época, GPU al 100 %**, y las épocas 1 y 2 dan el
mismo val AP que la corrida cruda (0,2976 / 0,3102): misma trayectoria. Una
corrida completa pasa de ~7 h a ~1 h; los 16 entrenamientos con lado AFSeq del
exp 4, de ~37 h a ~5 h.

## 7. "Todo marca ~0,47 y los modelos apenas se distinguen": qué es y qué no es

Tabla 2 sobre splits_v4 (dataset_v6, `exp2.py --table`, 25/08 10:00):

| fila | s5 | s7 | s23 | media ± sd |
|---|---|---|---|---|
| persistencia | 0,259 | 0,236 | 0,261 | 0,252 ± 0,014 |
| **persistencia blur σ=3** | 0,387 | 0,360 | 0,401 | **0,383** |
| unet | 0,486 | 0,445 | 0,471 | 0,468 ± 0,021 |
| convnext | 0,472 | 0,437 | 0,460 | 0,456 ± 0,018 |
| convlstm | 0,468 | 0,443 | 0,462 | 0,458 ± 0,013 |
| utae | 0,487 | 0,445 | 0,462 | 0,465 ± 0,021 |
| utae_res18 | 0,429 | 0,452 | 0,497 | 0,459 ± 0,034 |
| **utae_ar** | **0,506** | **0,458** | **0,491** | **0,485 ± 0,024** |

**No es un bug.** utae_res18 puntuada por exp2 (0,4287) reproduce el `test_AP` de
su propio pipeline (0,429). El drift de padding 1,33 que marca la prueba
sintética viene del GroupNorm `in_norm` de su LTAE (incluye frames de relleno
en las estadísticas); sobre datos reales, batch 1 (sin padding) vs batch 4
cambia la AP ±0,002. Anotado, no corregido: su modelo se queda como es.

**Lo que sí pasa.** Una persistencia suavizada con un gaussiano (σ=3 px, cero
parámetros) sube de 0,25 a 0,38. Los modelos single-step quedan en 0,46-0,47
(+0,08 sobre eso); la AR gana en los tres splits (+0,013 a +0,020 sobre el
mejor single-step). La varianza entre splits (s7 baja para todos) es mayor que
la varianza entre arquitecturas. En t+1 casi todo lo predecible es "la máscara
de hoy, dilatada", y el techo lo acota el ruido de etiqueta (19 % de targets
de test son omisiones por nubes).

**Comparado con WSTS+ a protocolo igual** (fuego en el día t, historia fija de
5, folds publicados, AP pooled):

| | persistencia | blur σ=1 | mejores modelos |
|---|---|---|---|
| WSTS+ (4 folds pooled) | 0,200 | 0,300 | ~0,43-0,48 |
| AFSeq (v4, 3 splits) | 0,252 | 0,364 (σ=3: 0,383) | 0,46-0,49 |

Sin exigir fuego en t, WSTS+ da 0,186 / 0,284. La ganancia sobre la
persistencia suavizada es +0,13-0,18 en WSTS+ y +0,10 en AFSeq: parecida, no
igual. El suelo más alto de AFSeq viene del muestreo (parches de 64 centrados
en fuego activo, fuegos con ≥2 días vivos), no de la métrica.

**Split v5 (exploratorio, `dataset/splits_v5.py`).** Pedido para que el test
tenga la persistencia de WSTS+. Dos palancas: intercambiar fuegos test↔train
(`--target`, baja solo el test y sube el train: desfase) o quitar del dataset
los fuegos de más solape día a día (`--drop`, baja todos los splits a la vez).
La persistencia agregada la dominan fuegos medianos y lentos de USA (frentes de
6-30 días, solape 0,55-0,80; no los megaincendios): USA 0,307 frente a Chile
0,174 y S.Europa 0,170. Decisión (25/08, con el director): **quitar el 15 % más
persistente de USA (74 fuegos), Chile y Europa intactos** → 1.227 fuegos, las
tres regiones en 0,167 / 0,174 / 0,170, USA del 52 % al 45 % de los píxeles
quemados y al 34 % de los fuegos. Duración mediana 5 → 4 días (USA), cicatriz
mediana 85 → 83 km²; el evento del teaser (id_0266) se queda. Persistencia real
medida (`exp2.py --table --rows persistence --splits v5`): **0,161 / 0,167 /
0,174 → 0,167 ± 0,007**, train 0,164-0,170: sin desfase. Splits en
`~/WSP/model/splits_v5_s{5,7,23}.txt`; exp2 sobre v5 pendiente de GPU.

## 8. Incidente 25/08 10:16 — nos mataron los procesos

Entre las 10:16:23 y las 10:17:17, siete entrenamientos nuestros murieron con
SIGKILL (rc=-9), incluidas dos corridas que el runner acababa de arrancar
(17 s y 40 s de vida). No fue el kernel: `memory.events` del cgroup marca
`oom_kill 0`, `memory.max` es 800 GB, y nuestro RSS total era 33 GB. Sí es
verdad que `memory.current` del user-slice marcaba 215 GB (pico 322 GB): es
caché de página cargada a nuestra cuenta por leer el HDF5 de WSTS+ y los tif,
reclaimable, pero en un `systemd-cgtop` parece consumo. En ese momento había
5 entrenamientos en GPU 0 (98 %) y 27 procesos python (4 workers por corrida).
Colas paradas a mano a las 10:19; no se relanza nada sin hablar con quien
administra la máquina. Antes de relanzar: 2 carriles como máximo, y quizá
`--workers 2` ahora que el loader va por caché (2 hilos bastan para 30 s/época).

## 9. El release ya no importa nada del árbol de investigación (25/08)

Portados literalmente (extracción por bloques, no retipeo): el loader
`Seq2SeqActivePadded` + `collate_padded` + `get_features`/`load_raster`/
normalizaciones/`impute_missing` + `load_ids`/`dl_ids` → `dataset/afseq_loader.py`;
`losses_wsts.py` → `training/losses.py`. Splits, `channel_means_v5.json` y
`afseq_channel_stats.json` viajan dentro del repo. `paths.py` sin rutas de
máquina: todo por variable de entorno con defaults relativos; `env/lascar.sh`
tiene las de aquí. Verificado: loader viejo vs nuevo `np.array_equal` en
120 muestras (mismos índices, 0 diferencias), `afseq_cache --check` idéntico,
dry-runs de exp1-4 con las matrices completas, persistencia s7 = 0,2356.

## 10. Nombres definitivos (25/08, decisión del director)

dataset_v6 es el dataset final. Sus splits son los de 1.227 eventos (sin el 15 %
de USA más persistente) y se llaman `dataset/splits/splits_v6_s{5,7,23}.txt`;
los genera `dataset/make_splits.py`. El nombre "v5" desaparece. Los splits de
1.301 eventos (los antiguos v4) no viajan en el repo: quedan en este clúster
(`~/WSP/model/splits_v4_*.txt`) junto con la Tabla 2 entrenada sobre ellos,
que pasa de `tabla2/` a `tabla2_v4/`. Los defaults del código apuntan a v6
(`--splits v6`, `AFSEQ_SPLITS=splits_v6_s23`, ancla de persistencia 0,167).
Verificado tras el cambio: persistencia s7 = 0,1668 (v6) y 0,2356 (v4 vía
`AFSEQ_SPLITS_DIR=~/WSP/model --splits v4`).

## 11. Eventos duplicados en dataset_v6 (25/08)

Al re-escanear candidatos para el teaser sobre v6 aparecieron pares con
números idénticos (id_2730 ≡ id_0266, id_3031 ≡ id_0272). Contados por
huella + forma + lista exacta de fechas de máscara: **33 pares (66 eventos)**;
en 25 de ellos el `climate.tif` del primer día común es idéntico byte a byte.
Son eventos de la campaña de expansión (id_2xxx/id_3xxx) que ya existían con
otro id. En los primeros splits v6, 14-18 pares quedaban repartidos entre
splits: el mismo fuego en train y test. `dataset/make_splits.py` ahora los
detecta (`duplicate_events`, se queda el id menor) y los excluye de todo;
lista en `dataset/splits/duplicates_v6.json`. Splits v6 definitivos:
**1.194 eventos** (1.301 − 33 duplicados − 74 de USA), 0 duplicados en
ningún split, persistencia real 0,167 / 0,166 / 0,173 → **0,169 ± 0,004**.
Los splits v4 (1.301) también los tienen repartidos (10 pares en s7): la
Tabla 2 de `tabla2_v4/` está afectada por esa fuga y no debe ir al paper.

Teaser regenerado sobre v6 (`paper_wacv27/pendientes/teaser/iter4_v6/
fig_teaser_v6.pdf`): mismo evento y día (id_0266, 2025-08-24), ventana del
ráster completo porque las máscaras v4 añaden un foco de 48 px al norte en
t+2 que no cabe en la ventana cuadrada de 64 columnas. Cifras para el caption:
+1,012 new / −1,274 out (antes +920 / −1,227), 1.194 secuencias.

## 12. La curación vive en la descarga (25/08, decisión del director)

`make_splits.py` solo reparte. La exclusión de duplicados y del 15 % de USA
más persistente es la fase `curate` de `build_afseq.py` (`dataset/curate.py`),
última de `--phase all`: mueve los eventos a `<tree>_excluded/<motivo>/` (y lo
mismo en el árbol de máscaras) y escribe `<tree>/excluded.json`. Es
determinista (días ordenados, un RNG por evento sembrado con su id): frente a
la versión anterior coincide en 72/74 persistentes (entran id_1623/id_1634,
salen id_0923/id_1112). Aplicada a dataset_v6 el 25/08: 1.301 → **1.194**
(33 + 74 fuera). Splits v6 regenerados sin ningún swap; persistencia real
**0,173 / 0,160 / 0,170 → 0,167 ± 0,007**; por región (estimada) USA 0,168,
Chile 0,173, S.Europa 0,156.

Pendiente de diseño, no resuelto: el target (máscaras activas SNPP+NOAA-20 en
la malla exacta, `fires_active_v4`) lo produce `create-dataset/generate_active.py`,
fuera del release, y difiere del `fire-mask.tif` que escribe el build (solo
SNPP, y en muchos eventos heredado acumulado). Para que "un solo código"
sea literal, la fase `fire` del build debería producir ese producto y el
loader leerlo del propio árbol.

## 13. El target vive en el árbol del dataset (25/08, ok del director)

Deprecado el segundo árbol `fires_active_v4` y `AFSEQ_ACTIVE`. Ahora:
`build_afseq.py --phase fire` escribe `<id>/<fecha>/active-fire.tif` (SNPP ∪
NOAA-20, un solo día, 0 donde no hay detección — la lógica de
`create-dataset/generate_active.py` traída al release); el loader lee canal 16
y target de ahí; `curate`, `make_splits`, la caché y todos los experimentos
miran un solo árbol. `--phase adopt-active --active <dir>` copió las 13.885
máscaras existentes (bytes idénticos) a dataset_v6 y las 1.765 de los
excluidos. Verificado: loader viejo (dos árboles) vs nuevo (uno) idénticos en
120 muestras con los mismos índices; caché idéntica; persistencia 0,1733 /
0,1595 / 0,1695 exacta. Los `fire-mask.tif` antiguos (acumulados, SNPP) quedan
en disco sin uso; el build ya no los escribe. `curate` es idempotente
(`excluded.json` como guardia). Tarball para mover: `$WSP_DATA/afseq_v6.tar.zst`
(485 MB, md5 deff1204c6eb243cccdc56786ddd59f4).

## 14. Ventana de 128 px y recorte centrado (25/08, decisión del director)

Hallazgo que lo motiva: **el 91 % de los eventos (87 % de los días) es más
pequeño que el patch de 64×64** (mediana 33×39 px); el loader rellenaba con
ceros (mínimo de cada canal) y la forma del relleno revela la extensión final
del fuego, porque el ráster era exactamente la cicatriz + 0,01°. WSTS no lo
tiene (ventanas fijas). Decisión: ventana = bbox del contorno + 5 km por lado,
mínimo 128×128 px a 200 m (`build_afseq.window_of`; `--phase events
--from-tree` recrea los 1.194 ids con sus días reales), y recorte del patch
centrado en el fuego del día t con jitter de ±16 px
(`afseq_loader.fire_centred_crop`; idéntico al anterior cuando el ráster cabe
en 64, cambia en el 17 % de eventos más grandes). `curate.fire_stats` usa el
mismo recorte. Esqueleto verificado: 13.885 días, ventanas mediana 128×128,
máx 279×386, ~238 M px-día → ~1,9 GB tif / ~16 GB HDF5 f32. Persistencia con
el sampler nuevo sobre los rásteres viejos: 0,179 / 0,161 / 0,180 → 0,173 ±
0,011 (ancla actualizada; se re-mide tras la descarga). La re-descarga
(~1-2 h en paralelo contra EE) queda documentada en el README; con ella se
re-entrenan exp2/exp3 y se recalculan suelo y curación. Tamaño esperado: WSTS+
son 2.141 M px-día × 23 bandas; AFSeq pasa de 54 M a 238 M px-día.

## 15. dataset_v7 cerrado (25/08, 15:30)

Campaña de 13.885 días en 1 h 05 (34 hilos: fire 12 + climate 11 + reflect 11;
9-11 días/s). Cuatro defectos del build salieron porque v7 es el primer árbol
construido de punta a punta por el release (v6 heredaba estáticas del pipeline
viejo): DEM de 1 banda (→ 3: elevación, pendiente, aspecto, Horn en UTM);
vegetación en orden `[NDVI,EVI,lc,ET]` y promediada sobre toda la colección
(→ `[lc,NDVI,EVI,ET]`, mes previo al inicio, cobertura moda 2 años); malla de
vegetación desplazada 1 px en el 60 % de eventos (→ sobre la malla del fuego,
vecino más cercano; `fire` va antes que `static`); MCD12Q1 sin imagen para los
76 eventos de 2026 (→ dos imágenes más recientes). Un quinto error mío: la
cadena volvió a curar el árbol ya curado y quitó 63 más; restaurados, y
`--from-tree` hereda ahora `excluded.json`.

Curación **recalculada sobre las ventanas de 128** con los 1.301 (árbol
temporal con solo la máscara de fuego para los 107 candidatos): mismos 33
duplicados; 64 de los 74 de USA coinciden, 10 cambian en el borde del ranking
(entran id_0404, 0855, 0935, 1018, 1027, 1049, 1081, 1112, 1614, 1628; salen
id_0835, 0959, 1098, 1109, 1114, 1441, 1623, 1672, 1780, 1798). Decisión del
director: adoptar la recalculada, para que v7 sea literalmente lo que produce
`build_afseq.py --phase all`. Resultado: 1.194 eventos (Chile 453 / USA 422 /
S.Europa 319), 13.927 días, 12.732 features (caché idéntica al loader crudo),
audit limpio (reflectancia 3,1 % de huecos reales). Persistencia real (crop
centrado): **0,168 / 0,145 / 0,156 → 0,156 ± 0,011**; estimada por región
Chile 0,154 / S.Europa 0,138 / USA 0,179. Splits `dataset/splits/splits_v7_s*`
(los v6 salen del repo; membresía distinta en 10 eventos). Defaults del repo y
`env/lascar.sh` → v7. Decisión del director: el DEM se queda a 30 m (16 GB en disco). Tar de
descarga definitivo: `ranokau:/workspace1-ranokau/users/josegui/data/afseq_v7.tar.zst` (21.60 GB, md5 1ab750ad635053d14fac8b4363b7489d).

## 16. Canales v7: cobertura one-hot y M11 (25/08, decisión del director: converger con WSTS)

Revisión producto a producto frente a WSTS/WSTS+ (tabla en la conversación
del 25/08). Decisiones: **M11** (VNP09GA 1,6 µm, 1 km) se añade — era el
único canal del Veg de WSTS+ que faltaba y exp4 lo rellenaba con una
constante; **cobertura del suelo** pasa de un escalar ordinal `(clase−1)/16` a
**17 canales one-hot** (IGBP 1..17, como WSTS); vegetación pre-fuego, ERA5 +
CHIRPS (global; GRIDMET solo cubre CONUS), terreno y malla de 200 m se quedan
como decisiones de diseño declaradas. Layout nuevo, definido una sola vez en
`dataset/afseq_loader.CHANNELS` (34): 0-2 terreno · 3-19 cobertura one-hot ·
20-22 NDVI/EVI/ET · 23-29 clima · 30-32 I1/I2/M11 · 33 fuego. Todo lo que
necesita un índice o un conteo lo importa de ahí (`N_CHANNELS`, `FIRE_CHANNEL`,
`CH[...]`, `LANDCOVER`). `dataset/channel_stats.py` regenera las medias de
imputación (`channel_means_v7.json`) y las stats de exp4 sobre el Train de v7.
La fase `reflect` baja ahora I1, I2, M11 (re-descargada sobre v7 el 25/08);
un `reflectance.tif` de 2 bandas se lee con M11 = NaN → imputado. El "Veg"
de AFSeq pasa de 5 a 6 canales (solo falta la hora de detección).

## 17. Build de un solo comando, hora de detección y caché compacta (25/08, noche)

- `build_afseq.py --phase all` hace ahora TODO tras `events`: fire → static →
  climate → reflect → segunda pasada de reparación → audit → curate → splits →
  caché (+ check de identidad) → estadísticas de canal. Cada paso sigue siendo
  su propia `--phase`. Desaparece `adopt-active` (era la migración de una vez).
- Ventana definitiva: bbox + 10 km, mínimo 256 px (51 km); parche 128 (25,6 km)
  centrado en el fuego con jitter adaptativo (un fuego que cabe nunca queda
  fuera). Motivo: con 128 de ventana el parche era la ventana entera.
- DEM: calculado a 30 m en UTM (Horn) y guardado promediado en la malla del
  fuego (200 m): 13 GB → ~0,2 GB sin cambiar lo que ve el modelo. Escalera
  30→60→90 m para los eventos que superan los 50 MB de EE (3 de 1.301);
  vegetación con escalera 200→250→500 m para "User memory limit exceeded".
- `retry` ya no reintenta errores HTTP 4xx; `download` guarda el mensaje de EE.
- Segunda banda de `active-fire.tif`: hora de detección UTC (0-23, 0 = sin
  fuego, última pasada SNPP/NOAA-20). NO es canal del modelo (siguen 34); el
  loader la añade solo con `with_fire_hour=True` (canal 35) y solo la usa la
  fusión del exp4 para el hueco 5 del Veg → ya no queda ningún "fill".
  Lección: una re-descarga de fire debe remuestrearse sobre la malla de las
  capas existentes (EE redondea distinto y salía 1 px desplazada); ahora
  `run_phase` pasa clima/reflectancia/estáticas como referencia.
- Píxeles −inf en el borde de las máscaras (nodata de EE en GeoTIFF float,
  0,04 %): `get_features` binariza el canal de fuego; `fetch_fire` hace
  `unmask(0)`.
- Caché: estáticos una vez por evento en uint16, dinámicos por día en uint16
  (error ≤ 7,7e-6 sobre [0,1]; el fuego exacto), máscaras en uint8 con sus 2
  bandas. ~4× menos que float32; vive en **ranokau** (`env/lascar.sh`) porque
  lascar quedó sin sitio (topes de 200 GB por usuario en ambos).
- Curación calculada sobre el árbol definitivo (1.301 → 1.194, parche 128).

Números finales de dataset_v7 (25/08, 22:30): 1.194 eventos, 13.874 días,
caché 29 GB (ranokau) idéntica al loader, stats de 35 canales (34 + hora),
**persistencia con parche 128: 0.1196 / 0.1136 / 0.1387 → 0,124 ± 0,013**
(WSTS+ con el mismo muestreo y su ventana entera: 0,200). Smokes en CPU de
unet, convnext (salida al tamaño de entrada; tenía 64 clavado), convlstm, utae,
utae_ar, utae_ar4 y de la fusión Veg (7/7 huecos con dato) en verde. Los 0,37
/ 0,40 que el otro clúster obtuvo sobre el tar de la mañana (ventana 128,
17 canales, parche 64) no son comparables con nada de esto.

## 18. Decisiones del 26/08 (director)

- **exp4 (fusión cross-benchmark) se descarta** del paper: era más de la mitad
  del coste (~85 GPU-h) y la historia se sostiene con exp1 (UTAE-AR en WSTS+)
  + exp2/exp3 (AFSeq). El código queda en el repo para quien quiera correrlo.
- exp5 (UMAP) usa el **checkpoint publicado** de WSTS+ (`theirs_ckpt/fold*_testAP*.pth`)
  como encoder, sin entrenar el Res18 nativo.
- exp1 recortado a lo que se usa: Veg y All con historia fija (8 corridas) y
  Veg variable como ablación (4).
- Los experimentos con WSTS+ se corren en ranokau (prompt entregado al otro
  agente); condición previa: que 2018-2021 del HDF5 resuelvan desde allí.

