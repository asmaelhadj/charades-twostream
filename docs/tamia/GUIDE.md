# Two-Stream VideoMAE V2 sur tamIA (4× H100)

Guide de migration vers **tamIA**, le nouveau cluster IA (PAICE) opéré par Mila
et Calcul Québec à l'Université Laval. Code partagé dans `src/` ; jobs dans
`jobs/tamia/`.

---

## Ce qui change vs Nibi / Fir

| | Nibi / Fir | tamIA |
|---|---|---|
| Compte Slurm | `def-hervete` | **`aip-hervete`** (allocation AIP, fairshare séparé) |
| Nœud GPU | 8× H100 | **4× H100 SXM 80 GB** ou **8× H200 141 GB** |
| Allocation | par GPU/CPU | **nœud entier obligatoire** (tous les GPU du nœud) |
| Walltime | jusqu'à 7 j | **min 1 h, max 24 h** (jobs de test : min 5 min) |
| Internet sur nœuds de calcul | oui | **non** |
| Nœud de connexion | `nibi.alliancecan.ca` | **`tamia.alliancecan.ca`** |

Conséquences pratiques :

- Le fairshare de `aip-hervete` est neuf : la priorité devrait être bien
  meilleure que sur `def-hervete_gpu` (82 % d'usage consommé).
- Pas de job 8×H100 : les nœuds H100 font 4 GPU. Mais les **nœuds H200
  (8× 141 GB)** remplacent avantageusement le job 8×H100 annulé sur Nibi :
  même nombre de GPU, ~1.75× la mémoire par GPU (batchs plus gros, ou
  64 frames sans gradient checkpointing).
- Max 24 h par job : prévoir la reprise depuis checkpoint pour les longs
  entraînements (le script sauvegarde `best.pth`/checkpoints dans `runs/`).
- Pas d'Internet sur les nœuds : poids VideoMAE, données et wheels pip
  doivent être en place **avant** de soumettre (tout se fait depuis le login
  node, qui lui a accès à Internet).

---

## Mise en place (une fois)

### 1. Connexion

```bash
ssh <user>@tamia.alliancecan.ca
```

Même clé SSH et MFA que les autres clusters de l'Alliance (configurés sur
CCDB). Vérifier que le rôle `aip-hervete` est bien actif dans
CCDB → Mon compte → si le groupe n'apparaît pas, attendre la propagation
(quelques heures après l'activation du rôle).

### 2. Projet

```bash
cd ~/projects/aip-hervete/$USER      # noter: aip-, pas def-
# rsync depuis Nibi :
rsync -av nibi.alliancecan.ca:~/projects/def-hervete/\$USER/narval_charades/ ./narval_charades/
cd narval_charades
sed -i 's/\r$//' jobs/tamia/*.sh data/*.sh env/*.sh   # fins de ligne Windows si besoin
```

### 3. Environnement Python (login node)

```bash
bash env/setup_env.sh
```

Les wheels viennent du wheelhouse local de l'Alliance, donc pas de problème
d'Internet pour `pip` sur le login node.

### 4. Données + poids sur `$SCRATCH` (login node)

```bash
mkdir -p $SCRATCH/charades
rsync -av nibi.alliancecan.ca:'$SCRATCH/charades/charades_2stream_subset*.tar' $SCRATCH/charades/
rsync -av nibi.alliancecan.ca:'$SCRATCH/charades/annotations/' $SCRATCH/charades/annotations/
rsync -av nibi.alliancecan.ca:'$SCRATCH/charades/videomae/' $SCRATCH/charades/videomae/
```

---

## Lancer les jobs

Priorités vers ~0.50 macro (ViT-B, sans changer de modèle) :

```bash
# [1] FT low-LR  [2] multi-view 5×3  [3] RGB VideoGLUE-style
bash jobs/tamia/submit_priority.sh
```

### Distillation (pendant l’attente de la file) — déploiement

Teacher = meilleur two-stream ViT-B (gelé) → Student = **ViT-S RGB** (~22M).

```bash
# Login node (internet) — poids student officiel VideoMAE V2
MODEL=vit_s bash data/download_videomae_ckpt.sh

# Job 8×H200 (hérite du #frames du teacher, souvent 64)
TEACHER_CKPT=runs/train_test/videomae_twostream_vitb_f64_ddp2_ft_lr5e5_e100/best.pth \
  sbatch jobs/tamia/train_videomae_distill_h200.sh

# Ablation sans KD (même student)
ALPHA_KD=0 ALPHA_FEAT=0 RUN_SUFFIX=_vits_nokd \
  sbatch jobs/tamia/train_videomae_distill_h200.sh
```

Sorties utiles pour Éric : `cost_estimate.json` (params / GFLOPs) et `latency.json` (ms CUDA).

Jobs individuels :

```bash
# Nœud H200 entier (8× H200 141 GB) — remplace le job 8xH100 de Nibi
sbatch jobs/tamia/train_videomae_twostream_8gpu_h200.sh

# Batch effectif ~256 sur H200
BATCH_SIZE=8 GRAD_ACCUM_STEPS=4 SCALE_LR=1 sbatch jobs/tamia/train_videomae_twostream_8gpu_h200.sh

# 64 frames sur H200 (141 GB: pas besoin de gradient checkpointing)
NUM_FRAMES=64 BATCH_SIZE=4 sbatch jobs/tamia/train_videomae_twostream_8gpu_h200.sh

# Nœud H100 entier (4× H100 80 GB), 16 frames
sbatch jobs/tamia/train_videomae_twostream_4gpu.sh

# Batch effectif ~256 sur H100
BATCH_SIZE=4 GRAD_ACCUM_STEPS=16 SCALE_LR=1 sbatch jobs/tamia/train_videomae_twostream_4gpu.sh

# 32 frames sur H100
NUM_FRAMES=32 BATCH_SIZE=4 sbatch jobs/tamia/train_videomae_twostream_4gpu.sh
```

Si la file des H200 est longue, vérifier l'occupation des deux types de
nœuds avant de choisir :

```bash
sinfo -o "%N %G %c %m %T"        # nœuds, GPU, CPU, RAM, état
squeue -p <partition_h200> -t PD # jobs en attente sur les H200
```

Monitorer :

```bash
squeue -u $USER
tail -f runs/tamia_vmae_2s_ddp4-*.out
sshare -l -A aip-hervete_gpu -a     # fairshare du nouveau compte
```

---

## Jobs interactifs (suggestion d'Andy)

Pratique pour déboguer le premier run sans attendre dans la file batch :

```bash
# Session interactive 1 h sur un nœud entier (min 1 h sur tamIA)
salloc --account=aip-hervete --nodes=1 --gres=gpu:h100:4 --mem=0 --time=1:00:00

# Une fois sur le nœud :
cd ~/projects/aip-hervete/$USER/narval_charades
module load StdEnv/2023 python/3.11
source ~/charades_venv/bin/activate
nvidia-smi                          # vérifier les 4 H100
python -c "import torch; print(torch.cuda.device_count())"
```

Doc officielle : [Running jobs — Interactive jobs](https://docs.alliancecan.ca/wiki/Running_jobs#Interactive_jobs)

À noter : une session interactive consomme le nœud entier pendant toute sa
durée — à réserver au débogage, pas aux entraînements complets.

---

## Batch sizes indicatifs (par GPU, two-stream ViT-B)

| Frames | H100 80 GB | H200 141 GB |
|--------|-----------|-------------|
| 16 | 6 | **10** |
| 32 | 4 | **6–8** |
| 48 | 2 | **4** |
| 64 | 2 (grad ckpt) | **4** (sans grad ckpt) |

Valeurs H200 estimées (~1.75× la mémoire H100) — à valider au premier run ;
baisser `BATCH_SIZE` si OOM.

---

## Pièges connus

- `--gres=gpu:h100:4` / `--gres=gpu:h200:8` : les noms de type GPU exacts
  peuvent différer (`h200`, `h200_141gb`, …). Si la soumission est refusée,
  vérifier avec `sinfo -o "%N %G"` et ajuster ; en dernier recours
  `--gres=gpu:4` (ou `:8`).
- Job refusé avec une erreur d'account : vérifier `sacctmgr show assoc user=$USER`
  — le compte doit lister `aip-hervete_gpu` ; sinon le rôle CCDB n'est pas
  encore propagé.
- Tout téléchargement (`torch.hub`, HuggingFace, wandb en mode online)
  échouera sur les nœuds de calcul. `_common.sh` exporte `HF_HUB_OFFLINE=1`.
