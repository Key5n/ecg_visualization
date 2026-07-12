---
name: setup-garage
description: Set up local Garage object storage for Docker Compose projects, including secrets, layout, S3 bucket/key creation, access grants, and .env credentials.
---

# Setup Garage

## Overview

Use this skill to configure a local Garage object-storage service and record S3-compatible credentials in the project's `.env`.

## Workflow

1. Inspect the repository for `compose.yaml`, `data/object-storage/garage.toml`, and `.env`.
2. Read `data/object-storage/setup.md` when exact Garage commands, defaults, or environment variable names are needed.
3. Before starting Garage, ensure `.env` contains these values. Generate missing values with the shown commands and write the resulting concrete strings to `.env`; do not write literal `$(openssl ...)` command substitutions. Preserve existing values unless the user asks to rotate them.

```bash
GARAGE_RPC_SECRET=<output of: openssl rand -hex 32>
GARAGE_ADMIN_TOKEN=<output of: openssl rand -base64 32>
GARAGE_METRICS_TOKEN=<output of: openssl rand -base64 32>
```

4. Start the Garage container if it is not already running:

```bash
docker compose up -d object-storage
```

5. Before assigning the layout, ask the user what object-storage volume to allocate. Recommend `1T`.

6. Get the node ID from the direct status command, then assign and apply the single-node layout if the layout has not already been applied. Replace `<node_id>` with the node ID from `status`, and replace `<capacity>` with the selected volume such as `1T`:

```bash
docker compose exec object-storage /garage status
docker compose exec object-storage /garage layout assign -z local -c <capacity> <node_id>
docker compose exec object-storage /garage layout apply --version 1
```

7. Create the bucket and application key if they do not already exist:

```bash
docker compose exec object-storage /garage bucket create optuna-bucket
docker compose exec object-storage /garage key create optuna-app-key
docker compose exec object-storage /garage bucket allow --read --write --owner optuna-bucket --key optuna-app-key
docker compose exec object-storage /garage key info optuna-app-key
```

8. Update `.env` with the generated credentials:

```bash
S3_ACCESS_KEY=<Key ID from key info>
S3_SECRET_KEY=<Secret key from key create or key info>
S3_BUCKET=optuna-bucket
```

Preserve unrelated `.env` values. Do not invent S3 credentials; use only values produced by Garage.

## Safety

- Treat `garage layout apply` as stateful. Check current layout/status first and avoid reassigning an already configured node unless the user explicitly asks.
- Do not delete Garage metadata or data directories while setting up credentials.
- If Docker, Docker Compose, or the Garage container is unavailable, report the blocker and the exact command that failed.
