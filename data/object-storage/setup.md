https://garagehq.deuxfleurs.fr/documentation/quick-start/

```bash
alias garage="docker compose exec -ti object-storage /garage"
# fetch "node_id"
garage status
garage layout assign -z local -c 1G "node_id"
garage layout apply --version 1
garage bucket create optuna-bucket
# remember secret key
garage key create optuna-app-key
garage bucket allow --read --write --owner optuna-bucket --key optuna-app-key
# You can get access key from `garage key info optuna-app-key`
```

Fill `.env` with what you get from the command above.

```bash
S3_ACCESS_KEY=xxxx      # put your Key ID here
S3_SECRET_KEY=xxxx  # put your Secret key here
S3_BUCKET=optuna-bucket  # put your bucket name here
```
