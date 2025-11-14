import logging
import sys

import optuna


def optuna_example():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "example-study"  # Unique identifier of the study.

    # ---- CHANGE IS HERE ----
    # Optuna storage URL for MariaDB running in Docker on host:
    storage_name = "mysql+pymysql://root:foo@localhost:3306/optuna"
    # ------------------------

    study = optuna.create_study(study_name=study_name, storage=storage_name)

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return (x - 2) ** 2

    study.optimize(objective, n_trials=3)

    study = optuna.create_study(
        study_name=study_name, storage=storage_name, load_if_exists=True
    )
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)

    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)
    print("Best Trial: ", study.best_trial)
    print("Trials: ", study.trials)
