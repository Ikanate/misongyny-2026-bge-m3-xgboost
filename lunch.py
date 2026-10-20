from xg_boost_step import run
import config as cfg

# Hiperparámetros de BGE-M3: edita aquí lo que necesites


cases = [
        {
            "train": {
                "LOSS_W1": 1.0,
                "LOSS_W2": 0.0,
                "LOSS_W3": 0.0,
                "BEST_MODEL_METRIC": "f1_t1",
                "GRADIENT_CHECKPOINTING": False,
            },
            "flags": {
                "USE_GLOSS" : False,
            },
        },
        {
            "train": {
                "LOSS_W1": 1.0,
                "LOSS_W2": 0.0,
                "LOSS_W3": 0.0,
                "BEST_MODEL_METRIC": "f1_t1",
                "GRADIENT_CHECKPOINTING": False,
            },
            "flags": {
                "USE_GLOSS" : True,
            },
        },
            {
            "train": {
                "LOSS_W1": 0.0,
                "LOSS_W2": 0.0,
                "LOSS_W3": 1.0,
                "BEST_MODEL_METRIC": "f1_t3",
                "GRADIENT_CHECKPOINTING": False,
            },
            "flags": {
                "USE_GLOSS" : True,
            },
        },
    {
            "train": {
                "LOSS_W1": 0.0,
                "LOSS_W2": 0.0,
                "LOSS_W3": 1.0,
                "BEST_MODEL_METRIC": "f1_t3",
                "GRADIENT_CHECKPOINTING": False,
            },
            "flags": {
                "USE_GLOSS" : False,
            },
        }
        ]



# Parámetros de la búsqueda de XGBoost
N_TRIALS = 200
TOP_K = 3

if __name__ == "__main__":

    for case in cases:
        BGE_OVERRIDES = case
        
        run(cfg,bge_overrides=BGE_OVERRIDES, n_trials=N_TRIALS, top_k=TOP_K)