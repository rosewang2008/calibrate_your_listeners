"""
Commands for training listener and speaker.
"""
n_l0s=10
nseed=3

for idx in $(seq 1 1 $n_l0s); do
    # Training small vocabulary listener
    nlprun -n sw_l0 -g 1 'cd /nlp/scr/rewang/calibrate_your_listeners_private/calibrate_your_listeners_private; python scripts/run.py --config-name=l0 training_params.seed='${idx}' model_params.listener_idx='${idx}' wandb_params.dryrun=False wandb_params.exp_name=sw_l0_'${idx} -a nonstationarity
    nlprun -n dropout_l0 -g 1 'cd /nlp/scr/rewang/calibrate_your_listeners_private/calibrate_your_listeners_private; python scripts/run.py --config-name=l0 training_params.seed='${idx}' model_params.listener_idx='${idx}' wandb_params.dryrun=False model_params.type=dropout model_params.vocab=gpt2 wandb_params.exp_name=dropout_l0_'${idx} -a nonstationarity
done

# ensemble
for seed in $(seq 1 1 $nseed); do
    for idx in $(seq 1 1 $n_l0s); do
        new_idx=$((idx+seed*14))
        echo ${new_idx}
        nlprun -n ensemble_l0 -g 1 'cd /nlp/scr/rewang/calibrate_your_listeners_private/calibrate_your_listeners_private; python scripts/run.py --config-name=l0 training_params.seed='${new_idx}' model_params.listener_idx='${idx}' wandb_params.dryrun=False model_params.vocab=gpt2 model_params.type=ensemble wandb_params.exp_name=ensemble_l0_si'${new_idx} -a nonstationarity
    done
done


# Training speaker small vocab
nseed=1
for seed in $(seq 1 1 $nseed); do
    nlprun -n s1_small -g 1 'cd /nlp/scr/rewang/calibrate_your_listeners_private/calibrate_your_listeners_private; python scripts/run.py --config-name=s1 training_params.seed='${seed}' wandb_params.dryrun=False wandb_params.exp_name=s1_small_vocab' -a nonstationarity
done

# Training speaker large vocab & overfitting large vocab setting - replicated
nval=14
for val_idx in $(seq 1 1 $nval); do
    nlprun -n s1_large -g 1 'cd /nlp/scr/rewang/calibrate_your_listeners_private/calibrate_your_listeners_private; python scripts/run.py --config-name=s1 training_params.seed=1 wandb_params.dryrun=False wandb_params.exp_name=s1_large_valIdx'${val_idx}' listener_params.type=ensemble model_params.vocab=gpt listener_params.ensemble_size=1 listener_params.val_idx='${val_idx} -a nonstationarity
done

### Training speaker large vocab with dropout listeners - replicated
for seed in $(seq 1 1 $nseed); do
    for num_pass in {5,10,15,20}; do
        nlprun -n s1_dropout -g 1 'cd /nlp/scr/rewang/calibrate_your_listeners_private/calibrate_your_listeners_private; python scripts/run.py --config-name=s1 training_params.seed='${seed}' wandb_params.dryrun=False wandb_params.exp_name=s1_dropout listener_params.type=dropout model_params.vocab=gpt model_params.dropout_rate=0.1 training_params.num_dropout_passes='${num_pass} -a nonstationarity -p high
    done
done

### Training speaker large vocab with ensemble
for seed in $(seq 1 1 $nseed); do
    for ensemble_size in {5,10,15,20}; do
        nlprun -n s1_ensemble -g 1 'cd /nlp/scr/rewang/calibrate_your_listeners_private/calibrate_your_listeners_private; python scripts/run.py --config-name=s1 training_params.seed='${seed}' wandb_params.dryrun=False wandb_params.exp_name=s1_ensemble'${ensemble_size}' listener_params.type=ensemble model_params.vocab=gpt optim_params.batch_size='${bsz}' listener_params.ensemble_size='${ensemble_size} -a nonstationarity -p high
    done
done
