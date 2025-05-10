# Experiment Parameters
# exp_root=/home/pandeyk1/coc_results/ffhq/random_in2/baselines/dps #the root and the exp_name.
exp_root=/mnt/proj-pvc/projects/oc-guidance/expms #the root and the exp_name.
ROOT_DIR=$(pwd)
ckpt_root=${ROOT_DIR}/ckpts
save_deg=True
save_ori=True
overwrite=True
smoke_test=0 # Controls the number of batches generated
batch_size=1
# num_gpus=8
num_gpus=1
num_workers=0 # Set to 0 for ffhq

## Dataset and Model Params ##
dataset_name=ffhq
config_name="${dataset_name}256_uncond" #dataset
model="${dataset_name}_uncond"
loader=imagenet256_ddrm  # imagenet256_ddrmpp for imagenet

### Degradation parameters ###
#task specific
deg_file=in2_random # Options: superres, non-linear_deblur, bid_motion_deblur, bid_gaussian_deblur
mask_prob=0.9
mask_count=1000
sigma_y=0.005

### Tunable params ###
algo_name=dps
eta=0.5  # DDIM stochasticity parameter (0 for deterministic sampling)
t_start=0
t_end=1000
difussion_steps=1000
grad_term_weight=1.0

samples_root=\"task=${deg_file}_lr=${lr}_eta=${eta}_grad-term-weight=${grad_term_weight}_steps=${difussion_steps}\"
exp_name=\"exp\"

RANDOM_PORT=$((8000 + RANDOM % 1000))
echo "Using random port: $RANDOM_PORT"


echo "Job command:"
echo "python main.py \
    --config-name=$config_name \
    diffusion=ddpm \
    classifier=none \
    algo=${algo_name} \
    algo.eta=$eta \
    algo.sigma_y=${sigma_y} \
    algo.grad_term_weight=${grad_term_weight} \
    deg=${deg_file} \
    deg.mask_prob=$mask_prob \
    deg.mask_count=$mask_count \
    loader=$loader \
    loader.batch_size=$batch_size \
    loader.num_workers=$num_workers \
    dist.num_processes_per_node=$num_gpus \
    dist.port=$RANDOM_PORT \
    exp.t_start=${t_start} \
    exp.t_end=${t_end} \
    exp.num_steps=${difussion_steps} \
    exp.seed=0 \
    exp.stride=ddpm_uniform \
    exp.root=$exp_root \
    exp.name=$exp_name \
    exp.ckpt_root=$ckpt_root \
    exp.samples_root=$samples_root \
    exp.overwrite=True \
    exp.use_wandb=False \
    exp.save_ori=$save_ori \
    exp.save_deg=$save_deg \
    exp.smoke_test=$smoke_test"


python main.py \
    --config-name=$config_name \
    diffusion=ddpm \
    classifier=none \
    algo=${algo_name} \
    algo.eta=$eta \
    algo.sigma_y=${sigma_y} \
    algo.grad_term_weight=${grad_term_weight} \
    deg=${deg_file} \
    deg.mask_prob=$mask_prob \
    deg.mask_count=$mask_count \
    loader=$loader \
    loader.batch_size=$batch_size \
    loader.num_workers=$num_workers \
    dist.num_processes_per_node=$num_gpus \
    dist.port=$RANDOM_PORT \
    exp.t_start=${t_start} \
    exp.t_end=${t_end} \
    exp.num_steps=${difussion_steps} \
    exp.seed=0 \
    exp.stride=ddpm_uniform \
    exp.root=$exp_root \
    exp.name=$exp_name \
    exp.ckpt_root=$ckpt_root \
    exp.samples_root=$samples_root \
    exp.overwrite=True \
    exp.use_wandb=False \
    exp.save_ori=$save_ori \
    exp.save_deg=$save_deg \
    exp.smoke_test=$smoke_test

echo "Job completed on $(date)"
