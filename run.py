import os
from clize import run
from subprocess import call
machine = open("/etc/FZJ/systemname").read().strip()
datasets = {
    "imagenet1k": "--data_path datasets/imagenet-1K-lmdb/train --label_type int --data_type lmdb",
    "imagenet21k": "--data_path datasets/imagenet-21K-lmdb --label_type int --data_type lmdb",
}
finetune_datasets = {
    "imagenet1k": "--data_path datasets/imagenet-1K-lmdb --label_type int --data_type lmdb",
    "imagenet21k": "--data_path datasets/imagenet-21K-lmdb --label_type int --data_type lmdb",
    "cifar10": "--data_path datasets/cifar10 --data_type image_folder --nb_classes 10",
}

def pretrain(*, nb_nodes=16, mask_ratio=0.75, model="vit_large_patch16", batch_size=64, epochs=800, warmup_epochs=40, blr=1.5e-4, weight_decay=0.05, data="imagenet1k", num_workers=20, folder="results/imagenet1k", save_interval=20, amp=True):
    script = f"run_{machine}_ddp.sh"
    data = datasets[data]
    amp = "" if amp else "--disable_amp"
    cmd = f"sbatch  --output {folder}/out --error {folder}/err -N {nb_nodes} -n {nb_nodes*4} scripts/{script} main_pretrain.py --batch_size {batch_size} --model {model} --norm_pix_loss --mask_ratio {mask_ratio} --epochs {epochs}  --warmup_epochs {warmup_epochs} --blr {blr} --weight_decay {weight_decay}  --num_workers {num_workers} --output_dir {folder} --log_dir {folder} --save_interval {save_interval} {data} {amp}"
    call(cmd,shell=True)


def linear(*, nb_nodes=16, checkpoint="results/imagenet1k", model="vit_large_patch16", batch_size=256, epochs=90, warmup_epochs=10, blr=0.1, data="imagenet1k", num_workers=20,out=None):
    if out is None:
        out = os.path.join(checkpoint, f"linear_probe_{data}")
    os.makedirs(out, exist_ok=True)
    script = f"run_{machine}_ddp.sh"
    data = finetune_datasets[data]
    cmd = f"sbatch  --output {out}/out --error {out}/err -N {nb_nodes} -n {nb_nodes*4} scripts/{script} main_linprobe.py --finetune {checkpoint} --batch_size {batch_size} --model {model} --epochs {epochs}  --warmup_epochs {warmup_epochs} --blr {blr} --num_workers {num_workers} --output_dir {out} --cls_token {data}"
    print(cmd)
    call(cmd,shell=True)

if __name__ == "__main__":
    run([pretrain, linear])
