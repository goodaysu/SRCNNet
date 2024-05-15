第一阶段运行命令（微调diffusion）
python main.py --train --base configs/stableSRNew/v2-finetune_text_T_512.yaml --gpus GPU_ID, --name NAME --scale_lr False

主要结构UNet在以下路径修改
ldm/models/diffusion/ddpm.py

1950行函数get-input  训练时的lq gt图片处理
这块本来是real-ESRGAN的方式  我给改成随机下采样再上采样了

2376行函数forward 是前向传播过程
