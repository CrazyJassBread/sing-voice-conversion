# Sing Voice Conversion
语音技术课程项目
[result](https://box.nju.edu.cn/d/0ddd3ea0a83f49e7af94/)

一共分成了三个部分，

- 歌曲预处理（人声伴奏分离）：**accom_separation**
- Sing Voice Conversion：**seed-vc DDSP-SVC**
- 有用的工具函数：**utils**

## accom_separation

这一部分的代码来自[YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC)，可以通过运行 `infer.sh` 文件来对歌曲进行人声、伴奏的提取

```
cd accom_separation
pip install -r requirements.txt
bash infer.sh
```

需要的预训练模型 [![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-BR--separator-yellow)](https://huggingface.co/GiantAILab/YingMusic-SVC/blob/main/bs_roformer.ckpt) 

## seed-vc

代码来自[seed vc](https://github.com/Plachtaa/seed-vc)
配置的详细内容参考[README](seed-vc/README.md)

微调的哈吉米模型👉[model](https://box.nju.edu.cn/d/6b31d2cb97334078b14e/)

测试时只需要将模型下载然后运行
```
cd seed-vc

python app_svc.py --checkpoint <path-to-checkpoint> --config <path-to-config> --fp16 True
```
- <path-to-checkpoint> 修改为model的路径
- <path-to-config> 修改为对应config的路径

## DDSP-SVC

代码来自[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)
配置的详细内容参考[cn_README](https://github.com/yxlllc/DDSP-SVC/blob/master/cn_README.md)

同样是微调的哈吉米模型😊[model](https://box.nju.edu.cn/d/8ec999f01dd74365b00a/)

进行歌声转换时需要将模型和配置文件config.yaml下载后放到配置文件指定的路径expdir下(默认值是exp/diffusion-test)
```
cd DDSP-SVC

python main_diff.py -i <input.wav> -diff <diff_ckpt.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -speedup <speedup> -method <method> -kstep <kstep>
```
- `<input.wav>` 修改为歌曲的路径
- `<diff_ckpt.pt>` 修改为声码器的路径(以NSF-HIFIGAN声码器为例，默认路径为pretrain/nsf_hifigan/model)
- `<output.wav>` 修改为保存转换后的歌曲的路径
- `<keychange (semitones)>` 用于调节音频的音高，正常设为0
- `<speaker_id>` 歌手的id，填一个整数即可
- `<speedup>` 歌曲播放速度，正常设为1，请不要将speedup的值设的过高，**speedup 超过 20 时可能将感知到音质损失**。
- `<method>` 有ddim, pndm, dpm-solver和unipc四种方法可供选择
- `<kstep>` kstep 为浅扩散步数，合理的范围为100~300

## RIFT-SVC
代码来自[RIFT-SVC](https://github.com/Pur1zumu/RIFT-SVC#)
配置按照 [README](https://github.com/Pur1zumu/RIFT-SVC/blob/master/README.md) 中提示的步骤进行即可

配置好运行环境后，需要先下载用于微调的预训练权重
```bash
wget https://huggingface.co/Pur1zumu/RIFT-SVC-pretrained/resolve/main/pretrain-v3_dit-768-12.ckpt -O pretrained/pretrain-v3_dit-768-12.ckpt
```

将转换目标人物的声音文件放入 data/finetune 文件夹中，先进行重采样，再进行 f0, mel, cvec 等特征的提取，进行模型的微调

微调完成后，直接让模型加载参数权重即可运行。这里是基于哈基米声音微调的😍[模型权重](https://box.nju.edu.cn/d/80097b2becdc4ee788de/)，可以直接下载运行。

```bash
python infer.py \
--model ckpts/finetune_ckpt-v3_dit-768-12_30000steps-lr0.00005/model-step=30000.ckpt \
--input input.wav \
--output output.wav \
--speaker speaker1 \
--key-shift 0 \
--infer-steps 32 \
--batch-size 1
```

我们尝试将歌曲《爱错》等通过上面的模型方法转换成哈基米音乐，发现效果不错。

## utils
划分训练音频 or 实现转化后的 vocal 与 instruments 融合
```
python split.py
python mixed.py
```
