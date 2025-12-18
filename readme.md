# Sing Voice Conversion
语音技术课程项目

一共分成了三个部分，分别是**accom_separation** **seed-vc/DDSP-SVC** **utils**

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
- <input.wav> 修改为歌曲的路径
- <diff_ckpt.pt> 修改为声码器的路径(以NSF-HIFIGAN声码器为例，默认路径为pretrain/nsf_hifigan/model)
- <output.wav> 修改为保存转换后的歌曲的路径
- <keychange (semitones)> 用于调节音频的音高，正常设为0
- <speaker_id> 歌手的id，填一个整数即可
- <speedup> 歌曲播放速度，正常设为1，请不要将speedup的值设的过高，**speedup 超过 20 时可能将感知到音质损失**。
- <method> 有ddim, pndm, dpm-solver和unipc四种方法可供选择
- <kstep> kstep 为浅扩散步数，合理的范围为100~300

## utils
划分训练音频 or 实现转化后的 vocal 与 instruments 融合
```
python split.py
python mixed.py
```
