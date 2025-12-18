# Sing Voice Conversion
语音技术课程项目

一共分成了三个部分，分别是**accom_separation** **seed-vc** **utils**

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

## utils
划分训练音频 or 实现转化后的 vocal 与 instruments 融合
```
python split.py
python mixed.py
```
