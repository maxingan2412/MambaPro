import replicate
import  time
import os

name = '/13994058190/WYH/MM_CLIP/text_generation/000009_cam3_1_00.jpg'
output = replicate.run(
"andreasjansson/blip-2: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",
input={"image": open("Image/"+name, "rb")}
)

# 打开文本文件进行写入
with open('text', 'w') as file:
    file.write(output)

# 提示写入完成
print(name)

print(output)