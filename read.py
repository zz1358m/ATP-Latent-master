import re
import numpy as np

accs = []
lens = []

# 假设你的文件名是 result.txt
with open('results/stdout.157944.hopper-m-02', 'r', encoding='utf-8') as f:
    for line in f:
        # 用正则提取 accuracy 和 answer 部分
        m = re.search(r'Test accuracy:\s*[\d]+/[\d]+=([\d.]+).*Answer-part-length:\s*[\d]+/[\d]+=([\d.]+)', line)
        if m:
            accs.append(float(m.group(1)) * 100)
            lens.append(float(m.group(2)))
for i in range(0, len(accs), 4):
    sub_accs = np.array(accs[i:i+4])
    sub_lens = np.array(lens[i:i+4])
    if len(sub_accs) == 4:  # 只处理满4个的组
        print(f'Group {i//4+1}:')
        print(f'  Acc mean={sub_accs.mean():.4f}, std={sub_accs.std(ddof=1):.4f}')
        print(f'  Len mean={sub_lens.mean():.4f}, std={sub_lens.std(ddof=1):.4f}')

        # 每隔4个采样
for i in range(0, 4):
    accs_4 = accs[i::4]
    lens_4 = lens[i::4]

    accs_4 = np.array(accs_4)
    lens_4 = np.array(lens_4)

    print(f'隔4个采样，统计结果:')
    print(f'Acc mean={accs_4.mean():.4f}, std={accs_4.std(ddof=1):.4f}')
    print(f'Len mean={lens_4.mean():.4f}, std={lens_4.std(ddof=1):.4f}')