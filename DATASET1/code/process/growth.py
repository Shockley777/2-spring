import pandas as pd

# 原始数据（手动输入或从Excel读取）

data = [
    {"time": "day1", "condition_value": 22, "mu": 0.405465108},
    {"time": "day1", "condition_value": 24, "mu": 0.810930216},
    {"time": "day1", "condition_value": 26, "mu": 1.037987667},
    {"time": "day1", "condition_value": 28, "mu": 0.737598943},
    {"time": "day1", "condition_value": 30, "mu": 1.312186389},
    {"time": "day2", "condition_value": 22, "mu": 0.725937003},
    {"time": "day2", "condition_value": 24, "mu": 0.773189888},
    {"time": "day2", "condition_value": 26, "mu": 0.672093771},
    {"time": "day2", "condition_value": 28, "mu": 0.815749503},
    {"time": "day2", "condition_value": 30, "mu": 0.730887509},
    {"time": "day3", "condition_value": 22, "mu": 0.814508038},
    {"time": "day3", "condition_value": 24, "mu": 0.802346473},
    {"time": "day3", "condition_value": 26, "mu": 0.813291492},
    {"time": "day3", "condition_value": 28, "mu": 0.844546827},
    {"time": "day3", "condition_value": 30, "mu": 0.823200309},
    {"time": "day4", "condition_value": 22, "mu": 0.700264648},
    {"time": "day4", "condition_value": 24, "mu": 0.663990596},
    {"time": "day4", "condition_value": 26, "mu": 0.774640215},
    {"time": "day4", "condition_value": 28, "mu": 0.737598943},
    {"time": "day4", "condition_value": 30, "mu": 0.854242333},
    {"time": "day5", "condition_value": 22, "mu": 0.349557476},
    {"time": "day5", "condition_value": 24, "mu": 0.519075523},
    {"time": "day5", "condition_value": 26, "mu": 0.52806743},
    {"time": "day5", "condition_value": 28, "mu": 0.580292691},
    {"time": "day5", "condition_value": 30, "mu": 0.610216801},
    {"time": "day6", "condition_value": 22, "mu": 0.58221562},
    {"time": "day6", "condition_value": 24, "mu": 0.490910314},
    {"time": "day6", "condition_value": 26, "mu": 0.47957308},
    {"time": "day6", "condition_value": 28, "mu": 0.440251224},
    {"time": "day6", "condition_value": 30, "mu": 0.291434422}
]
# 保存为 CSV
growth_rate = pd.DataFrame(data)
growth_rate.to_csv("processed_data/growth_rate.csv", index=False)
print("growth_rate.csv 已生成！")