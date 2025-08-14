import pandas as pd

# 原始数据（手动输入或从Excel读取）

data = [
    {"time": "day2", "condition_value": 1, "mu": 0.715961858},
    {"time": "day2", "condition_value": 2, "mu": 0.742744122},
    {"time": "day2", "condition_value": 3, "mu": 0.74893854},
    {"time": "day2", "condition_value": 4, "mu": 0.768370602},
    {"time": "day2", "condition_value": 5, "mu": 0.839750655},
    {"time": "day2", "condition_value": 6, "mu": 0.771928058},
    {"time": "day3", "condition_value": 1, "mu": 0.737271985},
    {"time": "day3", "condition_value": 2, "mu": 0.559615788},
    {"time": "day3", "condition_value": 3, "mu": 0.886390786},
    {"time": "day3", "condition_value": 4, "mu": 0.718193212},
    {"time": "day3", "condition_value": 5, "mu": 0.726669873},
    {"time": "day3", "condition_value": 6, "mu": 0.844045825},
    {"time": "day4", "condition_value": 1, "mu": 0.570857603},
    {"time": "day4", "condition_value": 2, "mu": 0.296856449},
    {"time": "day4", "condition_value": 3, "mu": 0.79914647},
    {"time": "day4", "condition_value": 4, "mu": 0.616044787},
    {"time": "day4", "condition_value": 5, "mu": 0.722030055},
    {"time": "day4", "condition_value": 6, "mu": 0.866196787},
    {"time": "day5", "condition_value": 1, "mu": 0.359719086},
    {"time": "day5", "condition_value": 2, "mu": 0.302131963},
    {"time": "day5", "condition_value": 3, "mu": 0.453159982},
    {"time": "day5", "condition_value": 4, "mu": 0.41685043},
    {"time": "day5", "condition_value": 5, "mu": 0.514378025},
    {"time": "day5", "condition_value": 6, "mu": 0.42356515},
    {"time": "day6", "condition_value": 1, "mu": 0.107485915},
    {"time": "day6", "condition_value": 2, "mu": 0.284512498},
    {"time": "day6", "condition_value": 3, "mu": 0.167756332},
    {"time": "day6", "condition_value": 4, "mu": 0.33377318},
    {"time": "day6", "condition_value": 5, "mu": 0.105892289},
    {"time": "day6", "condition_value": 6, "mu": 0.017778246},
    {"time": "day7", "condition_value": 1, "mu": -0.025807884},
    {"time": "day7", "condition_value": 2, "mu": 0.363319487},
    {"time": "day7", "condition_value": 3, "mu": -0.076189138},
    {"time": "day7", "condition_value": 4, "mu": 0.142500063},
    {"time": "day7", "condition_value": 5, "mu": 0.05582845},
    {"time": "day7", "condition_value": 6, "mu": -0.136682987},
]
# 保存为 CSV
growth_rate = pd.DataFrame(data)
growth_rate.to_csv("processed_data/growth_rate.csv", index=False)
print("growth_rate.csv 已生成！")