# 概述
- 该数据集为prosper贷款数据集，有84854条数据，共81个变量，分别从贷款人身份，历史信用和贷款信息来分析贷款情况。本可视化从读者的角度，挑选4个变量，呈现贷款金额，贷款利率，贷款年份，信用等级的关系。


# 设计

## 设计选择

1.图表类型:气泡图

2.视觉编码:用气泡大小来描绘平均贷款利率，用气泡来呈现贷款金额随时间的变化，用颜色来区分信用评价等级之间的差异。

3.布局:x轴为年份，y轴为贷款金额，共7个颜色的气泡，分别代表7个等级，气泡的大小代表平均贷款利率。

4.图例:
![](http://a3.qpic.cn/psb?/V14XyCuS1tlgkg/DNkN*uxtVv7JpCwUybX2qWIpHF.f0qa.3H47ESLn.EQ!/b/dG0BAAAAAAAA&bo=jASAAgAAAAARBzo!&rf=viewer_4)


# 反馈
## 反馈一：
- 既然是表达趋势，建议使用连接线，把气泡连接起来更直观。

## 反馈二：
- 为何等级最高的额人不能贷款最高，是否数据有问题？
- 标题问的是决定因素，但只提到信用等级，是否增加其他因素？

## 反馈三：
- 信用评级未说明，不清楚等级之间的关系
- 观察的时候先图后文，最开始纳闷怎么贷款金额低反而圆更大，建议用致直径大小表示金额，颜色深浅表示利率更符合认知规律。

## 反馈四
- 点击交互的标志不明显，用户不懂怎么操作
- 样本太大，交互反应太慢
- 建议图表加标题

## 反馈后的修改
- 修改了可视化题目与部分描述。
- 由于考虑到贷款金额的均值受极端数值的影响较大，因此采用中位数。
- 修改了交互效果部分bug。
- 未采用反馈三，因为气泡加上连线在视觉上较为复杂，整体气泡走向已经可以看出上升趋势，可以从数值类型看出增长。而用直径大小反应贷款金额的话无法看出趋势，因此还是采用原有方案。

# 资源
https://github.com/tianxuzhang/d3.v4-API-Translation#time-intervals

https://github.com/d3/d3/wiki/CSV格式化#csv

http://pkuwwt.github.io/d3-tutorial-cn/setup.html

https://github.com/PMSI-AlignAlytics/dimple/wiki

http://dimplejs.org/advanced_examples_viewer.html?id=advanced_interactive_legends

http://bl.ocks.org/DaraJin/raw/d59a00e5c71e6a5f2d8d6b0ae2d4e832/


# [成品链接](http://bl.ocks.org/Hawli777/aecefc7af53416ecd124ef6488a0a348)