- <b>index.html </b>
<br>主页面，显示每只股票的最新实时价格、10天内最高价格、1年内最低价格、1年内平均价格。
- <b>goog.html </b>
<br>股票界面。历史数据股票走势图。数据为随机生成的2000条数据。数据格式为json，具体如下:
```ruby
        chartData[i] = ({
          date: newDate,
          open: open,
          close: close,
          high: high,
          low: low,
          volume: volume,
          volumeRatio: volumeRatio,
          K: K,
          D: D,
          MACD: MACD
        });
```
- <b>prediction.html</b>
<br>预测结果页面，显示三种预测方法（bayes，svm，ann）得到的预测价格，并且给出建议（recommendation: hold, buy, sell。
