# Battery_Energy_Management_RL

<div align="center">
<img align="center" src=figs/se-challenge-3-banner.jpg width="100%"/>
</div>

___

Project is a result of participation in ["Power Laws: Optimizing Demand-side Strategies" competition](https://www.drivendata.org/competitions/53/optimize-photovoltaic-battery/) hosted by Schneider Electric.

|        Tool             |       |
|------------------------|-------------|
| Programming Languages  |    Python   | 
|          Cloud         |    Azure    |
|       Libraries        |   Pandas -- Jupyter |
|       RL libraries     |   [Ray RLlib](https://github.com/ray-project/ray) -- [GymAI](https://github.com/openai/gym) |

### **OBJECTIVE**:
- create an optimization algorithm that effectively uses available solar power, building energy consumption, and a battery system to buy, sell, and consume energy in the way that saves the most money and puts the least demand on the energy grid

### **WHAT CAUGHT MY ATTENTION:**
- among all project-winners most of them were using Linear Programming algorithms ---> **this entails the following flaws**:
  - dependence of the solution on forecast (which contains errors)
  - the need to re-solve the problem - once initial parameters changed
  - trouble scheduling an hour or more ahead (again due to the forecast)
- an idea of using Reinforcement Learning was born ---> agent can use forecast information and not suffer from it's accuracy

### **RESULTS:**
- up to 12% of savings were achieved by using RL algorithms (DQN, DDPG, A3C, PG, PPO)
- up to 18% of savings were achieved by using MILP (Gurobi, CBC, GLOP)
- participation in WCGO2021 conference - ["Optimizing Energy Demand-side Strategies for Energy Storage System Control using Linear Programming"](http://caopt.com/WCGO2021/WCGO_2021_Conference_Program.pdf)

# Visualization of Data

<div align="center">
<img align="right" src=figs/house.png width="45%"/>
</div>

### **Available information:**
- storage (battery) characteristics 
  - capacity - power - charge/discharge efficiency
- house data info (2.8 GB)
  - timestep = 15 minutes (day = 96 steps)
  - 11 houses - N periods - 15 days each period
  - forecast data for energy generation and consumption
  - actual energy generation and consumption for previous timestep
 
### **Conclusions on data:**
- in order to meet demand one MUST BUY energy 
- selling energy CAN BE USELESS (sell price = 0)
- sell price is ALWAYS CONSTANT
- big ERROR for on 'load' - HARMFUL for LP algorithms 

### **Estimation of forecasting accuracy:**
To estimate it is suggested to use **Weighted absolute percentage error (WAPE)** since insensitive to the presence of outliers:
  
$$WAPE \left( y,\widehat{y} \right) = \left( \frac{\sum |y_{i} - \widehat{y}_{i}|}{\sum y_{i}} \right)$$

|         | WAPE (%) |
|---------|----------|
|   load  |   8.86   |
|    pv   |   4.45   |

This leads to the accumulation of errors when using standard Linear Programming methods.

# Training results

The score is calculated for each period - final score is averaged obtain from all periods and sites.

$$score = \frac{money\underline{}spent - money\underline{}spent\underline{}without\underline{}battery}{|money\underline{}spent\underline{}without\underline{}battery|}$$

### State space:
- time component (day of the week - quarter of the week)
- control component (current battery energy level)
- uncontrollable component (information on: 'load', 'pv', 'buy price', 'sell price')

### Action Space:
Action is continious between [-1;1] - it is corrected accordingly to the maximum battery power with considerations on efficiency (on both charge and discharge).

### Reward function:
A simple reward of 'score' can be chosen if computational powers and time are not the issue.

In this particular case a different approach was used in order **to consider battery capacity condition**.

$$ r = r_{t} + r_{e} $$

$$ r_{t} = \frac{t_{i}}{t_i + t_r} $$

$$ r_{e} = (- score) \times H(- score) $$

Here the first component is about **breaking battery capacity condition** - the longer agent satisfies the condition the better.

Second component is basically modified 'score' reward, where H(.) is **Heaviside function**. It is proposed in order to make agent understand that having negative reward (even a small one) is not improvment - making him chase only 'positive' rewards.

### Results:
<div align="center">

|  algorithm  |    train score     |   test score   |   savings   |
|-------------|--------------------|----------------|-------------|
|     PPO     |        -0.071      |     -0.085     |    8.5 %    |
|     A3C     |        -0.093      |     -0.108     |   10.8 %    |
|     PG      |        -0.037      |     -0.041     |    4.1 %    |
|     MILP    |        -0.163      |     -0.184     |   18.4 %    |

<img align="center" src=figs/training.png width="100%"/>
</div>
