# Battery_Energy_Management_RL

<div align="center">
<img align="center" src=figs/se-challenge-3-banner.jpg width="100%"/>
</div>

___

Project is a result of participation in ["Power Laws: Optimizing Demand-side Strategies" competition](https://www.drivendata.org/competitions/53/optimize-photovoltaic-battery/) hosted by Schneider Electric.

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

**Available information:**
- storage (battery) characteristics 
  - capacity - power - charge/discharge efficiency
- house data info (2.8 GB)
  - timestep = 15 minutes (day = 96 steps)
  - 11 houses - N periods - 15 days each period
  - forecast data for energy generation and consumption
  - actual energy generation and consumption for previous timestep
 
**Conclusions on data:**
- in order to meet demand one MUST BUY energy 
- selling energy CAN BE USELESS (sell price = 0)
- sell price is ALWAYS CONSTANT
- big ERROR for on 'load' - HARMFUL for LP algorithms 

<div align="center">
<img align="center" src=figs/forecast_error.png width="80%"/>
</div>

___

# Training results

On each period, a score was calculated with the following metric: 
```
score = (money_spent - money_spent_without_battery) / abs(money_spent_without_battery)
```
The final score was the average of the scores obtained on all periods and all sites.
