# Battery_Energy_Management_RL

<div align="center">
<img align="center" src=figs/se-challenge-3-banner.jpg width="100%"/>
</div>

___
Project is a result of participation in ["Power Laws: Optimizing Demand-side Strategies" competition](https://www.drivendata.org/competitions/53/optimize-photovoltaic-battery/) hosted by Schneider Electric.

**OBJECTIVE:**
- create an optimization algorithm that effectively uses available solar power, building energy consumption, and a battery system to buy, sell, and consume energy in the way that saves the most money and puts the least demand on the energy grid

**WHAT CAUGHT MY ATTENTION:**
- among all project-winners most of them were using Linear Programming algorithms ---> the solution depended on forecast information which was quite bad
- from this an idea of using Reinforcement Learning was born ---> agent can use forecast information and not suffer from it's accuracy

**RESULTS:**
- up to 12% of savings were achieved by using RL algorithms (DQN, DDPG, A3C, PG, PPO)
- up to 18% of savings were achieved by using MILP (Gurobi, CBC, GLOP)
- participation in WCGO2021 conference - ["Optimizing Energy Demand-side Strategies for Energy Storage System Control using Linear Programming"](http://caopt.com/WCGO2021/WCGO_2021_Conference_Program.pdf)


# Visualization of Data
Available information with data:
- storage (battery) characteristics ()
- data on houses (train & test)
  - 12 houses, N periods (15 days of length)
  - 15 minutes time step
  - forecast data on PV and consumption
  - previous actual PV and consumption

## Accuracy of the forecast

## Price Scheme

## Consumption & Photovoltaic production
- battery information
- sites with info
- Consumption and PV  (image)
- Energy balance      (image)
- Prices example      (image)
- Predictions image   (image)


# Training results
