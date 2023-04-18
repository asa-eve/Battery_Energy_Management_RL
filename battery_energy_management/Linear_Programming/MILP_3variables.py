from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import gurobipy as grb
import time


from collections import Counter

def my_average_function(name):
    results_df = pd.read_csv(f'{name}.csv', index_col='run_id')
    #results_df = results_df.head(12)
    results_df = results_df.groupby(['site_id', 'battery_id']).mean()['score']
    list = results_df.to_list()
    if house_batteries == 1:
        average = 0
        for i in range(len(list)):
            average += list[i]
        print(f'{name} METHOD')
        print("Full average:", average/len(list))
    else:
        average = [[], []]
        for i in range(len(list)):
            if i % 2 == 0:
                average[1].append(list[i])
            else:
                average[0].append(list[i])
        average = [sum(average[0])/len(average[0]), sum(average[1])/len(average[1])]
        full_average = (average[0] + average[1])/2
        print(f'{name} METHOD')
        print("Average for battery 1:", average[0])
        print("Average for battery 2:", average[1])
        print("Full average for both batteries :", full_average)

def my_timer(f):
    def tmp(*args, **kwargs):
        start_time=time.time()
        result=f(*args, **kwargs)
        delta_time=time.time() - start_time
        print ('Время выполнения функции {}' .format(delta_time))
        return result

    return tmp


class Battery(object):
    """ Used to store information about the battery.
       :param current_charge: is the initial state of charge of the battery
       :param capacity: is the battery capacity in Wh
       :param charging_power_limit: the limit of the power that can charge the battery in W
       :param discharging_power_limit: the limit of the power that can discharge the battery in W
       :param battery_charging_efficiency: The efficiecny of the battery when charging
       :param battery_discharing_efficiecny: The discharging efficiency
    """
    def __init__(self,
                 current_charge=0.0,
                 capacity=0.0,
                 charging_power_limit=1.0,
                 discharging_power_limit=-1.0,
                 charging_efficiency=0.95,
                 discharging_efficiency=0.95):
        self.current_charge = current_charge
        self.capacity = capacity
        self.charging_power_limit = charging_power_limit
        self.discharging_power_limit = discharging_power_limit
        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency


class BatteryController(object):
    step = 960

    def propose_state_of_charge(self,
                                site_id,
                                timestamp,
                                battery,
                                actual_previous_load,
                                actual_previous_pv_production,
                                price_buy,
                                price_sell,
                                load_forecast,
                                pv_forecast):

        self.step -= 1
        number_step = max(1, min(96, self.step))
        #
        price_buy = price_buy.tolist()
        price_sell = price_sell.tolist()
        load_forecast = load_forecast.tolist()
        pv_forecast = pv_forecast.tolist()

        # battery
        c_p_l =  ( battery.charging_power_limit * battery.charging_efficiency) / 4.
        d_p_l = (battery.discharging_power_limit / battery.discharging_efficiency) / 4.

        capacity = battery.capacity
        charging_efficiency = 1. / battery.charging_efficiency
        discharging_efficiency = battery.discharging_efficiency
        current = battery.current_charge             # так как current_charge это доля (от 0 до 1), то энергия будет определяться произведением вместимости на долю

        infinity = grb.GRB.INFINITY

        model = grb.Model(name="Gurobi Model")

        # Описываем переменные (в виде словарей, ставим одновременно условия на переменные)
        X = [model.addVar(name="X" + str(i), lb=0.0, ub=capacity, vtype=grb.GRB.CONTINUOUS) for i in range(number_step + 1)]    # energy[i] - кол-во энергии, покупаемой на i-ом шаге
        Y = [model.addVar(name="Y" + str(i), lb=0.0, ub=c_p_l, vtype=grb.GRB.CONTINUOUS) for i in range(number_step)]           # charge[i] - кол-во энергии, которой заряжаем батарею на i-ом шаге
        Z = [model.addVar(name="Z" + str(i), lb=0.0, ub=infinity, vtype=grb.GRB.CONTINUOUS) for i in range(number_step)]        # grid[i] - кол-во энергии нагрузик на i-ом шаге
        mu = [model.addVar(name="l" + str(i), lb=0, ub=1, vtype=grb.GRB.BINARY) for i in range(number_step)]

        model.update()
        model.setParam('OutputFlag', 0, 0)

        model.addConstr(X[0] == (current * capacity), f'init_constraint')
        for i in range(0, number_step):
            model.addConstr((X[i+1] - X[i]) >= d_p_l, f'x_constraint_lower_{i}')            # мы не можем разрядить больше, чем на d_p_l
            model.addConstr((Y[i] - (X[i + 1] - X[i])) >= 0., f'ec_constraint_lower_{i}')   # то, чем мы зарядим батарею есть больше

            model.addConstr(X[i+1] - X[i] >= (1-mu[i]) * d_p_l, f'constraint_4_ub_{i}')
            model.addConstr(Y[i] <= mu[i] * c_p_l, f'constraint_5_lb_{i}')


            if (pv_forecast[i] < 20): pv_forecast[i] = 0.0
            model.addConstr((Z[i] + Y[i] * (discharging_efficiency - charging_efficiency) - discharging_efficiency * (X[i + 1] - X[i])) >= (load_forecast[i] - pv_forecast[i]), f'g_constraint_lower_{i}')
            #model.addConstr((Z[i] + Y[i] * (discharging_efficiency - charging_efficiency) - discharging_efficiency * (X[i + 1] - X[i])) <= infinity, f'g_constraint_upper_{i}')

        model.setObjective(grb.quicksum([Z[i] * (price_buy[i] - price_sell[i]) + Y[i] * (price_buy[i]/2000.0 + price_sell[i] * (charging_efficiency - discharging_efficiency)) + X[i] * ((price_sell[i-1] - price_sell[i]) *
                                                                                                                                                                                         discharging_efficiency) for i in range(1,
                                                                                                                                                                                                                                number_step)]) + Z[0] * (price_buy[0] - price_sell[0]) + Y[0] * (price_buy[0]/2000.0 + price_sell[0] * (charging_efficiency - discharging_efficiency)) + X[number_step] * price_sell[number_step - 1] * discharging_efficiency, grb.GRB.MINIMIZE)
        model.update()
        model.optimize()               # The msg parameter is used to display information from the solver. msg=False disables showing this information.

        if ((X[1].X >= X[0].X) & (pv_forecast[0] > load_forecast[0])):
            charging = X[1].X - X[0].X
            energy = pv_forecast[0] - load_forecast[0]
            for i in range(1, number_step):
                if ((price_sell[i] != price_sell[i - 1]) | (pv_forecast[i] < load_forecast[i]) | (X[i].X > X[i + 1].X)): break
                energy = energy + pv_forecast[i] - load_forecast[i]
                charging = charging + X[i + 1].X - X[i].X
            if (energy <= 0.0): return X[1].X / capacity
            return battery.current_charge + ((pv_forecast[0] - load_forecast[0]) * (charging / energy)) / capacity

        if ((X[1].X <= X[0].X) & (pv_forecast[0] < load_forecast[0])):
            charging = X[0].X - X[1].X
            energy = load_forecast[0] - pv_forecast[0]
            for i in range(1, number_step):
                if ((price_sell[i] != price_sell[i - 1]) | (price_buy[i] != price_buy[i - 1]) | (pv_forecast[i] > load_forecast[i]) | (X[i].X < X[i + 1].X)): break
                energy = energy + load_forecast[i] - pv_forecast[i]
                charging = charging + X[i].X - X[i + 1].X
            if (energy <= 0.0): return X[1].X / capacity
            return battery.current_charge + ((pv_forecast[0] - load_forecast[0]) * (charging / energy)) / capacity
        return X[1].X / capacity


class Simulation(object):
    """ Handles running a simulation.
    """
    def __init__(self,
                 data,
                 battery,
                 site_id):
        """ Creates initial simulation state based on data passed in.
            :param data: contains all the time series needed over the considered period
            :param battery: is a battery instantiated with 0 charge and the relevant properties
            :param site_id: the id for the site (building)
        """

        self.data = data

        # building initialization
        self.actual_previous_load = self.data.actual_consumption.values[0]
        self.actual_previous_pv = self.data.actual_pv.values[0]

        # align actual as the following, not the previous 15 minutes to
        # simplify simulation
        self.data.loc[:, 'actual_consumption'] = self.data.actual_consumption.shift(-1)  # сдвигает все элементы вверх (удаляя первый) и ставит значение последнего элемента на NaN
        self.data.loc[:, 'actual_pv'] = self.data.actual_pv.shift(-1)

        self.site_id = site_id
        self.load_columns = data.columns.str.startswith('load_')        # возвращает True или False в зависимости от того, начинается ли название столбца с load или нет
        self.pv_columns = data.columns.str.startswith('pv_')
        self.price_sell_columns = data.columns.str.startswith('price_sell_')
        self.price_buy_columns = data.columns.str.startswith('price_buy_')

        # initialize money at 0.0
        self.money_spent = 0.0
        self.money_spent_without_battery = 0.0

        # battery initialization
        self.battery = battery

    def run(self):
        """ Executes the simulation by iterating through each of the data points
            It returns both the electricity cost spent using the battery and the
            cost that would have been incurred with no battery.
        """
        battery_controller = BatteryController()  # создаем переменную, она сразу решает опт. проблему для текущего шага

        #for current_time, timestep in tqdm(self.data.iterrows(), total=self.data.shape[0], desc=' > > > > timesteps\t'):    # iterrows - итерация по индексам и строкам {current_time - index, timestep - rows}
        for current_time, timestep in self.data.iterrows():
                                                                                                                        # rows берет все столбцы, а index позволяет обращаться полностью к строке таблицы
                                                                                                                        # т.е. они обращаются к каждой строке таблицы
                                                                                                                        # total - для tdqm (100%), а desc - описание просто
            # can't calculate results without actual, so skip (should only be last row)
            if pd.notnull(timestep.actual_consumption):                                     # РАБОТАЕМ СО СТРОКОЙ (если элемент столбца 'actual_consumption' не пустой)
                self.simulate_timestep(battery_controller, current_time, timestep)          # проводим симуляцию для батареи с имеющейся строкой данных (получим затраты и предполагаемое сост. батареи)

        return self.money_spent, self.money_spent_without_battery                           # возвращаем потраченную сумму с и без батареи

    def simulate_timestep(self, battery_controller, current_time, timestep):
        """ Executes a single timestep using `battery_controller` to get
            a proposed state of charge and then calculating the cost of
            making those changes.
            :param battery_controller: The battery controller
            :param current_time: the timestamp of the current time step
            :param timestep: the data available at this timestep
        """
        # get proposed state of charge from the battery controller
        proposed_state_of_charge = battery_controller.propose_state_of_charge(              # решаем для батареи опт. задачу с имеющимися параметрами
            self.site_id,
            current_time,
            deepcopy(self.battery),                     # используем функцию библиотеки copy, чтобы скопировать объект battery (иначе меняя battery мы бы меняли оба элемента, если присвоить через =)
            self.actual_previous_load,
            self.actual_previous_pv,
            timestep[self.price_buy_columns],
            timestep[self.price_sell_columns],
            timestep[self.load_columns],
            timestep[self.pv_columns]
        )

        # get energy required to achieve the proposed state of charge
        grid_energy, battery_energy_change = self.simulate_battery_charge(self.battery.current_charge,
                                                                          proposed_state_of_charge,
                                                                          timestep.actual_consumption,
                                                                          timestep.actual_pv)

        grid_energy_without_battery = timestep.actual_consumption - timestep.actual_pv

        # buy or sell energy depending on needs
        price = timestep.price_buy_00 if grid_energy >= 0 else timestep.price_sell_00
        price_without_battery = timestep.price_buy_00 if grid_energy_without_battery >= 0 else timestep.price_sell_00

        # calculate spending based on price per kWh and energy per Wh
        self.money_spent += grid_energy * (price / 1000.)
        self.money_spent_without_battery += grid_energy_without_battery * (price_without_battery / 1000.)

        # update current state of charge
        self.battery.current_charge += battery_energy_change / self.battery.capacity
        self.actual_previous_load = timestep.actual_consumption
        self.actual_previous_pv = timestep.actual_pv

    def simulate_battery_charge(self, initial_state_of_charge, proposed_state_of_charge, actual_consumption, actual_pv):
        """ Charges or discharges the battery based on what is desired and
            available energy from grid and pv.
            :param initial_state_of_charge: the current state of the battery
            :param proposed_state_of_charge: the proposed state for the battery
            :param actual_consumption: the actual energy consumed by the building           (last time)
            :param actual_pv: the actual pv energy produced and available to the building   (last time)
        """
        # charge is bounded by what is feasible
        proposed_state_of_charge = np.clip(proposed_state_of_charge, 0.0, 1.0)  # все значения сделать от 0 до 1 (если что-то вылезает за рамки, то оно приравнивается к краевым)

        # calculate proposed energy change in the battery
        target_energy_change = (proposed_state_of_charge - initial_state_of_charge) * self.battery.capacity   # находим число изменившейся энергии за 15 минут (в долях)

        # efficiency can be different whether we intend to charge or discharge
        if target_energy_change >= 0:                                                       # если заряжаем
            efficiency = self.battery.charging_efficiency
            target_charging_power = target_energy_change / ((15. / 60.) * efficiency)       # нужна мощность заряда (в часах) => умножаем то что нужно на 4 (1/0.25), делим на eff (нужно знать сколько исходно, а не получим)
        else:                                                                               # если разряжаем
            efficiency = self.battery.discharging_efficiency                                # поскольку разрядка, то ранее мы обозначили eff >= 1
            target_charging_power = target_energy_change * efficiency / (15. / 60.)         # нужна мощность заряда (в часах) => умножаем на eff (>=1) и на 4 [чтобы узнать исходную мощность, что нужна]

        # actual power is bounded by the properties of the battery
        actual_charging_power = np.clip(target_charging_power,                              # зажимаем значение реального заряда батареи между лимитами зарядки/разрядки
                                        self.battery.discharging_power_limit,
                                        self.battery.charging_power_limit)

        # actual energy change is based on the actual power possible and the efficiency
        if actual_charging_power >= 0:                                                      # если зарядка батареи
            actual_energy_change = actual_charging_power * (15. / 60.) * efficiency         # то умножаем на эффективность (в долях <=1) {меньше будет} И на 15/60 {поскольку величина в часах у нас}
        else:                                                                               # если разрядка батареи
            actual_energy_change = actual_charging_power * (15. / 60.) / efficiency         # то делим на эффективность (>= 1) {меньше будет} И умножаем на 15/60 {поскольку величина в часах у нас}

        # what we need from the grid = (the power put into the battery + the consumption) - what is available from pv
        grid_energy = (actual_charging_power * (15. / 60.) + actual_consumption) - actual_pv

        # if positive, we are buying from the grid; if negative, we are selling
        return grid_energy, actual_energy_change


if __name__ == '__main__':
    start = time.time()
    metadata_path = 'metadata.csv'
    metadata = pd.read_csv(metadata_path,index_col= 0,sep = ';')

    #metadata = metadata.head(1)   # кол-во домов
    house_batteries = 2
    forecasts_correction = True

    results = [] # store results of each run

    # # execute two runs with each battery for every row in the metadata file:
    for site_id, parameters in tqdm(metadata.iterrows(), desc='sites\t\t\t', total=metadata.shape[0]):
    #-------------------------------
        site_data_path = "submit" + str(site_id) + '.csv'

        #if site_data_path.exists():
        site_data = pd.read_csv(site_data_path, index_col='timestamp', sep=';', parse_dates=['timestamp'])

        for batt_id in tqdm([i+1 for i in range(house_batteries)], desc=' > batteries \t\t'):
            # create the battery for this run
            # (Note: Quantities in kW are converted to watts here)
            batt = Battery(capacity=parameters[f"Battery_{batt_id}_Capacity"] * 1000,
                            charging_power_limit=parameters[f"Battery_{batt_id}_Power"] * 1000,
                            discharging_power_limit=-parameters[f"Battery_{batt_id}_Power"] * 1000,
                            charging_efficiency=parameters[f"Battery_{batt_id}_Charge_Efficiency"],
                            discharging_efficiency=parameters[f"Battery_{batt_id}_Discharge_Efficiency"])

            # execute the simulation for each simulation period in the data
            n_periods = site_data.period_id.nunique()
            for g_id, g_df in tqdm(site_data.groupby('period_id'), total=n_periods, desc=' > > periods\t\t'):   # формирует из .csv кол-во dataFrame (таблиц), равное числу периодов и работает с ними (с каждой табл отдельно)
                # reset battery to no charge before simulation
                batt.current_charge = 0
                sim = Simulation(g_df, batt, site_id)           # создаем переменную класса Simulation
                money_spent, money_no_batt = sim.run()          # вызываем метод класса для переменной (там прогоняется все сразу)

                # store the results
                results.append({                                # в лист записываем оформляем результаты как показано
                    'run_id': f"{site_id}_{batt_id}_{g_id}",
                    'site_id': site_id,
                    'battery_id': batt_id,
                    'period_id': g_id,
                    'money_spent': money_spent,
                    'money_no_batt': money_no_batt,
                    'score': (money_spent - money_no_batt) / np.abs(money_no_batt),
                })
    # -------------------------------

    # write all results out to a file
    results_df = pd.DataFrame(results).set_index('run_id')                                                          # делаем из листа df и в качестве индекса берем run_id
    results_df = results_df[['site_id', 'battery_id', 'period_id', 'money_spent', 'money_no_batt', 'score']]        # выбираем конкретные столбцы из df

    end = time.time()
    print("Затраченное время = ", end - start, " c.")


    results_df.to_csv(f'results_GUROBI_(3-variables)_MILP.csv')
    my_average_function(f'results_GUROBI_(3-variables)_MILP')

