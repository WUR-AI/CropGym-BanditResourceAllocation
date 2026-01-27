import numpy as np

from abc import ABC, abstractmethod

from cropgymzoo.envs.pcse_env import get_random_weather_provider
from cropgymzoo.utils.nitrogen_helpers import input_nue, get_surplus_n, get_n_deposition_pcse, get_nh4_deposition_pcse, get_no3_deposition_pcse
import cropgymzoo.utils.process_pcse_output as process_pcse


def reward_functions_without_baseline():
    return ['GRO', 'DEP', 'ENY', 'NUE', 'HAR', 'NUP']


def reward_functions_with_baseline():
    return ['DEF', 'ANE', 'END', 'PNR', 'MPN']


def reward_function_list():
    return ['DEF', 'GRO', 'DEP', 'ENY', 'NUE', 'DNU', 'HAR', 'NUP', 'END', 'FIN']


def reward_functions_end():
    return ['END', 'ENY']


def get_min_yield(loc="52.57-5.63"):
    if loc == "52.57-5.63":
        return 0
    else:
        return 0


def get_max_yield(loc="52.57-5.63"):
    if loc == "52.57-5.63":
        return 10_000
    else:
        return 10_000


class Rewards:
    def __init__(self, reward_var, timestep, costs_nitrogen=10.0, vrr=0.7, with_year=False):
        self.reward_var = reward_var
        self.timestep = timestep
        self.costs_nitrogen = costs_nitrogen
        self.vrr = vrr
        self.profit = 0
        self.with_year = with_year

        # fertilizer_price and costs_nitrogen redundant
        self.crop_price = 0
        self.fertilizer_price = costs_nitrogen

    def growth_storage_organ(self, output, amount, multiplier=1):
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
        costs = self.costs_nitrogen * amount
        reward = growth - costs
        return reward, growth

    def growth_reward_var(self, output, amount):
        growth = process_pcse.compute_growth_var(output, self.timestep, self.reward_var)
        costs = self.costs_nitrogen * amount
        reward = growth - costs
        return reward, growth

    def default_winterwheat_reward(self, output, output_baseline, amount, multiplier=1):
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
        growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep, multiplier)
        benefits = growth - growth_baseline
        costs = self.costs_nitrogen * amount
        reward = benefits - costs
        return reward, growth

    def deployment_reward(self, output, amount, multiplier=1, vrr=None):
        """
        reward function that mirrors a realistic (financial) cost of DT deployment in a field
        one unit of reward equals the price of 1kg of wheat yield
        """
        # recovered_fertilizer = amount * vrr
        # unrecovered_fertilizer = (amount - recovered_fertilizer) * self.various_costs()['environmental']
        if amount == 0:
            cost_deployment = 0
        else:
            cost_deployment = self.various_costs()['to_the_field']

        growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
        # growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep)
        # fertilizer_price = self.various_costs()['fertilizer'] * amount
        costs = (self.costs_nitrogen * amount) + cost_deployment
        reward = growth - costs
        return reward, growth

    # agronomic nitrogen use efficiency (ee Vanlauwe et al, 2011)
    def ane_reward(self, ane_obj, output, output_baseline, amount):
        # agronomic nitrogen use efficiency
        reward, growth = ane_obj.reward(output, output_baseline, amount)
        return reward, growth

    def end_reward(self, end_obj, output, output_baseline, amount, multiplier=1):
        end_obj.calculate_cost_cumulative(amount)
        end_obj.calculate_positive_reward_cumulative(output, output_baseline)
        reward = 0 - amount * self.costs_nitrogen
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

        return reward, growth

    def nue_reward(self, nue_obj, output, output_baseline, amount, multiplier=1):
        nue_obj.calculate_cost_cumulative(amount)
        nue_obj.calculate_positive_reward_cumulative(output, output_baseline, multiplier)
        reward = 0 - amount * self.costs_nitrogen
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

        return reward, growth

    def calc_misc_cost(self, end_obj, cost):
        end_obj.calculate_misc_cumulative_cost(cost)

    # TODO create reward surrounding crop N demand; WIP
    def n_demand_yield_reward(self, output, multiplier=1):
        assert 'TWSO' and 'Ndemand' in self.reward_var, f"reward_var does not contain TWSO and Ndemand"
        n_demand_diff = process_pcse.compute_growth_var(output, self.timestep, 'Ndemand')
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
        benefits = growth - n_demand_diff
        print(f"the N demand is {n_demand_diff}")
        print(f"the benefits are {benefits}")
        return benefits, growth

    def reset(self):
        self.profit = 0

    def calculate_profit(self, output, amount, year, multiplier, with_year=False, country='NL'):
        
        profit, _ = calculate_net_profit(output, amount, year, multiplier, self.timestep, with_year=with_year, country=country)

        return profit

    def update_profit(self, output, amount, year, multiplier=1, country='NL'):
        self.profit += self.calculate_profit(output, amount, year, multiplier, with_year=self.with_year)

    def calculate_nue_on_terminate(self, n_input, n_so, year, start=None, end=None, no3_depo=None, nh4_depo=None, crop_name=None):
        return calculate_nue(n_input, n_so, year=year, start=start, end=end, no3_depo=no3_depo, nh4_depo=nh4_depo, crop_name=crop_name)

    """
    Classes that determine the reward function
    """

    class Rew(ABC):
        def __init__(self, timestep, costs_nitrogen, budget_left = None):
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen
            self.budget_left = budget_left

        @abstractmethod
        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            raise NotImplementedError

    class DEF(Rew):
        """
        Relative yield reward function, as implemented in Kallenberg et al (2023)
        """

        def __init__(self, timestep, costs_nitrogen):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep, multiplier)
            benefits = growth - growth_baseline
            costs = self.costs_nitrogen * amount
            reward = benefits - costs
            return reward, growth

    class GRO(Rew):
        """
        Absolute growth reward function, modified from Kallenberg et al. (2023)
        """

        def __init__(self, timestep, costs_nitrogen):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            costs = self.costs_nitrogen * amount
            reward = growth - costs
            return reward, growth

    class LOS(Rew):
        """
        Absolute growth reward function with N loss penalty, modified from Kallenberg et al. (2023)
        """

        def __init__(self, timestep, costs_nitrogen):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            costs = self.costs_nitrogen * amount
            loss = process_pcse.compute_growth_var(output, self.timestep, 'NLOSSCUM')
            costs_loss = 0.1 * loss
            reward = growth - costs - costs_loss
            return reward, growth

    class DEP(Rew):
        """
        Reward function that considers a realistic (financial) cost of DT deployment in a field
        one unit of reward equals the price of 1kg of wheat yield
        """

        def __init__(self, timestep, costs_nitrogen):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)
            cost_deployment = 0 if amount == 0 else self.various_costs()['to_the_field']

            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            # growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep)
            # fertilizer_price = self.various_costs()['fertilizer'] * amount
            costs = (self.costs_nitrogen * amount) + cost_deployment
            reward = growth - costs
            return reward, growth

        @staticmethod
        def various_costs():
            return dict(
                to_the_field=10,
                fertilizer=1,
                environmental=2
            )

    class END(Rew):
        """
        Sparse reward function, modified from Kallenberg et al. (2023)
        Only provides positive reward at harvest
        """

        def __init__(self, timestep, costs_nitrogen):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)
            obj.calculate_cost_cumulative(amount)
            obj.calculate_positive_reward_cumulative(output, output_baseline)
            reward = 0 - amount * self.costs_nitrogen
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

            return reward, growth

    class NUE(Rew):
        """
        Sparse reward based on calculated nitrogen use efficiency
        """

        def __init__(self, timestep, costs_nitrogen):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)
            obj.calculate_cost_cumulative(amount)
            obj.calculate_positive_reward_cumulative(output, output_baseline, multiplier)
            reward = 0 - self.costs_nitrogen if amount > 0 else 0
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

            return reward, growth

    class PNB(Rew):
        """
        Profit, NUE, Budget left reward function
        """

        def __init__(self, timestep, costs_nitrogen, fertilizer_price=None, crop_price=None, budget_left=None):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen
            self.fertilizer_price = fertilizer_price
            self.crop_price = crop_price

            self.fertilizer_beta = 1
            self.nsurp_beta = 1
            self.nue_beta = 1
            self.budget_beta = 0

        def return_reward(
                self,
                output,
                amount,
                output_baseline=None,
                multiplier=1,
                obj=None,
                price_crop=None,
                price_fertilizer=None,
                budget_left=None,
                fresh_yield_fn=None,
        ):

            obj.calculate_amount(amount)

            self.update_crop_price(price_crop)
            self.update_fertilizer_price(price_fertilizer)
            obj.update_fertilizer_price(price_fertilizer)
            obj.update_crop_price(price_crop)

            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            obj.calculate_positive_reward_cumulative(output, output_baseline, multiplier)

            if fresh_yield_fn is not None:
                growth = fresh_yield_fn(growth)

            # get profit
            profit = obj.calculate_profit_term(
                action=amount,
                growth=growth,
                price_crop=price_crop,
                price_fertilizer=price_fertilizer
            )


            return profit, growth

        def return_final_reward(
                self,
                obj=None,
                n_fertilized=None,
                n_output=None,
                no3_depo=None,
                nh4_depo=None,
                budget_left=None,
                crop_name=None,
        ):
            # Maybe give negative reward if did not act at all; soil mining most likely
            # if obj.get_total_fertilization == 0:
            #     return -obj.cum_profit - 100

            n_surplus = get_surplus_n(n_input=n_fertilized, n_so=n_output, no3_depo=no3_depo, nh4_depo=nh4_depo, crop_name=crop_name)

            nue = calculate_nue(n_input=n_fertilized, n_so=n_output, no3_depo=no3_depo, nh4_depo=nh4_depo, crop_name=crop_name)

            n_surplus_penalty =  obj.n_surplus_penalty(n_surplus)
            nue_penalty = obj.nue_penalty(nue, n_output)

            # budget_left_bonus = self.budget_beta * obj.budget_left_bonus(budget_left)

            # End reward in three terms that describe profit
            reward = (
                    - abs(self.nsurp_beta * n_surplus_penalty)
                    - abs(self.nue_beta * nue_penalty)
                    # + budget_left_bonus
            )
            return reward


        def update_fertilizer_price(self, fertilizer_price):
            self.fertilizer_price = fertilizer_price

        def update_crop_price(self, crop_price):
            self.crop_price = crop_price




    class PNY(Rew):
        """
        Profit, NUE and Yield reward function
        """

        def __init__(self, timestep, costs_nitrogen,
                     mu_profit:float=250.0, # euros
                     k_profit:float=0.002,
                     beta_p:float=0.5,
                     mu_yield:float=7_000, # kg/ha
                     k_yield:float=0.0045,
                     beta_y:float=0.0,
                     beta_n:float=0.5,):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen
            self.mu_profit = mu_profit
            self.k_profit = k_profit
            self.beta_p = beta_p

            self.mu_yield = mu_yield
            self.k_yield = k_yield
            self.beta_y = beta_y

            self.beta_n = beta_n

            self.check_max()

        def check_max(self):
            max_possible = self.beta_p * (1 - self.squash_profit_reward(0)) + self.beta_y * (1 - self.squash_yield_reward(0)) + self.beta_n * 1
            assert max_possible <= 1.0, "all the beta terms should be <= 1"

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None,
                          price_crop=None, price_fertilizer=None, budget_left=None):
            # generic
            obj.calculate_amount(amount)
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            obj.calculate_positive_reward_cumulative(output, output_baseline, multiplier)

            #----------- profit
            prev_profit = obj.cum_profit

            # also updates obj.cum_profit
            profit_now = obj.calculate_profit_term(action=amount, growth=growth,
                                       price_crop=price_crop, price_fertilizer=price_fertilizer)

            # get gradient from previous
            profit_reward = self.squash_profit_reward(obj.cum_profit) - self.squash_profit_reward(prev_profit)

            obj.calculate_running_profit_reward(self.beta_p * profit_reward)

            #------------ yield
            prev_yield = obj.cum_growth

            obj.calculate_growth(growth)

            yield_reward = self.squash_yield_reward(obj.cum_growth) - self.squash_yield_reward(prev_yield)

            #------------ calc

            profit_term = profit_reward * self.beta_p

            yield_term = yield_reward * self.beta_y

            reward = profit_term + yield_term

            return reward, growth

        def return_final_reward(self, obj=None, n_fertilized=None, n_output=None, no3_depo=None, nh4_depo=None):
            nue_term = obj.calculate_nue_term(n_fertilized, n_output, no3_depo=no3_depo, nh4_depo=nh4_depo)

            reward_nue = self.beta_n * nue_term

            expected_profit_reward = self.beta_p * (self.squash_profit_reward(obj.cum_profit) -\
                                self.squash_profit_reward(0))

            assert abs(obj.cum_running_profit - expected_profit_reward) < 1e-6

            return reward_nue

        def squash_profit_reward(self, profit_term):
            return 1 / (1 + np.exp(-self.k_profit * (profit_term - self.mu_profit)))

        def squash_yield_reward(self, yield_term):
            return 1 / (1 + np.exp(-self.k_yield * (yield_term - self.mu_yield)))

        def reward_bounds(self):
            sigma0_p = self.squash_profit_reward(0)
            sigma0_y = self.squash_yield_reward(0)
            sigma_p_max = 1 - sigma0_p
            r_min = self.beta_p * (-sigma0_p) + self.beta_y * (-sigma0_y) + 0
            r_max = self.beta_p * (1 - sigma0_p) + self.beta_y * (1 - sigma0_y) + self.beta_n
            return r_min, r_max


    class RPN(Rew):
        """
        Reward as ratio of Profit and N applied
        """

        def __init__(self, timestep, costs_nitrogen, fertilizer_price=None, crop_price=None, budget_left=None):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen
            self.fertilizer_price = fertilizer_price
            self.crop_price = crop_price

            self.fertilizer_beta = 1
            self.nsurp_beta = 7 * 5
            self.nue_beta = 7 * 5
            self.budget_beta = 0

        def return_reward(
                self,
                output,
                amount,
                output_baseline=None,
                multiplier=1,
                obj=None,
                price_crop=None,
                price_fertilizer=None,
                budget_left=None,
                fresh_yield_fn=None,
        ):

            obj.calculate_amount(amount)

            self.update_crop_price(price_crop)
            self.update_fertilizer_price(price_fertilizer)
            obj.update_fertilizer_price(price_fertilizer)
            obj.update_crop_price(price_crop)

            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            obj.calculate_positive_reward_cumulative(output, output_baseline, multiplier)

            if fresh_yield_fn is not None:
                growth = fresh_yield_fn(growth)

            # get profit
            profit = obj.calculate_profit_term(
                action=amount,
                growth=growth,
                price_crop=price_crop,
                price_fertilizer=price_fertilizer
            )

            labor_cost = 30 if amount > 0 else 0

            profit_now = profit - labor_cost

            return profit_now, growth

        def return_final_reward(
                self,
                obj=None,
                n_fertilized=None,
                n_output=None,
                no3_depo=None,
                nh4_depo=None,
                budget_left=None,
                crop_name=None,
        ):
            # Maybe give negative reward if did not act at all; soil mining most likely
            # if obj.get_total_fertilization == 0:
            #     return -obj.cum_profit - 100

            n_surplus = get_surplus_n(n_input=n_fertilized, n_so=n_output, no3_depo=no3_depo, nh4_depo=nh4_depo, crop_name=crop_name)

            nue = calculate_nue(n_input=n_fertilized, n_so=n_output, no3_depo=no3_depo, nh4_depo=nh4_depo, crop_name=crop_name)

            n_surplus_penalty =  obj.n_surplus_penalty(n_surplus)
            nue_penalty = obj.nue_penalty(nue)

            budget_left_bonus = self.budget_beta * obj.budget_left_bonus(budget_left)

            # End reward in three terms that describe profit
            reward = budget_left_bonus - abs(self.nsurp_beta * n_surplus_penalty) - abs(self.nue_beta * nue_penalty)
            return reward


        def update_fertilizer_price(self, fertilizer_price):
            self.fertilizer_price = fertilizer_price

        def update_crop_price(self, crop_price):
            self.crop_price = crop_price

    class PNR(Rew):
        """
        Relative profit and NUE reward function
        """

        def __init__(self, timestep, costs_nitrogen, fertilizer_price=None, crop_price=None, budget_left=None):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen
            self.fertilizer_price = fertilizer_price
            self.crop_price = crop_price

            self.fertilizer_beta = 1
            self.nsurp_beta = 1
            self.nue_beta = 1
            self.budget_beta = 0

        def return_reward(
                self,
                output,
                amount,
                output_baseline=None,
                multiplier=1,
                obj=None,
                price_crop=None,
                price_fertilizer=None,
                budget_left=None,
                fresh_yield_fn=None,
        ):
            obj.calculate_amount(amount)

            self.update_crop_price(price_crop)
            self.update_fertilizer_price(price_fertilizer)
            obj.update_fertilizer_price(price_fertilizer)
            obj.update_crop_price(price_crop)

            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            if output_baseline:
                growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep, multiplier)
            obj.calculate_positive_reward_cumulative(output, output_baseline, multiplier)

            if fresh_yield_fn is not None:
                growth = fresh_yield_fn(growth)
                if output_baseline:
                    growth_baseline = fresh_yield_fn(growth_baseline)

            # get profit
            profit_now = obj.calculate_profit_term(
                action=amount,
                growth=growth,
                price_crop=price_crop,
                price_fertilizer=price_fertilizer
            )

            if output_baseline:
                profit_baseline = obj.calculate_profit_term(
                    action=0,
                    growth=growth_baseline,
                    price_crop=price_crop,
                    price_fertilizer=price_fertilizer
                )

            profit = profit_now - (profit_baseline if output_baseline else 0)

            return profit, growth

        def return_final_reward(
                self,
                obj=None,
                n_fertilized=None,
                n_output=None,
                no3_depo=None,
                nh4_depo=None,
                budget_left=None,
                crop_name=None,
        ):

            n_surplus = get_surplus_n(n_input=n_fertilized, n_so=n_output, no3_depo=no3_depo, nh4_depo=nh4_depo,
                                      crop_name=crop_name)

            nue = calculate_nue(n_input=n_fertilized, n_so=n_output, no3_depo=no3_depo, nh4_depo=nh4_depo,
                                crop_name=crop_name)

            n_surplus_penalty = obj.n_surplus_penalty(n_surplus)
            nue_penalty = obj.nue_penalty(nue, n_output)

            obj.accumulate_profit(- n_surplus_penalty - nue_penalty)

            # budget_left_bonus = self.budget_beta * obj.budget_left_bonus(budget_left)

            obj.calculate_final_nue_penalty_profit(nue_penalty, n_surplus_penalty)

            # End reward in three terms that describe profit
            reward = (
                    - abs(self.nsurp_beta * n_surplus_penalty)
                    - abs(self.nue_beta * nue_penalty)
                # + budget_left_bonus
            )
            return reward

        def update_fertilizer_price(self, fertilizer_price):
            self.fertilizer_price = fertilizer_price

        def update_crop_price(self, crop_price):
            self.crop_price = crop_price

    class NSU(PNR):
        """
        Sparse reward based on calculated nitrogen surplus
        """

        def __init__(self, timestep, costs_nitrogen):
            super().__init__(timestep, costs_nitrogen)

        def return_reward(self,
                output,
                amount,
                output_baseline=None,
                multiplier=1,
                obj=None,
                price_crop=None,
                price_fertilizer=None,
                budget_left=None,
                fresh_yield_fn=None,):

            self.update_crop_price(price_crop)
            self.update_fertilizer_price(price_fertilizer)
            obj.update_fertilizer_price(price_fertilizer)
            obj.update_crop_price(price_crop)

            obj.calculate_amount(amount)
            obj.calculate_cost_cumulative(amount)
            obj.calculate_positive_reward_cumulative(output, output_baseline, multiplier)
            reward = 0
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

            if fresh_yield_fn is not None:
                growth = fresh_yield_fn(growth)

            _ = obj.calculate_profit_term(
                action=amount,
                growth=growth,
                price_crop=price_crop,
                price_fertilizer=price_fertilizer
            )

            return reward, growth


    class MPN(Rew):
        """
        Marginal profit increase per unit of N applied
        """

        def __init__(self, timestep, costs_nitrogen, fertilizer_price=None, crop_price=None, budget_left=None,
                     alpha: float = 1.0, squash_k: float = 1):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen
            self.fertilizer_price = fertilizer_price
            self.crop_price = crop_price

            # per-kg stabilizer to prevent huge ratios when amount≈0
            self.alpha = float(alpha)
            # optional squashing gain for tanh
            self.squash_k = float(squash_k)

            # penalty weights
            self.fertilizer_beta = 1
            self.nsurp_beta = 1
            self.nue_beta = 1
            self.budget_beta = 0

        def return_reward(
                self,
                output,
                amount,
                output_baseline=None,
                multiplier=1,
                obj=None,
                price_crop=None,
                price_fertilizer=None,
                budget_left=None,
                fresh_yield_fn=None,
        ):
            obj.calculate_amount(amount)

            self.update_crop_price(price_crop)
            self.update_fertilizer_price(price_fertilizer)
            obj.update_fertilizer_price(price_fertilizer)
            obj.update_crop_price(price_crop)

            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            if output_baseline:
                growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep, multiplier)
            obj.calculate_positive_reward_cumulative(output, output_baseline, multiplier)

            if fresh_yield_fn is not None:
                growth = fresh_yield_fn(growth)
                if output_baseline:
                    growth_baseline = fresh_yield_fn(growth_baseline)

            # get profit
            profit_now = obj.calculate_profit_term(
                action=amount,
                growth=growth,
                price_crop=price_crop,
                price_fertilizer=price_fertilizer
            )

            if output_baseline:
                profit_baseline = obj.calculate_profit_term(
                    action=0,
                    growth=growth_baseline,
                    price_crop=price_crop,
                    price_fertilizer=price_fertilizer
                )
            else:
                profit_baseline = 0.0

            # Incremental profit from taking the current action vs. zero-N baseline
            d_profit = profit_now - profit_baseline
            abs_mag_profit = abs(profit_baseline)

            reward_step = round(
                float(
                    d_profit / (abs_mag_profit + 1e-8)
                ), 3
            )

            return reward_step, growth

        def return_final_reward(
                self,
                obj=None,
                n_fertilized=None,
                n_output=None,
                no3_depo=None,
                nh4_depo=None,
                budget_left=None,
                crop_name=None,
        ):
            n_surplus = get_surplus_n(
                n_input=n_fertilized,
                n_so=n_output,
                no3_depo=no3_depo,
                nh4_depo=nh4_depo,
                crop_name=crop_name
            )

            nue = calculate_nue(
                n_input=n_fertilized,
                n_so=n_output,
                no3_depo=no3_depo,
                nh4_depo=nh4_depo,
                crop_name=crop_name
            )

            n_surplus_penalty = obj.n_surplus_penalty(n_surplus)
            nue_penalty = obj.nue_penalty(nue, n_output)

            # budget_left_bonus = self.budget_beta * obj.budget_left_bonus(budget_left)

            # End reward in three terms that describe profit
            reward = (
                - abs(self.nsurp_beta * n_surplus_penalty)
                - abs(self.nue_beta * nue_penalty)
                # + budget_left_bonus
            )
            return reward

        def update_fertilizer_price(self, fertilizer_price):
            self.fertilizer_price = fertilizer_price

        def update_crop_price(self, crop_price):
            self.crop_price = crop_price

    class DNE(Rew):
        """
        Dense reward based on calculated nitrogen use efficiency
        """

        def __init__(self, timestep, costs_nitrogen):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)
            obj.calculate_cost_cumulative(amount)
            obj.calculate_positive_reward_cumulative(output, output_baseline, multiplier)
            reward = 0 - amount * self.costs_nitrogen
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

            return reward, growth

    class DSO(Rew):
        """
        Dense reward based on calculated nitrogen use efficiency
        """

        def __init__(self, timestep, costs_nitrogen, so_weight=20):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen
            self.so_weight = so_weight

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)
            obj.calculate_cost_cumulative(amount)
            obj.calculate_positive_reward_cumulative(output, output_baseline, multiplier)

            n_so = process_pcse.compute_growth_var(output, self.timestep, 'NamountSO') * self.so_weight

            reward = n_so - amount * self.costs_nitrogen

            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

            return reward, growth

    class NUP(Rew):
        """
        Reward based on Nitrogen Uptake, from Gautron et al. (2023)
        """

        def __init__(self, timestep, costs_nitrogen):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)
            growth = process_pcse.compute_growth_var(output, self.timestep, 'NuptakeTotal')
            costs = self.costs_nitrogen * amount
            reward = growth - costs
            return reward, growth

    class HAR(Rew):
        """
        Sparse reward based on Wu et al. (2021) considering N losses
        """

        def __init__(self, timestep, costs_nitrogen, threshold=200, loss_modifier=1, penalty_modifier=1):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.threshold = threshold
            self.costs_nitrogen = costs_nitrogen
            self.loss_modifier = loss_modifier
            self.penalty_modifier = penalty_modifier

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)
            # N application (N_t)
            obj.calculate_cost_n(amount)
            # N loss (N_l_t)
            n_loss = process_pcse.compute_growth_var(output, self.timestep, 'NLOSSCUM')
            obj.calculate_n_loss(n_loss)
            # Yield growth (Y)
            obj.calculate_positive_reward_cumulative(output)
            # Threshold
            # penalty = obj.calculate_threshold(amount, self.threshold)

            reward = 0 - amount * self.costs_nitrogen - n_loss * self.loss_modifier  # - penalty * self.penalty_modifier
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

            return reward, growth

    class DNU(Rew):
        """
        Dense reward of Nitrogen in Wheat Grain and N losses and N deposition
        """
        def __init__(self, timestep, costs_nitrogen):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen
            self.n_so_mod = 5
            self.n_dep_mod = 1
            self.n_loss_mod = 5
            self.n_fert_mod = 2

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)
            # N grain growth
            n_so = process_pcse.compute_growth_var(output, self.timestep, 'NamountSO')
            # N loss
            n_loss = process_pcse.compute_growth_var(output, self.timestep, 'NLOSSCUM')
            # N deposition
            # nh4, no3 = get_disaggregated_deposition(year=process_pcse.get_year_in_step(output),
            #                                         start_date=
            #                                         output[process_pcse.get_previous_index(output, self.timestep)][
            #                                             'day'],
            #                                         end_date=output[-1]['day'])
            n_dep = 25
            reward = (n_so * self.n_so_mod - amount * self.n_fert_mod
                      - n_dep * self.n_dep_mod - n_loss * self.n_loss_mod)
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

            return reward, growth

    class FIN(Rew):
        """
        Financial reward function, converting yield, N fertilizer and labour costs into a net profit reward.
        """
        def __init__(self, timestep, costs_nitrogen, labour=False):
            super().__init__(timestep, costs_nitrogen)
            self.labour = labour
            self.base_labour_cost_index = 28.9  # euros, in 2020
            self.time_per_hectare = 5 / 60  # minutes to hours
            self.country = 'NL'

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)

            year = process_pcse.get_year_in_step(output)

            reward, growth = calculate_net_profit(output, amount, year, multiplier, self.timestep, with_year=False)

            return reward, growth

    """
    Containers for certain reward functions
    """

    class ContainerEND:
        """
        Container to keep track of cumulative positive rewards for end of timestep
        """

        def __init__(self, timestep, costs_nitrogen=10.0):
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

            self.fertilizer_price = .0
            self.crop_price = .0

            self.cum_growth = .0
            self.cum_amount = .0
            self.cum_positive_reward = .0
            self.cum_cost = .0
            self.cum_misc_cost = .0
            self.cum_leach = .0
            self.actions = 0

            self.cum_profit = .0
            self.cum_running_profit = .0

        def reset(self):
            self.cum_growth = .0
            self.cum_amount = .0
            self.cum_positive_reward = .0
            self.cum_cost = .0
            self.cum_misc_cost = .0
            self.cum_leach = .0
            self.actions = 0

            self.cum_profit = .0
            self.cum_running_profit = .0

        def growth_storage_organ_wo_cost(self, output, multiplier=1):
            return process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

        def default_reward_wo_cost(self, output, output_baseline, multiplier=1):
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep, multiplier)
            benefits = growth - growth_baseline
            return benefits

        def growth_var(self, output, var):
            return process_pcse.compute_growth_var(output, self.timestep, var)

        def calculate_amount(self, action):
            self.actions += action

        def calculate_cost_cumulative(self, amount):
            self.cum_amount += amount
            self.cum_cost += amount * self.costs_nitrogen

        def calculate_misc_cumulative_cost(self, cost):
            self.cum_misc_cost += cost

        def calculate_positive_reward_cumulative(self, output, output_baseline=None, multiplier=1):
            if not output_baseline:
                benefits = self.growth_storage_organ_wo_cost(output, multiplier)
            else:
                benefits = self.default_reward_wo_cost(output, output_baseline, multiplier)
            self.cum_positive_reward += benefits

        def calculate_cost_n(self, amount):
            self.cum_amount += amount

        def accumulate_profit(self, profit):
            self.cum_profit += profit

        # debug function
        def calculate_running_profit_reward(self, profit):
            self.cum_running_profit += profit

        def calculate_growth(self, growth):
            self.cum_growth += growth

        def calculate_n_loss(self, n_loss):
            self.cum_leach += n_loss

        def calculate_threshold(self, amount, threshold):
            if amount == 0:
                return 0
            else:
                return self.cum_amount - threshold

        def update_fertilizer_price(self, fertilizer_price):
            self.fertilizer_price = fertilizer_price

        def update_crop_price(self, crop_price):
            self.crop_price = crop_price

        @property
        def get_total_fertilization(self):
            return self.actions

        @property
        def dump_cumulative_positive_reward(self) -> float:
            return self.cum_positive_reward

        @property
        def dump_cumulative_cost(self) -> float:
            return self.cum_cost + self.cum_misc_cost

    class ContainerNUE(ContainerEND):
        '''
        Container to keep track of rewards based on nitrogen use efficiency
        '''

        def __init__(self, timestep, costs_nitrogen=10.0):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def calculate_reward_nue(self, n_fertilized, n_output, year=None, start=None, end=None, no3_depo=None, nh4_depo=None, crop_name=None):
            if year is None or start is None or end is None:
                nue = calculate_nue(n_fertilized, n_output, no3_depo=no3_depo, nh4_depo=nh4_depo, crop_name=crop_name)
                n_surplus = get_surplus_n(n_fertilized, n_output, no3_depo=no3_depo, nh4_depo=nh4_depo, crop_name=crop_name)
            else:
                nue = calculate_nue(n_fertilized, n_output, year=year, start=start, end=end, crop_name=crop_name)
                n_surplus = get_surplus_n(n_fertilized, n_output, year=year, start=start, end=end, crop_name=crop_name)
            end_yield = super().dump_cumulative_positive_reward

            return self.formula_nue(n_surplus, nue, end_yield)

        def calculate_reward_nsurp(self, n_fertilized, n_output, no3_depo=None, nh4_depo=None, crop_name=None):
            n_surplus = get_surplus_n(n_fertilized, n_output, no3_depo=no3_depo, nh4_depo=nh4_depo, crop_name=crop_name)

            return self.nsurplus_score(n_surplus, low=15, max_dev=40)

        def calculate_reward_nue_simple(self, n_input, n_output, year=None, start=None, end=None):
            nue = calculate_nue(n_input, n_output, year=year, start=start, end=end)
            end_yield = super().dump_cumulative_positive_reward

            return self.nue_condition(nue) * end_yield

        def calculate_reward_nue_dense(self, n_input, n_output, pcse_output, year=None, start=None, end=None):
            nue = calculate_nue(n_input, n_output, year=year, start=start, end=end)
            yield_t = process_pcse.compute_growth_storage_organ(pcse_output, self.timestep)

            return self.nue_condition(nue) * yield_t

        @staticmethod
        def nsurplus_score(nsurp, low=0.0, high=40.0, max_dev=100.0):
            if low <= nsurp <= high:
                return 1.0

            # distance to nearest bound
            if nsurp < low:
                dist = low - nsurp
            else:
                dist = nsurp - high

            score = 1.0 - dist / max_dev
            return max(score, 0.0)

        def calculate_nue_term(self, n_fertilized, n_output, no3_depo=None, nh4_depo=None):
            nue = calculate_nue(n_fertilized, n_output, no3_depo=no3_depo, nh4_depo=nh4_depo)
            n_surplus = get_surplus_n(n_fertilized, n_output, no3_depo=no3_depo, nh4_depo=nh4_depo)
            return self.n_surplus_formula(n_surplus, nue)

        def calculate_profit_term(self, action, growth, price_crop, price_fertilizer):
            labor_cost = 30 if action > 0 else 0
            expense = action * price_fertilizer + labor_cost
            income = growth * price_crop
            profit = income - expense
            self.accumulate_profit(profit)
            return profit

        def calculate_final_nue_penalty_profit(
                self,
                nue_penalty,
                n_surplus_penalty,
        ):
            expense = - abs(nue_penalty) - abs(n_surplus_penalty)
            income = 0
            profit = income - expense
            self.accumulate_profit(profit)

        def budget_left_bonus(self, budget_left):
            return budget_left * self.fertilizer_price

        #  piecewise conditions
        @staticmethod
        def nue_condition(b, lower_bound=0.7, upper_bound=0.85):
            """
            For NUE reward, coefficient indicating how close the NUE in the range of lower_bound-upper_bound
            """
            if b < lower_bound:
                return upper_bound * np.exp(-10 * (lower_bound - b)) + 0.1
            elif lower_bound <= b <= upper_bound:
                return 1
            else:  # b > upper_bound
                return upper_bound * np.exp(-10 * (b - upper_bound)) + 0.1

        @staticmethod
        def nue_condition_simple(b, lower_bound=0.5, upper_bound=0.9):
            """
            For NUE reward, coefficient indicating how close the NUE in the range of lower_bound-upper_bound
            """
            if b < lower_bound:
                return 0
            elif lower_bound <= b <= upper_bound:
                return 1
            else:  # b > upper_bound
                return 0

        @staticmethod
        def n_surplus_condition(b, c):
            if 0 < b <= 40 and c == 1:
                return 1
            else:
                return 0

        @staticmethod
        def n_surplus_condition_linear(b, c, t=40):
            if c == 1:
                if 0 < b <= 40:
                    return 1
                elif 40 < b <= 40 + t:
                    return 1 - (b - 40) / t
                elif -t < b <= 0:
                    return 1 + (b / t)
            return 0

        @staticmethod
        def nue_formula(nue, nue_width=0.3):
            base_nue = max(0, min(1, 1 - (abs(nue - 0.7) - 0.2) / nue_width))
            return base_nue

        def n_surplus_penalty(self, nsurplus):
            if 0 < nsurplus <= 41:
                return 0
            else:
                return abs(nsurplus) * 3.5

        def nue_penalty(self, nue, n_output):
            if 0.5 <= nue <= 0.9:
                return 0
            elif nue <= 0.5:
                n_in_good = n_output / 0.5
                n_in_actual = n_output / nue
                n_difference = n_in_actual - n_in_good
            else:  # nue > 0.91
                n_in_good = n_output / 0.9
                n_in_actual = n_output / nue
                n_difference = n_in_good - n_in_actual
            return n_difference * 3.5

        @staticmethod
        def n_surplus_formula(n_surplus, nue, nsurp_width=100, nue_width=1):
            base_nsurp = max(0, min(1, 1 - (abs(n_surplus - 20) - 20) / nsurp_width))
            base_nue = max(0, min(1, 1 - (abs(nue - 0.7) - 0.2) / nue_width))
            return base_nsurp * base_nue

        def n_surplus_formula_piecewise(self, n_surplus, nue, nsurp_width=100, nue_width=1):
            base_nsurp = max(0, min(1, 1 - (abs(n_surplus - 20) - 20) / nsurp_width))
            base_nue = self.nue_condition_simple(nue)
            return base_nsurp * base_nue

        @staticmethod
        def normalize_yield(y, maxy=get_max_yield(), miny=get_min_yield()):
            return max(0, (y - miny) / (maxy - miny))

        @staticmethod
        def include_yield_req(req, y):
            return y if req == 1 else 0

        def formula_nue(self, n_surplus, nue, end_yield, piecewise_nue=False):
            if not piecewise_nue:
                reward_value = self.n_surplus_formula(n_surplus, nue)
            else:
                reward_value = self.n_surplus_formula_piecewise(n_surplus, nue)
            normalized_yield = self.normalize_yield(end_yield)
            return reward_value + self.include_yield_req(reward_value, normalized_yield)

        def reset(self):
            super().reset()

    # ane_reward object
    class ContainerANE:
        """
        A container to keep track of the cumulative ratio of kg grain / kg N
        """

        def __init__(self, timestep):
            self.timestep = timestep
            self.cum_growth = 0
            self.cum_baseline_growth = 0
            self.cum_amount = 0
            self.moving_ane = 0

        def reward(self, output, output_baseline, amount):
            growth = self.cumulative(output, output_baseline, amount)
            benefit = self.cum_growth - self.cum_baseline_growth

            if self.cum_amount == 0.0:
                ane = benefit / 1.0
            else:
                ane = benefit / self.cum_amount
                self.moving_ane = ane
            ane -= amount  # TODO need to add environmental penalty and reward ANE that favours TWSO
            return ane, growth

        def cumulative(self, output, output_baseline, amount, multiplier=1):
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep, multiplier)

            self.cum_growth += growth
            self.cum_baseline_growth += growth_baseline
            self.cum_amount += amount
            return growth

        def reset(self):
            self.cum_growth = 0
            self.cum_baseline_growth = 0
            self.cum_amount = 0


class ActionsContainer:
    def __init__(self):
        self.actions = 0

    def calculate_amount(self, action):
        self.actions += action

    def reset(self):
        self.actions = 0

    @property
    def get_total_fertilization(self):
        return self.actions


def calculate_nue(
        n_input,
        n_so,
        year=None,
        start=None,
        end=None,
        no3_depo=None,
        nh4_depo=None,
        crop_name=None
):
    n_in = input_nue(
        n_input,
        year=year,
        start=start,
        end=end,
        no3_depo=no3_depo,
        nh4_depo=nh4_depo,
        crop_name=crop_name
    )
    nue = n_so / n_in
    return nue


def compute_economic_reward(wso, fertilizer, price_yield_ton=400.0, price_fertilizer_ton=300.0):
    g_m2_to_ton_hectare = 0.01
    convert_wso = g_m2_to_ton_hectare * price_yield_ton
    convert_fert = g_m2_to_ton_hectare * price_fertilizer_ton
    return 0.001 * (convert_wso * wso - convert_fert * fertilizer)


def calculate_net_profit(output, amount, year, multiplier, timestep, with_year=False, with_labour=False, country='NL'):

    '''Get growth of Crop'''
    growth = process_pcse.compute_growth_storage_organ(output, timestep, multiplier)

    '''Convert growth to wheat price in the year'''
    wso_conv_eur = growth * get_wheat_price_in_kgs(year, with_year=with_year)

    '''Convert price of used fertilizer in the year'''
    n_conv_eur = get_fertilizer_price(amount, year, with_year=with_year)

    # '''Convert labour price based on year'''
    # labour_conv_eur = get_labour_price(year, with_labour=with_labour)
    #
    # '''Flag for fertilization action'''
    # labour_flag = 1 if amount else 0

    reward = wso_conv_eur - n_conv_eur  # - labour_conv_eur * labour_flag

    return reward, growth


def annual_price_wheat_per_ton(year):
    prices = {
        1989: 177.16, 1990: 168.27, 1991: 174.05, 1992: 171.61, 1993: 148.94, 1994: 135.27, 1995: 131.89,
        1996: 130.50, 1997: 120.84, 1998: 111.39, 1999: 111.62, 2000: 116.23, 2001: 112.17, 2002: 102.89,
        2003: 114.73, 2004: 116.95, 2005: 96.73, 2006: 117.95, 2007: 180.78, 2008: 169.84, 2009: 112.23,
        2010: 152.00, 2011: 197.5, 2012: 219.28, 2013: 203.23, 2014: 164.12, 2015: 159.43, 2016: 145.17,
        2017: 154.62, 2018: 176.23, 2019: 172.23, 2020: 181.67, 2021: 233.84, 2022: 312.56, 2023: 227.56
    }

    return prices[year]


def get_wheat_price_in_kgs(year, with_year=False, price_per_ton=157.75):
    if not with_year:
        return price_per_ton * 0.001
    return annual_price_wheat_per_ton(year) * 0.001


def get_nitrogen_price_in_kgs(year, with_year=False, price_per_quintal=20.928):
    if not with_year:
        return price_per_quintal * 0.01
    return annual_price_nitrogen_per_quintal(year) * 0.01


def annual_price_nitrogen_per_quintal(year):
    prices = {
        1989: 11.61, 1990: 11.61, 1991: 12.20, 1992: 11.04, 1993: 10.07, 1994: 10.24, 1995: 12.58,
        1996: 13.22, 1997: 11.49, 1998: 10.55, 1999: 9.48, 2000: 13.09, 2001: 15.60, 2002: 14.28,
        2003: 15.18, 2004: 15.89, 2005: 17.11, 2006: 18.85, 2007: 19.81, 2008: 33.12, 2009: 21.37,
        2010: 21.71, 2011: 29.39, 2012: 29.38, 2013: 27.13, 2014: 27.74, 2015: 27.85, 2016: 21.49,
        2017: 21.37, 2018: 22.90, 2019: 24.17, 2020: 20.49, 2021: 35.71, 2022: 76.62, 2023: 38.14
    }

    return prices[year]


def labour_index_per_year(year):
    """
    Linear function to estimate hourly labour costs per year in the Netherlands
    From https://ycharts.com/indicators/netherlands_labor_cost_index
    """
    index = 2.0016 * year - 3941.4

    index = index / 100  # convert to percentage
    return index


"""
Calculations for getting prices in the year
"""


def get_fertilizer_price(action, year, with_year=False):
    """
    Price of N fertilizer per kg in the year

    :param action: agent's action
    :param year: year of the action
    :return: nitrogen price per kg
    """
    amount = action * 10  # action to kg/ha
    price = get_nitrogen_price_in_kgs(year, with_year)

    return amount * price


def get_labour_price(year, base_labour_cost_index=28.9, time_per_hectare=0.0834, with_labour=False):
    """
    Price of hourly labour per year, considering the European labour cost index

    :param base_labour_cost_index: labour cost in the base year of the index
    :param time_per_hectare: assumption of the time needed to fertilize one hectare of land, currently defaults to
            5 minutes per hectare.
    :return: price of labour in euros
    """

    if with_labour:
        return 0

    return (base_labour_cost_index * labour_index_per_year(year) + base_labour_cost_index) * time_per_hectare
