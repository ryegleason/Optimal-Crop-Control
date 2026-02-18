import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from multiprocessing import Pool
import pickle
from tqdm import tqdm

from forward_models import RAIN_EXISTENCE_RATE, SOIL_DEPTH_MM, FIELD_CAPACITY, SOIL_POROSITY, generate_rain, hydrology_model, soil_organic_model, \
    inorganic_nitrogen_model
from dp_solver import solve_dp, calculate_probabilities_of_violation, KernelMetadata

rng = np.random.default_rng(12345)

CONTROL_TIMESTEP_DAYS = 36
NUM_CONTROL_STEPS = 1
# corn prices are measured in USD / 56 lbs at 15.5% moisture (by mass)
CORN_PRICE_USD_PER_G_GRAIN = 4.80 / (1 - 0.155) / 25401.2
N_PRICE_USD_PER_G = 0.0011
N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 = 1200
NNI_TO_RELATIVE_YIELD_CORRELATION = 0.99
MAX_BIOMASS_G_DRY_MASS_PER_M2 = 3000

INORGANIC_N_MODEL_DT_DAYS = 1.0 / 24.0 * 2.0 / 6.0

# Derived using PorporatoCharacterization.py
INITIAL_LITTER_CARBON_GC_PER_M3 = 400.0
INITIAL_LITTER_NITROGEN_GN_PER_M3 = 22.0
INITIAL_MICROBIAL_CARBON_GC_PER_M3 = 130.0
INITIAL_HUMUS_CARBON_GC_PER_M3 = 5500.0

NUM_DAYS_TOTAL = CONTROL_TIMESTEP_DAYS * NUM_CONTROL_STEPS
LEACHING_CONCENTRATION_LIMIT_MG_PER_LITER = 10
LEACHING_AVERAGE_WINDOW_DAYS = 3
LEACHING_VIOLATION_PENALTY_USD_PER_M2 = (N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN) * 1.0
FORECAST_LOOKAHEAD_DAYS = 3
PROBABILITY_OF_CLEAR_FORECAST = (1 - RAIN_EXISTENCE_RATE) ** FORECAST_LOOKAHEAD_DAYS

biomass_g_per_m2_to_critical_nitrogen_gN_per_m2 = np.vectorize(lambda w: 0.187 * (w ** 0.63))
day_to_biomass_g_per_m2 = np.vectorize(lambda day: day / NUM_DAYS_TOTAL * MAX_BIOMASS_G_DRY_MASS_PER_M2 + 0.001)
day_to_plant_N_demand_gN_per_m3_per_day = np.vectorize(lambda day: 0.02565 * np.exp(-0.00095 * day_to_biomass_g_per_m2(day)) * MAX_BIOMASS_G_DRY_MASS_PER_M2 / NUM_DAYS_TOTAL / (SOIL_DEPTH_MM / 1000.)) #MAX_BIOMASS_G_DRY_MASS_PER_M2 / NUM_DAYS_TOTAL is dW/dt

water_at_field_capacity_L_per_m3 = FIELD_CAPACITY * SOIL_POROSITY * 1000.0
leaching_safe_N_threshold_gN_per_m3 = LEACHING_CONCENTRATION_LIMIT_MG_PER_LITER * water_at_field_capacity_L_per_m3 / 1000.0

def nitrogen_deficit_cost(plant_total_N_gN_per_m3: npt.NDArray[np.float64], initial_day: float, dt_days: float = 1.0) -> float:
    range_len_days = len(plant_total_N_gN_per_m3) * dt_days
    time_days = np.linspace(initial_day, initial_day + range_len_days, len(plant_total_N_gN_per_m3))
    plant_biomass_g_dry_mass_per_m2 = day_to_biomass_g_per_m2(time_days)
    critical_N_gN_per_m3 = biomass_g_per_m2_to_critical_nitrogen_gN_per_m2(plant_biomass_g_dry_mass_per_m2) / (SOIL_DEPTH_MM / 1000.)
    NNI = np.clip(plant_total_N_gN_per_m3 / critical_N_gN_per_m3, a_min=0.0, a_max=1.0)

    decrease_to_full_season_NNI = (1.0 - np.average(NNI)) * range_len_days / NUM_DAYS_TOTAL
    return decrease_to_full_season_NNI * NNI_TO_RELATIVE_YIELD_CORRELATION * N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN

def leaching_limit_violated(leakage_rate_mm_per_day: npt.NDArray[np.float64], ammonium_leaching_gN_per_m3_per_day: npt.NDArray[np.float64], nitrate_leaching_gN_per_m3_per_day: npt.NDArray[np.float64], inorganic_n_model_dt_days: float = INORGANIC_N_MODEL_DT_DAYS) -> bool:
    assert ammonium_leaching_gN_per_m3_per_day.shape == nitrate_leaching_gN_per_m3_per_day.shape
    for i in range(0, NUM_DAYS_TOTAL, LEACHING_AVERAGE_WINDOW_DAYS):
        total_leakage_L_per_m3 = np.sum(leakage_rate_mm_per_day[i:(i+LEACHING_AVERAGE_WINDOW_DAYS)])
        inorganic_n_model_lower_limit = int(i / inorganic_n_model_dt_days)
        inorganic_n_model_upper_limit = int((i + LEACHING_AVERAGE_WINDOW_DAYS) / inorganic_n_model_dt_days)
        if total_leakage_L_per_m3 > 0 and (np.sum(ammonium_leaching_gN_per_m3_per_day[inorganic_n_model_lower_limit:inorganic_n_model_upper_limit]) + np.sum(nitrate_leaching_gN_per_m3_per_day[inorganic_n_model_lower_limit:inorganic_n_model_upper_limit])) * inorganic_n_model_dt_days / total_leakage_L_per_m3 > (LEACHING_CONCENTRATION_LIMIT_MG_PER_LITER / 1000.):
            return True
    return False

def simulate_from_rain_through_SOM(rain_quantity_mm: npt.NDArray, initial_soil_moisture: float, initial_litter_carbon_gC_per_m3: float = INITIAL_LITTER_CARBON_GC_PER_M3, initial_litter_nitrogen_gN_per_m3: float = INITIAL_LITTER_NITROGEN_GN_PER_M3, initial_microbial_carbon_gC_per_m3: float = INITIAL_MICROBIAL_CARBON_GC_PER_M3, initial_humus_carbon_gC_per_m3: float = INITIAL_HUMUS_CARBON_GC_PER_M3) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    num_days = len(rain_quantity_mm)
    soil_moisture, _, _, transpiration_rate_mm_per_day, leakage_rate_mm_per_day, moisture_effect_on_decomposition_factor, _ = hydrology_model(initial_soil_moisture, rain_quantity_mm)

    _, _, biomass_carbon_g_per_m3, _, net_flux_to_mineralized_nitrogen_g_per_m3_per_day = soil_organic_model(initial_litter_carbon_gC_per_m3, initial_litter_nitrogen_gN_per_m3, initial_microbial_carbon_gC_per_m3, initial_humus_carbon_gC_per_m3, moisture_effect_on_decomposition_factor, np.zeros(num_days), np.zeros(num_days))
    return leakage_rate_mm_per_day, transpiration_rate_mm_per_day, soil_moisture, net_flux_to_mineralized_nitrogen_g_per_m3_per_day, biomass_carbon_g_per_m3

def simulate_from_SOM(initial_ammonium_gN_per_m3: float, initial_nitrate_gN_per_m3: float, initial_day: float, leakage_mm_per_day: npt.NDArray[np.float64], transpiration_mm_per_day: npt.NDArray[np.float64], soil_moisture: npt.NDArray[np.float64], net_flux_to_mineral_nitrogen_gN_per_m3_per_day: npt.NDArray[np.float64], microbial_carbon_gC_per_m3: npt.NDArray[np.float64], inorganic_n_model_dt_days: float = INORGANIC_N_MODEL_DT_DAYS) -> Tuple[float, float, float, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    ammonium_nitrogen_g_per_m3, nitrate_nitrogen_g_per_m3, ammonium_leaching_gN_per_m3_per_day, nitrate_leaching_gN_per_m3_per_day, plant_passive_uptake_of_ammonium_gN_per_m3_per_day, plant_passive_uptake_of_nitrate_gN_per_m3_per_day, plant_active_uptake_of_ammonium_gN_per_m3_per_day, plant_active_uptake_of_nitrate_gN_per_m3_per_day, assumptions_violation, _ = inorganic_nitrogen_model(initial_ammonium_gN_per_m3, initial_nitrate_gN_per_m3, initial_day, leakage_mm_per_day, transpiration_mm_per_day, soil_moisture, net_flux_to_mineral_nitrogen_gN_per_m3_per_day, microbial_carbon_gC_per_m3, day_to_plant_N_demand_gN_per_m3_per_day, output_dt_days=inorganic_n_model_dt_days)

    plant_uptake_gN_per_m3_per_day = plant_passive_uptake_of_ammonium_gN_per_m3_per_day + plant_active_uptake_of_ammonium_gN_per_m3_per_day + plant_passive_uptake_of_nitrate_gN_per_m3_per_day + plant_active_uptake_of_nitrate_gN_per_m3_per_day
    plant_cumulative_N_uptake_gN_per_m3 = np.cumsum(plant_uptake_gN_per_m3_per_day * inorganic_n_model_dt_days)

    return float(soil_moisture[-1]), ammonium_nitrogen_g_per_m3[-1], nitrate_nitrogen_g_per_m3[-1], plant_cumulative_N_uptake_gN_per_m3, ammonium_leaching_gN_per_m3_per_day, nitrate_leaching_gN_per_m3_per_day

def simulate_all(num_days: int, initial_soil_moisture: float, initial_ammonium_gN_per_m3: float, initial_nitrate_gN_per_m3: float, initial_litter_carbon_gC_per_m3: float = INITIAL_LITTER_CARBON_GC_PER_M3, initial_litter_nitrogen_gN_per_m3: float = INITIAL_LITTER_NITROGEN_GN_PER_M3, initial_microbial_carbon_gC_per_m3: float = INITIAL_MICROBIAL_CARBON_GC_PER_M3, initial_humus_carbon_gC_per_m3: float = INITIAL_HUMUS_CARBON_GC_PER_M3):
    rain_quantity_mm = generate_rain(num_days)
    hydrology_and_SOM_results = simulate_from_rain_through_SOM(rain_quantity_mm, initial_soil_moisture, initial_litter_carbon_gC_per_m3, initial_litter_nitrogen_gN_per_m3, initial_microbial_carbon_gC_per_m3, initial_humus_carbon_gC_per_m3)
    return simulate_from_SOM(initial_ammonium_gN_per_m3, initial_nitrate_gN_per_m3, *hydrology_and_SOM_results)

MAX_AMMONIUM = 8.4 / (SOIL_DEPTH_MM / 1000)
MAX_NITRATE = 33.63 / (SOIL_DEPTH_MM / 1000)
AMMONIUM_NUM_STEPS = 10
NITRATE_NUM_STEPS = 25
AMMONIUM_GRID_SIZE = MAX_AMMONIUM / AMMONIUM_NUM_STEPS
NITRATE_GRID_SIZE = MAX_NITRATE / NITRATE_NUM_STEPS

MAX_MOISTURE = 0.8
MOISTURE_GRID_SIZE = 0.05
MOISTURE_NUM_STEPS = int(MAX_MOISTURE / MOISTURE_GRID_SIZE)

MAX_ACCUMULATED_N = 100
ACCUMULATED_N_GRID_SIZE = 2
ACCUMULATED_N_NUM_STEPS = int(MAX_ACCUMULATED_N / ACCUMULATED_N_GRID_SIZE)
MAX_SINGLE_STEP_ACCUMULATED_N = min(MAX_ACCUMULATED_N, CONTROL_TIMESTEP_DAYS * 15)
SINGLE_STEP_ACCUMULATED_N_NUM_STEPS = int(MAX_SINGLE_STEP_ACCUMULATED_N / ACCUMULATED_N_GRID_SIZE)

NUM_TRIALS_PER_INITIAL_CONDITION = 120

def values_to_indices(moisture: float, ammonium: float, nitrate: float) -> Tuple[int, int, int]:
    # 0 cell has to be a different size so it's completely leaching-safe
    if ammonium < leaching_safe_N_threshold_gN_per_m3 / 2.:
        ammonium_idx = 0
    elif ammonium <= AMMONIUM_GRID_SIZE:
        ammonium_idx = 1
    else:
        ammonium_idx = min(AMMONIUM_NUM_STEPS - 1, int(ammonium / AMMONIUM_GRID_SIZE))
    if nitrate < leaching_safe_N_threshold_gN_per_m3 / 2.:
        nitrate_idx = 0
    elif nitrate <= NITRATE_GRID_SIZE:
        nitrate_idx = 1
    else:
        nitrate_idx = min(NITRATE_NUM_STEPS - 1, int(nitrate / NITRATE_GRID_SIZE))
    moisture_idx = min(MOISTURE_NUM_STEPS - 1, int(moisture / MOISTURE_GRID_SIZE))
    return moisture_idx, ammonium_idx, nitrate_idx

def indices_to_values(day_idx: int, moisture_idx: int, ammonium_idx: int, nitrate_idx: int) -> Tuple[float, float, float, float]:
    initial_day = float(day_idx * CONTROL_TIMESTEP_DAYS)
    moisture = MOISTURE_GRID_SIZE / 2.0 + MOISTURE_GRID_SIZE * moisture_idx
    ammonium = AMMONIUM_GRID_SIZE * ammonium_idx
    nitrate = NITRATE_GRID_SIZE * nitrate_idx
    return initial_day, moisture, ammonium, nitrate

def sample_from_indices(moisture_idx: int, ammonium_idx: int, nitrate_idx: int) -> Tuple[float, float, float]:
    sampled_moisture = rng.uniform(MOISTURE_GRID_SIZE * moisture_idx, MOISTURE_GRID_SIZE * (moisture_idx + 1))

    if ammonium_idx == 0:
        sampled_ammonium = rng.uniform(0, leaching_safe_N_threshold_gN_per_m3 / 2.)
    elif ammonium_idx == 1:
        sampled_ammonium = rng.uniform(leaching_safe_N_threshold_gN_per_m3 / 2., AMMONIUM_GRID_SIZE * 2.)
    else:
        sampled_ammonium = rng.uniform(AMMONIUM_GRID_SIZE * ammonium_idx, AMMONIUM_GRID_SIZE * (ammonium_idx + 1))

    if nitrate_idx == 0:
        sampled_nitrate = rng.uniform(0, leaching_safe_N_threshold_gN_per_m3 / 2.)
    elif nitrate_idx == 1:
        sampled_nitrate = rng.uniform(leaching_safe_N_threshold_gN_per_m3 / 2., NITRATE_GRID_SIZE * 2.)
    else:
        sampled_nitrate = rng.uniform(NITRATE_GRID_SIZE * nitrate_idx, NITRATE_GRID_SIZE * (nitrate_idx + 1))
    return sampled_moisture, sampled_ammonium, sampled_nitrate

def accumulated_N_value_to_index(accumulated_N: float) -> int:
    return min(ACCUMULATED_N_NUM_STEPS - 1, int(accumulated_N / ACCUMULATED_N_GRID_SIZE))

def accumulated_N_index_to_value(accumulated_N_idx: int) -> float:
    return ACCUMULATED_N_GRID_SIZE / 2.0 + ACCUMULATED_N_GRID_SIZE * accumulated_N_idx

def single_step_accumulated_N_value_to_index(accumulated_N: float) -> int:
    return min(SINGLE_STEP_ACCUMULATED_N_NUM_STEPS - 1, int(accumulated_N / ACCUMULATED_N_GRID_SIZE))

def transition_probabilities_from_initial_condition(inputs_tuple: Tuple[int, int, int, int, int]) -> Tuple[int, int, int, int, int, npt.NDArray[np.uint8], npt.NDArray[np.float64]]:
    initial_day_idx, initial_forecast_idx, initial_moisture_idx, initial_ammonium_idx, initial_nitrate_idx = inputs_tuple
    initial_day  = indices_to_values(initial_day_idx, 0, 0, 0)[0]
    initial_forecast = "early_rain" if initial_forecast_idx == 1 else "no_early_rain"
    end_state_counts = np.zeros([2, SINGLE_STEP_ACCUMULATED_N_NUM_STEPS, MOISTURE_NUM_STEPS, AMMONIUM_NUM_STEPS, NITRATE_NUM_STEPS], np.uint8)
    costs = np.zeros([ACCUMULATED_N_NUM_STEPS], np.float64)
    for SOM_and_hydrology_results in SOM_and_hydrology_per_moisture[initial_moisture_idx][initial_forecast]:
        _, initial_ammonium, initial_nitrate = sample_from_indices(0, initial_ammonium_idx, initial_nitrate_idx)
        end_moisture, end_ammonium, end_nitrate, plant_cumulative_uptake_gN_per_m3, ammonium_leaching_gN_per_m3_per_day, nitrate_leaching_gN_per_m3_per_day = simulate_from_SOM(initial_ammonium, initial_nitrate, initial_day, *SOM_and_hydrology_results, inorganic_n_model_dt_days=INORGANIC_N_MODEL_DT_DAYS)
        N_uptake_over_period_gN_per_m3 = plant_cumulative_uptake_gN_per_m3[-1]
        is_leaching_limit_violated = int(leaching_limit_violated(SOM_and_hydrology_results[0], ammonium_leaching_gN_per_m3_per_day, nitrate_leaching_gN_per_m3_per_day))
        end_state_counts[is_leaching_limit_violated, single_step_accumulated_N_value_to_index(N_uptake_over_period_gN_per_m3), *values_to_indices(end_moisture, end_ammonium, end_nitrate)] += 1
        for initial_accumulated_N_index in range(ACCUMULATED_N_NUM_STEPS):
            costs[initial_accumulated_N_index] += nitrogen_deficit_cost(plant_cumulative_uptake_gN_per_m3 + accumulated_N_index_to_value(initial_accumulated_N_index), initial_day, dt_days=INORGANIC_N_MODEL_DT_DAYS)
    costs /= len(SOM_and_hydrology_per_moisture[initial_moisture_idx][initial_forecast])
    return initial_day_idx, initial_forecast_idx, initial_moisture_idx, initial_ammonium_idx, initial_nitrate_idx, end_state_counts, costs

class InitialConditionIterator:
    def __iter__(self):
        self.initial_day_idx = 0
        self.initial_forecast_idx = 0
        self.moisture_idx = 0
        self.ammonium_idx = 0
        self.nitrate_idx = 0
        return self

    def __next__(self):
        if self.initial_day_idx >= NUM_CONTROL_STEPS:
            raise StopIteration
        out = (self.initial_day_idx, self.initial_forecast_idx, self.moisture_idx, self.ammonium_idx, self.nitrate_idx)
        self.nitrate_idx += 1
        if self.nitrate_idx >= NITRATE_NUM_STEPS:
            self.nitrate_idx = 0
            self.ammonium_idx += 1
            if self.ammonium_idx >= AMMONIUM_NUM_STEPS:
                self.ammonium_idx = 0
                self.moisture_idx += 1
                if self.moisture_idx >= MOISTURE_NUM_STEPS:
                    self.moisture_idx = 0
                    self.initial_forecast_idx += 1
                    if self.initial_forecast_idx >= 2:
                        self.initial_forecast_idx = 0
                        self.initial_day_idx += 1
        return out

KERNEL_METADATA = KernelMetadata(NUM_TRIALS_PER_INITIAL_CONDITION, NUM_CONTROL_STEPS, ACCUMULATED_N_NUM_STEPS, MOISTURE_NUM_STEPS, AMMONIUM_NUM_STEPS, NITRATE_NUM_STEPS, SINGLE_STEP_ACCUMULATED_N_NUM_STEPS)
n_indices_to_values = lambda ammonium_idx, nitrate_idx: indices_to_values(0, 0, ammonium_idx, nitrate_idx)[2:4]    
PARETO_SWEEP_SAMPLES = 25
LEACHING_PENALTIES_USD_PER_M2 = np.linspace(0, LEACHING_VIOLATION_PENALTY_USD_PER_M2 * 2.0, PARETO_SWEEP_SAMPLES)

if __name__ == "__main__":
    rainfalls = {"early_rain": [], "no_early_rain": []}
    for forecast in ["early_rain", "no_early_rain"]:
        i = 0
        while i < NUM_TRIALS_PER_INITIAL_CONDITION:
            candidate_rain = generate_rain(CONTROL_TIMESTEP_DAYS)
            if (any(candidate_rain[:FORECAST_LOOKAHEAD_DAYS] > 0) and forecast == "early_rain") or (
                    (not any(candidate_rain[:FORECAST_LOOKAHEAD_DAYS] > 0)) and forecast == "no_early_rain"):
                rainfalls[forecast].append(candidate_rain)
                i += 1
    
    SOM_and_hydrology_per_moisture = []
    for moisture_idx_for_hydrology in range(MOISTURE_NUM_STEPS):
        trial_results = {"early_rain": [], "no_early_rain": []}
        for forecast in ["early_rain", "no_early_rain"]:
            for rainfall in rainfalls[forecast]:
                initial_moisture_for_hydrology = sample_from_indices(moisture_idx_for_hydrology, 0, 0)[0]
                result = simulate_from_rain_through_SOM(rainfall, initial_moisture_for_hydrology)
                trial_results[forecast].append(result)
        SOM_and_hydrology_per_moisture.append(trial_results)
    # time, forecast, moisture, starting ammonium, starting nitrate, -> does leaching limit get violated, n accumulation, end moisture, end ammonium, end nitrate
    transition_counts = np.zeros([NUM_CONTROL_STEPS, 2, MOISTURE_NUM_STEPS, AMMONIUM_NUM_STEPS, NITRATE_NUM_STEPS, 2, SINGLE_STEP_ACCUMULATED_N_NUM_STEPS, MOISTURE_NUM_STEPS, AMMONIUM_NUM_STEPS, NITRATE_NUM_STEPS], np.uint8)
    expected_plant_N_deficit_cost_USD = np.zeros([NUM_CONTROL_STEPS, ACCUMULATED_N_NUM_STEPS, 2, MOISTURE_NUM_STEPS, AMMONIUM_NUM_STEPS, NITRATE_NUM_STEPS], np.float64)
    
    with Pool(processes = 12) as pool: # parallel processing
        for transition_probability_results in tqdm(pool.imap_unordered(transition_probabilities_from_initial_condition, iter(InitialConditionIterator())), total=(NUM_CONTROL_STEPS * 2 * MOISTURE_NUM_STEPS * AMMONIUM_NUM_STEPS * NITRATE_NUM_STEPS)):
            transition_initial_day_idx, transition_initial_forecast_idx, transition_initial_moisture_idx, transition_initial_ammonium_idx, transition_initial_nitrate_idx, transition_end_state_counts, costs_by_accumulated_N = transition_probability_results
            transition_counts[transition_initial_day_idx, transition_initial_forecast_idx, transition_initial_moisture_idx, transition_initial_ammonium_idx, transition_initial_nitrate_idx] = transition_end_state_counts
            expected_plant_N_deficit_cost_USD[transition_initial_day_idx, :, transition_initial_forecast_idx, transition_initial_moisture_idx, transition_initial_ammonium_idx, transition_initial_nitrate_idx] = costs_by_accumulated_N
    
    for i in range(NUM_CONTROL_STEPS):
        for j in range(2):
            for k in range(MOISTURE_NUM_STEPS):
                for l in range(AMMONIUM_NUM_STEPS):
                    for m in range(NITRATE_NUM_STEPS):
                        if np.sum(transition_counts[i, j, k, l, m]) != NUM_TRIALS_PER_INITIAL_CONDITION:
                            print("Failed sanity check at [{}, {}, {}, {}]".format(i, j, k, l))
                            break
    
    with open(str(CONTROL_TIMESTEP_DAYS) + '_transition_counts.pickle', 'wb') as f:
        pickle.dump(transition_counts, f, pickle.HIGHEST_PROTOCOL)
    with open(str(CONTROL_TIMESTEP_DAYS) + '_expected_plant_N_deficit.pickle', 'wb') as f:
        pickle.dump(expected_plant_N_deficit_cost_USD, f, pickle.HIGHEST_PROTOCOL)

    pareto_sweep_expected_profit = np.ndarray([PARETO_SWEEP_SAMPLES, 2, MOISTURE_NUM_STEPS, AMMONIUM_NUM_STEPS, NITRATE_NUM_STEPS]) # sample idx, forecast, moisture, ammonium, nitrate
    pareto_sweep_probability_of_violation = np.ndarray([PARETO_SWEEP_SAMPLES, 2, MOISTURE_NUM_STEPS, AMMONIUM_NUM_STEPS, NITRATE_NUM_STEPS]) # sample idx, forecast, moisture, ammonium, nitrate
    
    for i in tqdm(range(PARETO_SWEEP_SAMPLES)):
        sweep_optimal_cost_to_go_USD_per_m2, sweep_optimal_ammonium_add_in_cells, sweep_optimal_nitrate_add_in_cells = solve_dp(transition_counts, expected_plant_N_deficit_cost_USD, KERNEL_METADATA, N_PRICE_USD_PER_G, (SOIL_DEPTH_MM / 1000.), n_indices_to_values, LEACHING_PENALTIES_USD_PER_M2[i], PROBABILITY_OF_CLEAR_FORECAST)
        pareto_sweep_expected_profit[i] = N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN - sweep_optimal_cost_to_go_USD_per_m2[0, 0, 0]
        pareto_sweep_probability_of_violation[i] = calculate_probabilities_of_violation(transition_counts, sweep_optimal_ammonium_add_in_cells, sweep_optimal_nitrate_add_in_cells, KERNEL_METADATA, PROBABILITY_OF_CLEAR_FORECAST)[0,0]
    
    
    with open(str(CONTROL_TIMESTEP_DAYS) + '_pareto_sweep_expected_profit.pickle', 'wb') as f:
        pickle.dump(pareto_sweep_expected_profit, f, pickle.HIGHEST_PROTOCOL)
    with open(str(CONTROL_TIMESTEP_DAYS) + '_pareto_sweep_probability_of_violation.pickle', 'wb') as f:
        pickle.dump(pareto_sweep_probability_of_violation, f, pickle.HIGHEST_PROTOCOL)
