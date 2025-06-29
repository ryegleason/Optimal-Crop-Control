import numpy as np
import numpy.typing as npt
from typing import Tuple

rng = np.random.default_rng(12345)

SOIL_POROSITY = 0.35 # Porporato had 0.4
SOIL_DEPTH_MM = 300 # Porporato had 800
SOIL_VOID_SPACE_MM = SOIL_POROSITY * SOIL_DEPTH_MM
HYGROSCOPIC_POINT = 0.04 # Porporato had 0.02
WILTING_POINT = 0.05 # Porporato had 0.065
MAX_MOISTURE_WITH_CLOSED_STOMATA = 0.16 # Porporato had 0.17
FIELD_CAPACITY = 0.35 # Porporato had 0.3
MAXIMUM_EVAPORATION_RATE_MM_PER_DAY = 0.1 # Porporato had 1.0
MAXIMUM_TRANSPIRATION_RATE_MM_PER_DAY = 3.6 # Porporato had 3.5
SATURATED_HYDRAULIC_CONDUCTIVITY_MM_PER_DAY = 1000 # Porporato had 1100
PORE_SIZE_DISTRIBUTION_INDEX = 4.05 # 4.05 for porporato conditions
BETA = 4.0 * PORE_SIZE_DISTRIBUTION_INDEX + 4.0
ADDED_RESIDUE_CARBON_G_PER_M2_PER_DAY = 1.5
PROPORTION_BIOMASS_DYING_PER_DAY = 0.0085
PROPORTION_LITTER_DECOMPOSING_PER_BIOMASS_PER_DAY = 0.000065
PROPORTION_HUMUS_DECOMPOSING_PER_BIOMASS_PER_DAY = 0.0000025
MAX_PROPORTION_AMMONIUM_IMMOBILIZED_PER_DAY = 1.0
MAX_PROPORTION_NITRATE_IMMOBILIZED_PER_DAY = 1.0
PROPORTION_AMMONIUM_NITRIFIED_PER_BIOMASS_PER_DAY = 0.006 # porporato had 0.6
DISSOLVED_FRACTION_AMMONIUM = 0.05
DISSOLVED_FRACTION_NITRATE = 1.0 # https://link.springer.com/chapter/10.1007/978-94-007-0394-0_23/tables/4 has 0.1 but that can't be right
RESPIRATED_FRACTION_OF_DECOMPOSED_CARBON = 0.6
NON_RESPIRATED_FRACTION_OF_DECOMPOSED_CARBON = 1 - RESPIRATED_FRACTION_OF_DECOMPOSED_CARBON
MAX_HUMIFIED_FRACTION_OF_DECOMPOSED_LITTER = 0.25
ADDED_RESIDUE_CN_RATIO = 58
BIOMASS_CN_RATIO = 11.5
HUMUS_CN_RATIO = 22 # porporato had 22
EXCESS_BIO_NITROGEN_PER_HUMUS_CARBON_DECOMPOSED = 1 / HUMUS_CN_RATIO - NON_RESPIRATED_FRACTION_OF_DECOMPOSED_CARBON / BIOMASS_CN_RATIO
# PLANT_AMMONIUM_DEMAND_GRAMS_N_PER_M3_PER_DAY = 0.2
# PLANT_NITRATE_DEMAND_GRAMS_N_PER_M3_PER_DAY = 0.5
PLANT_BIOMASS_G_DRY_MASS_PER_M2 = 600 # roughly 17L stage
DIFFUSION_COEFFICIENT_MM_PER_DAY = 100 # possibly should be 100 ? Porporato has 0.1 meters/day
DIFFUSION_MOISTURE_DEPENDENCE_EXPONENT = 1.5 # Porporato had 3
RAIN_EXISTENCE_RATE = 0.26 # porporato had 0.23, ag paper had 0.3
RAIN_MEAN_DEPTH_MM = 13.5 # porporato had 11, ag paper had 15.5
f_d = np.vectorize(lambda s : s / FIELD_CAPACITY if s <= FIELD_CAPACITY else FIELD_CAPACITY / s)
f_n = np.vectorize(lambda s: s / FIELD_CAPACITY if s <= FIELD_CAPACITY else (1 - s)/(1- FIELD_CAPACITY))

INORGANIC_N_MODEL_DT_DAYS = 1.0 / 24.0 * 2.0 / 6.0 # dt of 20 minutes

def generate_rain(num_days: int):
    rain_events = rng.random(num_days) < RAIN_EXISTENCE_RATE
    rain_quantity_mm = rng.exponential(RAIN_MEAN_DEPTH_MM, num_days) * rain_events
    return rain_quantity_mm

def hydrology_model(initial_soil_moisture: float, rain_quantity_mm: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    num_days = len(rain_quantity_mm)

    # Define constants for computational efficiency
    LEAKAGE_COEFF = SATURATED_HYDRAULIC_CONDUCTIVITY_MM_PER_DAY / (np.exp(BETA * (1 - FIELD_CAPACITY)) - 1)
    TRANSPIRATION_SLOPE = MAXIMUM_TRANSPIRATION_RATE_MM_PER_DAY / (MAX_MOISTURE_WITH_CLOSED_STOMATA - WILTING_POINT)

    # States and intermediate variables
    soil_moisture = np.zeros(num_days)
    infiltration_rate_mm_per_day = np.zeros(num_days)
    evaporation_rate_mm_per_day = np.zeros(num_days)
    transpiration_rate_mm_per_day = np.zeros(num_days)
    leakage_rate_mm_per_day = np.zeros(num_days)

    soil_moisture[0] = initial_soil_moisture

    for i in range(num_days):
        infiltration_rate_mm_per_day[i] = min(rain_quantity_mm[i], SOIL_VOID_SPACE_MM * (1 - soil_moisture[i]))

        # evaporation
        if soil_moisture[i] < HYGROSCOPIC_POINT:
            evaporation_rate_mm_per_day[i] = 0
        elif soil_moisture[i] <= WILTING_POINT:
            evaporation_rate_mm_per_day[i] = MAXIMUM_EVAPORATION_RATE_MM_PER_DAY * (soil_moisture[i] - HYGROSCOPIC_POINT)
        else:
            evaporation_rate_mm_per_day[i] = MAXIMUM_EVAPORATION_RATE_MM_PER_DAY

        # Transpiration
        if soil_moisture[i] <= WILTING_POINT:
            transpiration_rate_mm_per_day[i] = 0
        elif soil_moisture[i] <= MAX_MOISTURE_WITH_CLOSED_STOMATA:
            transpiration_rate_mm_per_day[i] = TRANSPIRATION_SLOPE * (soil_moisture[i] - WILTING_POINT)
        else:
            transpiration_rate_mm_per_day[i] = MAXIMUM_TRANSPIRATION_RATE_MM_PER_DAY

        # leakage/percolation
        leakage_rate_mm_per_day[i] = LEAKAGE_COEFF * (np.exp(BETA * (soil_moisture[i] - FIELD_CAPACITY)) - 1)
        if soil_moisture[i] - (leakage_rate_mm_per_day[i] / SOIL_VOID_SPACE_MM) < FIELD_CAPACITY: # make sure we don't drain more than is possible in very wet fields
            leakage_rate_mm_per_day[i] = max(0.0, (soil_moisture[i] - FIELD_CAPACITY) * SOIL_VOID_SPACE_MM)

        if i + 1 < num_days:
            soil_moisture[i+1] = soil_moisture[i] + (infiltration_rate_mm_per_day[i] - evaporation_rate_mm_per_day[i] - transpiration_rate_mm_per_day[i] - leakage_rate_mm_per_day[i]) / SOIL_VOID_SPACE_MM

    moisture_effect_on_decomposition_factor = f_d(soil_moisture)
    moisture_effect_on_nitrification_factor = f_n(soil_moisture)

    if any(infiltration_rate_mm_per_day < 0) or any(evaporation_rate_mm_per_day < 0) or any(transpiration_rate_mm_per_day < 0) or any(leakage_rate_mm_per_day < 0) or any(soil_moisture < 0) or any(soil_moisture > 1) or any(moisture_effect_on_decomposition_factor < 0) or any(moisture_effect_on_decomposition_factor > 1) or any(moisture_effect_on_nitrification_factor < 0) or any(moisture_effect_on_nitrification_factor > 1):
        print("Sanity check failed in hydrology model!")

    return soil_moisture, infiltration_rate_mm_per_day, evaporation_rate_mm_per_day, transpiration_rate_mm_per_day, leakage_rate_mm_per_day, moisture_effect_on_decomposition_factor, moisture_effect_on_nitrification_factor

def soil_organic_model(initial_litter_carbon_gC_per_m3: float, initial_litter_nitrogen_gN_per_m3: float, initial_microbial_carbon_gC_per_m3: float, initial_humus_carbon_gC_per_m3: float, moisture_decomposition_factor: npt.NDArray[np.float64], added_residue_nitrogen_gN_per_m3_per_day: npt.NDArray[np.float64], added_residue_carbon_gC_per_m3_per_day: npt.NDArray[np.float64]):
    array_length = len(moisture_decomposition_factor)
    assert len(added_residue_nitrogen_gN_per_m3_per_day) == array_length
    assert len(added_residue_carbon_gC_per_m3_per_day) == array_length

    # Set up arrays
    litter_cn_ratio_gC_per_gN = np.ndarray(array_length, np.float64)
    humified_fraction_of_decomposed_litter_carbon = np.ndarray(array_length, np.float64)
    microbified_fraction_of_decomposed_litter_carbon = np.ndarray(array_length, np.float64)
    litter_decomposition_carbon_gC_per_m3_per_day = np.ndarray(array_length, np.float64)
    humus_decomposition_carbon_gC_per_m3_per_day = np.ndarray(array_length, np.float64)
    microbial_death_carbon_gC_per_m3_per_day = np.ndarray(array_length, np.float64)
    litter_carbon_gC_per_m3 = np.ndarray(array_length, np.float64)
    litter_nitrogen_gN_per_m3 = np.ndarray(array_length, np.float64)
    microbial_carbon_gC_per_m3 = np.ndarray(array_length, np.float64)
    humus_carbon_gC_per_m3 = np.ndarray(array_length, np.float64)

    # Assign initial values
    litter_carbon_gC_per_m3[0] = initial_litter_carbon_gC_per_m3
    litter_nitrogen_gN_per_m3[0] = initial_litter_nitrogen_gN_per_m3
    microbial_carbon_gC_per_m3[0] = initial_microbial_carbon_gC_per_m3
    humus_carbon_gC_per_m3[0] = initial_humus_carbon_gC_per_m3

    # Simulate
    for i in range(array_length):
        # intermediaries
        litter_cn_ratio_gC_per_gN[i] = litter_carbon_gC_per_m3[i] / litter_nitrogen_gN_per_m3[i]
        humified_fraction_of_decomposed_litter_carbon[i] = min(MAX_HUMIFIED_FRACTION_OF_DECOMPOSED_LITTER, HUMUS_CN_RATIO / litter_cn_ratio_gC_per_gN[i])
        microbified_fraction_of_decomposed_litter_carbon[i] = NON_RESPIRATED_FRACTION_OF_DECOMPOSED_CARBON - humified_fraction_of_decomposed_litter_carbon[i]
        litter_decomposition_carbon_gC_per_m3_per_day[i] = PROPORTION_LITTER_DECOMPOSING_PER_BIOMASS_PER_DAY * moisture_decomposition_factor[i] * microbial_carbon_gC_per_m3[i] * litter_carbon_gC_per_m3[i]
        humus_decomposition_carbon_gC_per_m3_per_day[i] = PROPORTION_HUMUS_DECOMPOSING_PER_BIOMASS_PER_DAY * moisture_decomposition_factor[i] * microbial_carbon_gC_per_m3[i] * humus_carbon_gC_per_m3[i]
        microbial_death_carbon_gC_per_m3_per_day[i] = PROPORTION_BIOMASS_DYING_PER_DAY * microbial_carbon_gC_per_m3[i]

        # states
        if i + 1 < array_length:
            litter_carbon_gC_per_m3[i + 1] = litter_carbon_gC_per_m3[i] + added_residue_carbon_gC_per_m3_per_day[i] + microbial_death_carbon_gC_per_m3_per_day[i] - litter_decomposition_carbon_gC_per_m3_per_day[i]
            litter_nitrogen_gN_per_m3[i + 1] = litter_nitrogen_gN_per_m3[i] + added_residue_nitrogen_gN_per_m3_per_day[i] + (microbial_death_carbon_gC_per_m3_per_day[i] / BIOMASS_CN_RATIO) - (litter_decomposition_carbon_gC_per_m3_per_day[i] / litter_cn_ratio_gC_per_gN[i])
            microbial_carbon_gC_per_m3[i + 1] = microbial_carbon_gC_per_m3[i] + microbified_fraction_of_decomposed_litter_carbon[i] * litter_decomposition_carbon_gC_per_m3_per_day[i] + NON_RESPIRATED_FRACTION_OF_DECOMPOSED_CARBON * humus_decomposition_carbon_gC_per_m3_per_day[i] - microbial_death_carbon_gC_per_m3_per_day[i]
            humus_carbon_gC_per_m3[i + 1] = humus_carbon_gC_per_m3[i] + humified_fraction_of_decomposed_litter_carbon[i] * litter_decomposition_carbon_gC_per_m3_per_day[i] - humus_decomposition_carbon_gC_per_m3_per_day[i]

        # External flows
    excess_bio_nitrogen_per_litter_carbon_decomposed_gN_per_gC = 1 / litter_cn_ratio_gC_per_gN - humified_fraction_of_decomposed_litter_carbon / HUMUS_CN_RATIO - microbified_fraction_of_decomposed_litter_carbon / BIOMASS_CN_RATIO
    net_flux_to_mineral_nitrogen_gN_per_m3_per_day = EXCESS_BIO_NITROGEN_PER_HUMUS_CARBON_DECOMPOSED * humus_decomposition_carbon_gC_per_m3_per_day + excess_bio_nitrogen_per_litter_carbon_decomposed_gN_per_gC * litter_decomposition_carbon_gC_per_m3_per_day

    if any(humified_fraction_of_decomposed_litter_carbon < 0) or any(microbified_fraction_of_decomposed_litter_carbon < 0) or any(litter_decomposition_carbon_gC_per_m3_per_day < 0) or any(humus_decomposition_carbon_gC_per_m3_per_day < 0) or any(microbial_carbon_gC_per_m3 < 0) or any(microbial_carbon_gC_per_m3 > 500) or any(litter_carbon_gC_per_m3 < 0) or any(litter_carbon_gC_per_m3 > 10000) or any(litter_nitrogen_gN_per_m3 < 0) or any(litter_nitrogen_gN_per_m3 > 500) or any(microbial_carbon_gC_per_m3 < 0) or any(microbial_carbon_gC_per_m3 > 500) or any(humus_carbon_gC_per_m3 < 0) or any(humus_carbon_gC_per_m3 > 100000):
        print("Sanity check failed in soil organic model!")

    return litter_carbon_gC_per_m3, litter_nitrogen_gN_per_m3, microbial_carbon_gC_per_m3, humus_carbon_gC_per_m3, net_flux_to_mineral_nitrogen_gN_per_m3_per_day

def inorganic_nitrogen_model(initial_ammonium_gN_per_m3: float, initial_nitrate_gN_per_m3: float, initial_day: float, leakage_mm_per_day: npt.NDArray[np.float64], transpiration_mm_per_day: npt.NDArray[np.float64], soil_moisture: npt.NDArray[np.float64], net_flux_to_mineral_nitrogen_gN_per_m3_per_day: npt.NDArray[np.float64], microbial_carbon_gC_per_m3: npt.NDArray[np.float64], day_to_plant_N_demand_gN_per_m3_per_day: np.vectorize, input_dt_days: float = 1.0, output_dt_days: float = INORGANIC_N_MODEL_DT_DAYS) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64],npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool], npt.NDArray[np.float64]]:
    array_length = len(leakage_mm_per_day)
    assert len(transpiration_mm_per_day) == array_length
    assert len(soil_moisture) == array_length
    assert len(net_flux_to_mineral_nitrogen_gN_per_m3_per_day) == array_length
    assert len(microbial_carbon_gC_per_m3) == array_length

    # scaled arrays, for computational efficiency
    moisture_nitrification_factor = f_n(soil_moisture)
    moisture_decomposition_factor = f_d(soil_moisture)
    volume_water_per_unit_area_mm = soil_moisture * SOIL_DEPTH_MM * SOIL_POROSITY
    volume_water_per_unit_area_mm[volume_water_per_unit_area_mm == 0] = 0.01 # to avoid divide by zero
    nitrification_rate_constant = moisture_nitrification_factor * PROPORTION_AMMONIUM_NITRIFIED_PER_BIOMASS_PER_DAY * microbial_carbon_gC_per_m3
    ammonium_passive_uptake_rate_constant = DISSOLVED_FRACTION_AMMONIUM * transpiration_mm_per_day / volume_water_per_unit_area_mm
    ammonium_max_active_uptake_rate_constant = DISSOLVED_FRACTION_AMMONIUM * DIFFUSION_COEFFICIENT_MM_PER_DAY / volume_water_per_unit_area_mm * (soil_moisture ** DIFFUSION_MOISTURE_DEPENDENCE_EXPONENT)
    ammonium_leaching_rate_constant = DISSOLVED_FRACTION_AMMONIUM * leakage_mm_per_day / volume_water_per_unit_area_mm
    nitrate_passive_uptake_rate_constant = DISSOLVED_FRACTION_NITRATE * transpiration_mm_per_day / volume_water_per_unit_area_mm
    nitrate_max_active_uptake_rate_constant = DISSOLVED_FRACTION_NITRATE * DIFFUSION_COEFFICIENT_MM_PER_DAY / volume_water_per_unit_area_mm * (soil_moisture ** DIFFUSION_MOISTURE_DEPENDENCE_EXPONENT)
    nitrate_leaching_rate_constant = DISSOLVED_FRACTION_NITRATE * leakage_mm_per_day / volume_water_per_unit_area_mm
    # constants
    time_stretch = input_dt_days / output_dt_days
    output_array_length = int(array_length * time_stretch)
    # N demand
    absolute_day = np.linspace(initial_day, initial_day + array_length * input_dt_days, output_array_length)
    plant_N_demand_gN_per_m3_per_day = day_to_plant_N_demand_gN_per_m3_per_day(absolute_day)
    plant_ammonium_demand_gN_per_m3_per_day = np.zeros(plant_N_demand_gN_per_m3_per_day.shape) #TODO experiment with this, and the balance between ammonium vs nitrate demand
    plant_nitrate_demand_gN_per_m3_per_day = plant_N_demand_gN_per_m3_per_day

    # Set up arrays
    available_nitrogen_for_immobilization_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    maximum_immobilization_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    nitrification_assumptions_violation = np.full(output_array_length, False, np.bool)
    nitrification_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    mineralization_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    total_immobilization_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    immobilization_from_ammonium_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    plant_passive_uptake_of_ammonium_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    maximum_active_ammonium_uptake_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    plant_active_uptake_of_ammonium_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    ammonium_leaching_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    immobilization_from_nitrate_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    plant_passive_uptake_of_nitrate_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    maximum_active_nitrate_uptake_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    plant_active_uptake_of_nitrate_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    nitrate_leaching_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)

    ammonium_gN_per_m3 = np.ndarray(output_array_length, np.float64)
    nitrate_gN_per_m3 = np.ndarray(output_array_length, np.float64)

    # Assign initial values
    ammonium_gN_per_m3[0] = initial_ammonium_gN_per_m3
    nitrate_gN_per_m3[0] = initial_nitrate_gN_per_m3

    # Simulate
    for i in range(output_array_length):
        input_index = int(i / time_stretch)
        available_nitrogen_for_immobilization_gN_per_m3_per_day[i] = MAX_PROPORTION_AMMONIUM_IMMOBILIZED_PER_DAY * ammonium_gN_per_m3[i] + MAX_PROPORTION_NITRATE_IMMOBILIZED_PER_DAY * nitrate_gN_per_m3[i]
        maximum_immobilization_gN_per_m3_per_day[i] = moisture_decomposition_factor[input_index] * available_nitrogen_for_immobilization_gN_per_m3_per_day[i]
        if net_flux_to_mineral_nitrogen_gN_per_m3_per_day[input_index] > 0:
            mineralization_gN_per_m3_per_day[i] = net_flux_to_mineral_nitrogen_gN_per_m3_per_day[input_index]
            total_immobilization_gN_per_m3_per_day[i] = 0
        else:
            mineralization_gN_per_m3_per_day[i] = 0
            total_immobilization_gN_per_m3_per_day[i] = -net_flux_to_mineral_nitrogen_gN_per_m3_per_day[input_index]
        if total_immobilization_gN_per_m3_per_day[i] > maximum_immobilization_gN_per_m3_per_day[i]:
            print("Insufficient nitrogen to meet immobilization demand, assumptions violated!")
            nitrification_assumptions_violation[i] = True

        nitrification_gN_per_m3_per_day[i] = nitrification_rate_constant[input_index] * ammonium_gN_per_m3[i]

        # Duplicated between ammonium and nitrate
        immobilization_from_ammonium_gN_per_m3_per_day[i] = 0 if available_nitrogen_for_immobilization_gN_per_m3_per_day[i] == 0  else MAX_PROPORTION_AMMONIUM_IMMOBILIZED_PER_DAY * ammonium_gN_per_m3[i] / available_nitrogen_for_immobilization_gN_per_m3_per_day[i] * total_immobilization_gN_per_m3_per_day[i]
        plant_passive_uptake_of_ammonium_gN_per_m3_per_day[i] = ammonium_passive_uptake_rate_constant[input_index] * ammonium_gN_per_m3[i]
        maximum_active_ammonium_uptake_gN_per_m3_per_day[i] = ammonium_max_active_uptake_rate_constant[input_index] * ammonium_gN_per_m3[i]
        plant_active_uptake_of_ammonium_gN_per_m3_per_day[i] = max(0.0, min(maximum_active_ammonium_uptake_gN_per_m3_per_day[i], plant_ammonium_demand_gN_per_m3_per_day[i] - plant_passive_uptake_of_ammonium_gN_per_m3_per_day[i]))
        ammonium_leaching_gN_per_m3_per_day[i] = ammonium_leaching_rate_constant[input_index] * ammonium_gN_per_m3[i]

        immobilization_from_nitrate_gN_per_m3_per_day[i] = 0 if available_nitrogen_for_immobilization_gN_per_m3_per_day[i] == 0  else MAX_PROPORTION_NITRATE_IMMOBILIZED_PER_DAY * nitrate_gN_per_m3[i] / available_nitrogen_for_immobilization_gN_per_m3_per_day[i] * total_immobilization_gN_per_m3_per_day[i]
        plant_passive_uptake_of_nitrate_gN_per_m3_per_day[i] = nitrate_passive_uptake_rate_constant[input_index] * nitrate_gN_per_m3[i]
        maximum_active_nitrate_uptake_gN_per_m3_per_day[i] = nitrate_max_active_uptake_rate_constant[input_index] * nitrate_gN_per_m3[i]
        plant_active_uptake_of_nitrate_gN_per_m3_per_day[i] = max(0.0, min(maximum_active_nitrate_uptake_gN_per_m3_per_day[i], plant_nitrate_demand_gN_per_m3_per_day[i] - plant_passive_uptake_of_nitrate_gN_per_m3_per_day[i]))
        nitrate_leaching_gN_per_m3_per_day[i] = nitrate_leaching_rate_constant[input_index] * nitrate_gN_per_m3[i]

        # States
        if (i + 1) < output_array_length:
            ammonium_scaling_fix = 1.0
            ammonium_gN_per_m3[i + 1] = ammonium_gN_per_m3[i] + (mineralization_gN_per_m3_per_day[i] - immobilization_from_ammonium_gN_per_m3_per_day[i] - nitrification_gN_per_m3_per_day[i] - ammonium_leaching_gN_per_m3_per_day[i] - plant_passive_uptake_of_ammonium_gN_per_m3_per_day[i] - plant_active_uptake_of_ammonium_gN_per_m3_per_day[i]) * output_dt_days
            if ammonium_gN_per_m3[i + 1] < 0:
                ammonium_scaling_fix = ammonium_gN_per_m3[i] / (ammonium_gN_per_m3[i] - ammonium_gN_per_m3[i + 1])
                print("Ammonium underflow error! Applying scaling fix of", ammonium_scaling_fix)
                ammonium_gN_per_m3[i + 1] = 0.0
            nitrate_gN_per_m3[i + 1] = nitrate_gN_per_m3[i] + (nitrification_gN_per_m3_per_day[i] * ammonium_scaling_fix - immobilization_from_nitrate_gN_per_m3_per_day[i] - nitrate_leaching_gN_per_m3_per_day[i] - plant_passive_uptake_of_nitrate_gN_per_m3_per_day[i] - plant_active_uptake_of_nitrate_gN_per_m3_per_day[i]) * output_dt_days

    # if any(available_nitrogen_for_immobilization_gN_per_m3_per_day < 0) or any(maximum_immobilization_gN_per_m3_per_day < 0) or any(nitrification_assumptions_violation) or any(nitrification_gN_per_m3_per_day < 0) or any(mineralization_gN_per_m3_per_day < 0) or any(total_immobilization_gN_per_m3_per_day < 0) or any(plant_passive_uptake_of_ammonium_gN_per_m3_per_day < 0) or any(plant_active_uptake_of_ammonium_gN_per_m3_per_day < 0) or any(plant_passive_uptake_of_nitrate_gN_per_m3_per_day < 0) or any(plant_active_uptake_of_nitrate_gN_per_m3_per_day < 0) or any(ammonium_leaching_gN_per_m3_per_day < 0) or any(nitrate_leaching_gN_per_m3_per_day < 0) or any(ammonium_gN_per_m3 < 0) or any(ammonium_gN_per_m3 > 500) or any(nitrate_gN_per_m3 < 0) or any(nitrate_gN_per_m3 > 500):
    #     print("Inorganic nitrogen sanity check failed!")

    return ammonium_gN_per_m3, nitrate_gN_per_m3, ammonium_leaching_gN_per_m3_per_day, nitrate_leaching_gN_per_m3_per_day, plant_passive_uptake_of_ammonium_gN_per_m3_per_day, plant_passive_uptake_of_nitrate_gN_per_m3_per_day, plant_active_uptake_of_ammonium_gN_per_m3_per_day, plant_active_uptake_of_nitrate_gN_per_m3_per_day, nitrification_assumptions_violation, nitrification_gN_per_m3_per_day