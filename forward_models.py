import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import dataclass

rng = np.random.default_rng(12345)

@dataclass
class PorporatoParameters:
    SOIL_POROSITY: float = 0.35 # Porporato had 0.4
    SOIL_DEPTH_MM: float = 300 # Porporato had 800
    SOIL_VOID_SPACE_MM: float = SOIL_POROSITY * SOIL_DEPTH_MM
    HYGROSCOPIC_POINT: float = 0.04 # Porporato had 0.02
    WILTING_POINT: float = 0.05 # Porporato had 0.065
    MAX_MOISTURE_WITH_CLOSED_STOMATA: float = 0.16 # Porporato had 0.17
    FIELD_CAPACITY: float = 0.35 # Porporato had 0.3
    MAXIMUM_EVAPORATION_RATE_MM_PER_DAY: float = 0.1 # Porporato had 1.0
    MAXIMUM_TRANSPIRATION_RATE_MM_PER_DAY: float = 3.6 # Porporato had 3.5
    SATURATED_HYDRAULIC_CONDUCTIVITY_MM_PER_DAY: float = 1000 # Porporato had 1100
    PORE_SIZE_DISTRIBUTION_INDEX: float = 4.05 # 4.05 for porporato conditions
    BETA: float = 4.0 * PORE_SIZE_DISTRIBUTION_INDEX + 4.0
    ADDED_RESIDUE_CARBON_G_PER_M2_PER_DAY: float = 1.5
    PROPORTION_BIOMASS_DYING_PER_DAY: float = 0.0085
    PROPORTION_LITTER_DECOMPOSING_PER_BIOMASS_PER_DAY: float = 0.000065
    PROPORTION_HUMUS_DECOMPOSING_PER_BIOMASS_PER_DAY: float = 0.0000025
    MAX_PROPORTION_AMMONIUM_IMMOBILIZED_PER_DAY: float = 1.0
    MAX_PROPORTION_NITRATE_IMMOBILIZED_PER_DAY: float = 1.0
    PROPORTION_AMMONIUM_NITRIFIED_PER_BIOMASS_PER_DAY: float = 0.006 # porporato had 0.6
    DISSOLVED_FRACTION_AMMONIUM: float = 0.05
    DISSOLVED_FRACTION_NITRATE: float = 1.0 # https://link.springer.com/chapter/10.1007/978-94-007-0394-0_23/tables/4 has 0.1 but that can't be right
    RESPIRATED_FRACTION_OF_DECOMPOSED_CARBON: float = 0.6
    NON_RESPIRATED_FRACTION_OF_DECOMPOSED_CARBON: float = 1 - RESPIRATED_FRACTION_OF_DECOMPOSED_CARBON
    MAX_HUMIFIED_FRACTION_OF_DECOMPOSED_LITTER: float = 0.25
    ADDED_RESIDUE_CN_RATIO: float = 58
    BIOMASS_CN_RATIO: float = 11.5
    HUMUS_CN_RATIO: float = 22 # porporato had 22
    EXCESS_BIO_NITROGEN_PER_HUMUS_CARBON_DECOMPOSED: float = 1 / HUMUS_CN_RATIO - NON_RESPIRATED_FRACTION_OF_DECOMPOSED_CARBON / BIOMASS_CN_RATIO
    # PLANT_AMMONIUM_DEMAND_GRAMS_N_PER_M3_PER_DAY = 0.2
    # PLANT_NITRATE_DEMAND_GRAMS_N_PER_M3_PER_DAY = 0.5
    PLANT_BIOMASS_G_DRY_MASS_PER_M2: float = 600 # roughly 17L stage
    DIFFUSION_COEFFICIENT_MM_PER_DAY: float = 100 # possibly should be 100 ? Porporato has 0.1 meters/day
    DIFFUSION_MOISTURE_DEPENDENCE_EXPONENT: float = 1.5 # Porporato had 3
    RAIN_EXISTENCE_RATE: float = 0.26 # porporato had 0.23, ag paper had 0.3
    RAIN_MEAN_DEPTH_MM: float = 13.5 # porporato had 11, ag paper had 15.5

    def f_d(self, nd_array):
        return np.vectorize(lambda s : s / self.FIELD_CAPACITY if s <= self.FIELD_CAPACITY else self.FIELD_CAPACITY / s)(nd_array)

    def f_n(self, nd_array):
        return np.vectorize(lambda s: s / self.FIELD_CAPACITY if s <= self.FIELD_CAPACITY else (1 - s)/(1- self.FIELD_CAPACITY))(nd_array)

INORGANIC_N_MODEL_DT_DAYS = 1.0 / 24.0 * 2.0 / 6.0 # dt of 20 minutes

def generate_rain(num_days: int, porporato_parameters = PorporatoParameters()) -> npt.NDArray[np.float64]:
    """
    Generate num_days days worth of rainfall. Rainfalls represent total accumulation over the course of a day
    :param num_days: How many days of rainfall to generate
    :return: A 1D numpy array of length num_days, with each entry representing a daily rainfall in millimeters
    """
    rain_events = rng.random(num_days) < porporato_parameters.RAIN_EXISTENCE_RATE
    rain_quantity_mm = rng.exponential(porporato_parameters.RAIN_MEAN_DEPTH_MM, num_days) * rain_events
    return rain_quantity_mm

def hydrology_model(initial_soil_moisture: float,
                    rain_quantity_mm: npt.NDArray[np.float64],
                    porporato_parameters = PorporatoParameters()) \
        -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64],
        npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """
    Run the hydrology model from Porporato 2003, with a daily timestep.
    :param initial_soil_moisture: The starting soil moisture, as a percent in [0,1]
    :param rain_quantity_mm: A 1D array of daily rainfalls. The hydrology model will run from the beginning of the first
        day given to the end of the last.
    :return: A tuple of numpy arrays, all the same shape as rain_quality_mm, representing the daily values of the following:
        end-of-day soil moisture (as a percent in [0,1])
        infiltration rate (in mm/day)
        evaporation rate (in mm/day)
        transpiration rate (in mm/day)
        leakage rate (in mm/day)
        moisture effect on decomposition factor (f_d(s) from Porporato, unitless in [0,1])
        moisture effect on nitrification factor (f_n(s) from Porporato, unitless in [0,1])
    """
    num_days = len(rain_quantity_mm)

    # Define constants for computational efficiency
    LEAKAGE_COEFF = porporato_parameters.SATURATED_HYDRAULIC_CONDUCTIVITY_MM_PER_DAY / (np.exp(porporato_parameters.BETA * (1 - porporato_parameters.FIELD_CAPACITY)) - 1)
    TRANSPIRATION_SLOPE = porporato_parameters.MAXIMUM_TRANSPIRATION_RATE_MM_PER_DAY / (porporato_parameters.MAX_MOISTURE_WITH_CLOSED_STOMATA - porporato_parameters.WILTING_POINT)

    # States and intermediate variables
    end_of_day_soil_moistures = np.zeros(num_days)
    infiltration_rate_mm_per_day = np.zeros(num_days)
    evaporation_rate_mm_per_day = np.zeros(num_days)
    transpiration_rate_mm_per_day = np.zeros(num_days)
    leakage_rate_mm_per_day = np.zeros(num_days)
    moisture_effect_on_decomposition_factor = np.zeros(num_days)
    moisture_effect_on_nitrification_factor = np.zeros(num_days)

    start_of_day_soil_moisture = initial_soil_moisture

    for i in range(num_days):
        infiltration_rate_mm_per_day[i] = min(rain_quantity_mm[i], porporato_parameters.SOIL_VOID_SPACE_MM * (1 - start_of_day_soil_moisture))

        # evaporation
        if start_of_day_soil_moisture < porporato_parameters.HYGROSCOPIC_POINT:
            evaporation_rate_mm_per_day[i] = 0
        elif start_of_day_soil_moisture <= porporato_parameters.WILTING_POINT:
            evaporation_rate_mm_per_day[i] = porporato_parameters.MAXIMUM_EVAPORATION_RATE_MM_PER_DAY * (start_of_day_soil_moisture - porporato_parameters.HYGROSCOPIC_POINT)
        else:
            evaporation_rate_mm_per_day[i] = porporato_parameters.MAXIMUM_EVAPORATION_RATE_MM_PER_DAY

        # Transpiration
        if start_of_day_soil_moisture <= porporato_parameters.WILTING_POINT:
            transpiration_rate_mm_per_day[i] = 0
        elif start_of_day_soil_moisture <= porporato_parameters.MAX_MOISTURE_WITH_CLOSED_STOMATA:
            transpiration_rate_mm_per_day[i] = TRANSPIRATION_SLOPE * (start_of_day_soil_moisture - porporato_parameters.WILTING_POINT)
        else:
            transpiration_rate_mm_per_day[i] = porporato_parameters.MAXIMUM_TRANSPIRATION_RATE_MM_PER_DAY

        # leakage/percolation
        leakage_rate_mm_per_day[i] = LEAKAGE_COEFF * (np.exp(porporato_parameters.BETA * (start_of_day_soil_moisture - porporato_parameters.FIELD_CAPACITY)) - 1)
        if start_of_day_soil_moisture - (leakage_rate_mm_per_day[i] / porporato_parameters.SOIL_VOID_SPACE_MM) < porporato_parameters.FIELD_CAPACITY: # make sure we don't drain more than is possible in very wet fields
            leakage_rate_mm_per_day[i] = max(0.0, (start_of_day_soil_moisture - porporato_parameters.FIELD_CAPACITY) * porporato_parameters.SOIL_VOID_SPACE_MM)


        end_of_day_soil_moistures[i] = start_of_day_soil_moisture + (infiltration_rate_mm_per_day[i] - evaporation_rate_mm_per_day[i] - transpiration_rate_mm_per_day[i] - leakage_rate_mm_per_day[i]) / porporato_parameters.SOIL_VOID_SPACE_MM
        daily_average_soil_moisture = (start_of_day_soil_moisture + end_of_day_soil_moistures[i]) / 2.
        moisture_effect_on_decomposition_factor[i] = porporato_parameters.f_d(daily_average_soil_moisture)
        moisture_effect_on_nitrification_factor[i] = porporato_parameters.f_n(daily_average_soil_moisture)

        start_of_day_soil_moisture = end_of_day_soil_moistures[i]

    if any(infiltration_rate_mm_per_day < 0) or any(evaporation_rate_mm_per_day < 0) or any(transpiration_rate_mm_per_day < 0) or any(leakage_rate_mm_per_day < 0) or any(end_of_day_soil_moistures < 0) or any(end_of_day_soil_moistures > 1) or any(moisture_effect_on_decomposition_factor < 0) or any(moisture_effect_on_decomposition_factor > 1) or any(moisture_effect_on_nitrification_factor < 0) or any(moisture_effect_on_nitrification_factor > 1):
        print("Sanity check failed in hydrology model!")

    return end_of_day_soil_moistures, infiltration_rate_mm_per_day, evaporation_rate_mm_per_day, transpiration_rate_mm_per_day, leakage_rate_mm_per_day, moisture_effect_on_decomposition_factor, moisture_effect_on_nitrification_factor

def soil_organic_model(initial_litter_carbon_gC_per_m3: float,
                       initial_litter_nitrogen_gN_per_m3: float,
                       initial_microbial_carbon_gC_per_m3: float,
                       initial_humus_carbon_gC_per_m3: float,
                       moisture_decomposition_factor: npt.NDArray[np.float64],
                       added_residue_nitrogen_gN_per_m3_per_day: npt.NDArray[np.float64],
                       added_residue_carbon_gC_per_m3_per_day: npt.NDArray[np.float64],
                       porporato_parameters = PorporatoParameters()) \
        -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64],
        npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """
    Run the part of the Porporato (2003) soil model corresponding to the organic material soil states, under the
    assumption that decomposition is not rate-limited by N availability, i.e. phi = 1. Runs on a daily timestep, for a
    number of days equal to the length of moisture_decomposition_factor, which must be the same as the lengths of
    added_residue_nitrogen_gN_per_m3_per_day and added_residue_carbon_gC_per_m3_per_day

    :param initial_litter_carbon_gC_per_m3: The initial density of litter C in the soil, in gC/m^3
    :param initial_litter_nitrogen_gN_per_m3: The initial density of litter N in the soil, in gN/m^3
    :param initial_microbial_carbon_gC_per_m3: The initial density of microbial C in the soil, in gC/m^3
    :param initial_humus_carbon_gC_per_m3: The initial density of humus C in the soil, in gC/m^3
    :param moisture_decomposition_factor: A 1D np array containing the daily values of f_d(s) from Porporato, which are
        unitless in [0,1]
    :param added_residue_nitrogen_gN_per_m3_per_day: A 1D np array containing the daily mass of N added to the litter
        pool from residue fall, in gN/m^3
    :param added_residue_carbon_gC_per_m3_per_day: A 1D np array containing the daily mass of C added to the litter
        pool from residue fall, in gC/m^3
    :return: A tuple of numpy arrays, all the same shape as moisture_decomposition_factor, representing the daily values of the following:
        end-of-day litter carbon, in gC/m^3
        end-of-day litter N, in gN/m^3
        end-of-day microbial C, in gN/m^3
        end-of-day humus C, in gN/m^3
        net flux of organic to mineral nitrogen, in gN/m^3/day. Values may be negative if there's more immobilization
            than mineralization.
    """
    num_days = len(moisture_decomposition_factor)
    assert len(added_residue_nitrogen_gN_per_m3_per_day) == num_days
    assert len(added_residue_carbon_gC_per_m3_per_day) == num_days

    # Set up arrays
    litter_decomposition_carbon_gC_per_m3_per_day = np.ndarray(num_days, np.float64)
    humus_decomposition_carbon_gC_per_m3_per_day = np.ndarray(num_days, np.float64)
    microbial_death_carbon_gC_per_m3_per_day = np.ndarray(num_days, np.float64)

    # "sod" here stands for "start of day". These arrays are 1 longer than the others, because we actually care about
    # the end-of-day values, and the end of day n is the start of day n+1
    sod_litter_cn_ratio_gC_per_gN = np.ndarray(num_days + 1, np.float64)
    sod_humified_fraction_of_decomposed_litter_carbon = np.ndarray(num_days + 1, np.float64)
    sod_microbified_fraction_of_decomposed_litter_carbon = np.ndarray(num_days + 1, np.float64)
    sod_litter_carbon_gC_per_m3 = np.ndarray(num_days + 1, np.float64)
    sod_litter_nitrogen_gN_per_m3 = np.ndarray(num_days + 1, np.float64)
    sod_microbial_carbon_gC_per_m3 = np.ndarray(num_days + 1, np.float64)
    sod_humus_carbon_gC_per_m3 = np.ndarray(num_days + 1, np.float64)

    # Assign initial values
    sod_litter_carbon_gC_per_m3[0] = initial_litter_carbon_gC_per_m3
    sod_litter_nitrogen_gN_per_m3[0] = initial_litter_nitrogen_gN_per_m3
    sod_microbial_carbon_gC_per_m3[0] = initial_microbial_carbon_gC_per_m3
    sod_humus_carbon_gC_per_m3[0] = initial_humus_carbon_gC_per_m3

    # Simulate
    for i in range(num_days + 1):
        # intermediaries
        sod_litter_cn_ratio_gC_per_gN[i] = sod_litter_carbon_gC_per_m3[i] / sod_litter_nitrogen_gN_per_m3[i]
        sod_humified_fraction_of_decomposed_litter_carbon[i] = min(porporato_parameters.MAX_HUMIFIED_FRACTION_OF_DECOMPOSED_LITTER, porporato_parameters.HUMUS_CN_RATIO / sod_litter_cn_ratio_gC_per_gN[i])
        sod_microbified_fraction_of_decomposed_litter_carbon[i] = porporato_parameters.NON_RESPIRATED_FRACTION_OF_DECOMPOSED_CARBON - sod_humified_fraction_of_decomposed_litter_carbon[i]

        if i < num_days:
            litter_decomposition_carbon_gC_per_m3_per_day[i] = porporato_parameters.PROPORTION_LITTER_DECOMPOSING_PER_BIOMASS_PER_DAY * moisture_decomposition_factor[i] * sod_microbial_carbon_gC_per_m3[i] * sod_litter_carbon_gC_per_m3[i]
            humus_decomposition_carbon_gC_per_m3_per_day[i] = porporato_parameters.PROPORTION_HUMUS_DECOMPOSING_PER_BIOMASS_PER_DAY * moisture_decomposition_factor[i] * sod_microbial_carbon_gC_per_m3[i] * sod_humus_carbon_gC_per_m3[i]
            microbial_death_carbon_gC_per_m3_per_day[i] = porporato_parameters.PROPORTION_BIOMASS_DYING_PER_DAY * sod_microbial_carbon_gC_per_m3[i]

            sod_litter_carbon_gC_per_m3[i + 1] = sod_litter_carbon_gC_per_m3[i] + added_residue_carbon_gC_per_m3_per_day[i] + microbial_death_carbon_gC_per_m3_per_day[i] - litter_decomposition_carbon_gC_per_m3_per_day[i]
            sod_litter_nitrogen_gN_per_m3[i + 1] = sod_litter_nitrogen_gN_per_m3[i] + added_residue_nitrogen_gN_per_m3_per_day[i] + (microbial_death_carbon_gC_per_m3_per_day[i] / porporato_parameters.BIOMASS_CN_RATIO) - (litter_decomposition_carbon_gC_per_m3_per_day[i] / sod_litter_cn_ratio_gC_per_gN[i])
            sod_microbial_carbon_gC_per_m3[i + 1] = sod_microbial_carbon_gC_per_m3[i] + sod_microbified_fraction_of_decomposed_litter_carbon[i] * litter_decomposition_carbon_gC_per_m3_per_day[i] + porporato_parameters.NON_RESPIRATED_FRACTION_OF_DECOMPOSED_CARBON * humus_decomposition_carbon_gC_per_m3_per_day[i] - microbial_death_carbon_gC_per_m3_per_day[i]
            sod_humus_carbon_gC_per_m3[i + 1] = sod_humus_carbon_gC_per_m3[i] + sod_humified_fraction_of_decomposed_litter_carbon[i] * litter_decomposition_carbon_gC_per_m3_per_day[i] - humus_decomposition_carbon_gC_per_m3_per_day[i]

        # External flows
    excess_bio_nitrogen_per_litter_carbon_decomposed_gN_per_gC = 1 / sod_litter_cn_ratio_gC_per_gN - sod_humified_fraction_of_decomposed_litter_carbon / porporato_parameters.HUMUS_CN_RATIO - sod_microbified_fraction_of_decomposed_litter_carbon / porporato_parameters.BIOMASS_CN_RATIO
    net_flux_to_mineral_nitrogen_gN_per_m3_per_day = porporato_parameters.EXCESS_BIO_NITROGEN_PER_HUMUS_CARBON_DECOMPOSED * humus_decomposition_carbon_gC_per_m3_per_day + excess_bio_nitrogen_per_litter_carbon_decomposed_gN_per_gC[:-1] * litter_decomposition_carbon_gC_per_m3_per_day

    if (any(sod_humified_fraction_of_decomposed_litter_carbon < 0)
            or any(sod_microbified_fraction_of_decomposed_litter_carbon < 0)
            or any(litter_decomposition_carbon_gC_per_m3_per_day < 0)
            or any(humus_decomposition_carbon_gC_per_m3_per_day < 0)
            or any(sod_microbial_carbon_gC_per_m3 < 0)
            or any(sod_microbial_carbon_gC_per_m3 > 500)
            or any(sod_litter_carbon_gC_per_m3 < 0)
            or any(sod_litter_carbon_gC_per_m3 > 10000)
            or any(sod_litter_nitrogen_gN_per_m3 < 0)
            or any(sod_litter_nitrogen_gN_per_m3 > 500)
            or any(sod_microbial_carbon_gC_per_m3 < 0)
            or any(sod_microbial_carbon_gC_per_m3 > 500)
            or any(sod_humus_carbon_gC_per_m3 < 0)
            or any(sod_humus_carbon_gC_per_m3 > 100000)):
        print("Sanity check failed in soil organic model!")

    # We return the start-of-day values for days 1 to n+1, which is equivalent to end-of-day values for days 0 to n.
    return sod_litter_carbon_gC_per_m3[1:], sod_litter_nitrogen_gN_per_m3[1:], sod_microbial_carbon_gC_per_m3[1:], sod_humus_carbon_gC_per_m3[1:], net_flux_to_mineral_nitrogen_gN_per_m3_per_day

def inorganic_nitrogen_model(initial_ammonium_gN_per_m3: float,
                             initial_nitrate_gN_per_m3: float,
                             initial_day: float,
                             leakage_mm_per_day: npt.NDArray[np.float64],
                             transpiration_mm_per_day: npt.NDArray[np.float64],
                             start_of_period_soil_moistures: npt.NDArray[np.float64],
                             net_flux_to_mineral_nitrogen_gN_per_m3_per_day: npt.NDArray[np.float64],
                             start_of_period_microbial_carbon_gC_per_m3: npt.NDArray[np.float64],
                             day_to_plant_N_demand_gN_per_m3_per_day: np.vectorize,
                             input_dt_days: float = 1.0,
                             output_dt_days: float = INORGANIC_N_MODEL_DT_DAYS, porporato_parameters = PorporatoParameters()) \
        -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64],npt.NDArray[np.float64],
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64],
        npt.NDArray[np.bool], npt.NDArray[np.float64]]:
    """
    Run the inorganic N and plant uptake components of the Porporato (2003) model.
    :param initial_ammonium_gN_per_m3: The initial value for the soil ammonium pool, in gN/m^3
    :param initial_nitrate_gN_per_m3: The initial value for the soil nitrate pool, in gN/m^3
    :param initial_day: The starting day for the model (used for time-varying plant N demand)
    :param leakage_mm_per_day: a 1D np array where each value is the leakage rate of water out of the soil, in mm/day
    :param transpiration_mm_per_day: An array the same shape as leakage_mm_per_day, representing the transpiration
        rate of water out of the soil, in mm/day
    :param start_of_period_soil_moistures: a 1D np array with 1 more element than leakage_mm_per_day, representing the
        soil moisture at the start of each model period, as a percent in [0,1]. It is larger so the first element can be
        the initial soil moisture, and the last element the final moisture.
    :param net_flux_to_mineral_nitrogen_gN_per_m3_per_day: An array the same shape as leakage_mm_per_day, representing
        the net flux of mineral to organic N, in gN/m^3/day. Values may be negative.
    :param start_of_period_microbial_carbon_gC_per_m3: An array the same shape as start_of_period_soil_moisture,
        representing the soil microbe density in gC/m^3.
    :param day_to_plant_N_demand_gN_per_m3_per_day: A vectorized function to calculate the plant N demand, in gN/m^3/day,
        given an absolute day.
    :param input_dt_days: The timestep for the input arrays (leakage_mm_per_day, start_of_period_soil_moisture,
        net_flux_to_mineral_nitrogen_gN_per_m3_per_day, start_of_period_microbial_carbon_gC_per_m3) in days.
    :param output_dt_days: The timestep to run this part of the model at, in days.
    :return: A tuple of 1D arrays, all the same shape, representing the values of the following at each model timestep:
      end-of-period soil ammonium, in gN/m^3
      end-of-period soil nitrate, in gN/m^3
      rate of ammonium leaching, in gN/m^3/day
      rate of nitrate leaching, in gN/m^3/day
      rate of plant passive ammonium uptake, in gN/m^3/day
      rate of plant passive nitrate uptake, in gN/m^3/day
      rate of plant active ammonium uptake, in gN/m^3/day
      rate of plant active nitrate uptake, in gN/m^3/day
      whether the assumption that phi=1 was violated, as a boolean
      rate of nitrification, in gN/m^3/day
    """
    input_array_length = len(leakage_mm_per_day)
    assert len(transpiration_mm_per_day) == input_array_length
    assert len(net_flux_to_mineral_nitrogen_gN_per_m3_per_day) == input_array_length
    assert len(start_of_period_soil_moistures) == input_array_length + 1
    assert len(start_of_period_microbial_carbon_gC_per_m3) == input_array_length + 1
    num_days = round(input_array_length * input_dt_days)

    # timing
    time_stretch = input_dt_days / output_dt_days
    output_array_length = int(input_array_length * time_stretch)
    # sop stands for "start of period"
    sop_absolute_day = np.linspace(initial_day, initial_day + num_days, output_array_length + 1)
    input_sop_absolute_day = np.linspace(initial_day, initial_day + num_days, input_array_length + 1)

    # interpolate input arrays, evaluating at the middle of each output time step
    interpolated_soil_moisture = np.interp(sop_absolute_day[:-1] + output_dt_days / 2.0, input_sop_absolute_day, start_of_period_soil_moistures)
    interpolated_microbial_carbon_gC_per_m3 = np.interp(sop_absolute_day[:-1] + output_dt_days / 2.0, input_sop_absolute_day, start_of_period_microbial_carbon_gC_per_m3)

    # Pre-calculate as much as possible using vectorized operations
    moisture_nitrification_factor = porporato_parameters.f_n(interpolated_soil_moisture)
    moisture_decomposition_factor = porporato_parameters.f_d(interpolated_microbial_carbon_gC_per_m3)
    volume_water_per_unit_area_mm = interpolated_soil_moisture * porporato_parameters.SOIL_DEPTH_MM * porporato_parameters.SOIL_POROSITY
    volume_water_per_unit_area_mm[volume_water_per_unit_area_mm == 0] = 0.0001 # to avoid divide by zero
    nitrification_rate_constant = moisture_nitrification_factor * porporato_parameters.PROPORTION_AMMONIUM_NITRIFIED_PER_BIOMASS_PER_DAY * interpolated_microbial_carbon_gC_per_m3
    ammonium_passive_uptake_rate_constant = porporato_parameters.DISSOLVED_FRACTION_AMMONIUM * np.repeat(transpiration_mm_per_day, time_stretch) / volume_water_per_unit_area_mm
    ammonium_max_active_uptake_rate_constant = porporato_parameters.DISSOLVED_FRACTION_AMMONIUM * porporato_parameters.DIFFUSION_COEFFICIENT_MM_PER_DAY / volume_water_per_unit_area_mm * (interpolated_soil_moisture ** porporato_parameters.DIFFUSION_MOISTURE_DEPENDENCE_EXPONENT)
    ammonium_leaching_rate_constant = porporato_parameters.DISSOLVED_FRACTION_AMMONIUM * np.repeat(leakage_mm_per_day, time_stretch) / volume_water_per_unit_area_mm
    nitrate_passive_uptake_rate_constant = porporato_parameters.DISSOLVED_FRACTION_NITRATE * np.repeat(transpiration_mm_per_day, time_stretch) / volume_water_per_unit_area_mm
    nitrate_max_active_uptake_rate_constant = porporato_parameters.DISSOLVED_FRACTION_NITRATE * porporato_parameters.DIFFUSION_COEFFICIENT_MM_PER_DAY / volume_water_per_unit_area_mm * (interpolated_soil_moisture ** porporato_parameters.DIFFUSION_MOISTURE_DEPENDENCE_EXPONENT)
    nitrate_leaching_rate_constant = porporato_parameters.DISSOLVED_FRACTION_NITRATE * np.repeat(leakage_mm_per_day, time_stretch) / volume_water_per_unit_area_mm

    # N demand
    plant_N_demand_gN_per_m3_per_day = day_to_plant_N_demand_gN_per_m3_per_day(sop_absolute_day[:-1])
    plant_ammonium_demand_gN_per_m3_per_day = np.zeros(plant_N_demand_gN_per_m3_per_day.shape) #TODO experiment with this, and the balance between ammonium vs nitrate demand
    plant_nitrate_demand_gN_per_m3_per_day = plant_N_demand_gN_per_m3_per_day

    # Set up arrays
    available_nitrogen_for_immobilization_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    maximum_immobilization_gN_per_m3_per_day = np.ndarray(output_array_length, np.float64)
    immobilization_assumptions_violation = np.full(output_array_length, False, np.bool)
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

    # Start of period arrays are 1 longer, because we care about the end of period number output_array_length, which is
    # the start of period output_array_length + 1
    sop_ammonium_gN_per_m3 = np.ndarray(output_array_length + 1, np.float64)
    sop_nitrate_gN_per_m3 = np.ndarray(output_array_length + 1, np.float64)

    # Assign initial values
    sop_ammonium_gN_per_m3[0] = initial_ammonium_gN_per_m3
    sop_nitrate_gN_per_m3[0] = initial_nitrate_gN_per_m3

    # Simulate
    for i in range(output_array_length):
        input_index = int(i / time_stretch)
        available_nitrogen_for_immobilization_gN_per_m3_per_day[i] = porporato_parameters.MAX_PROPORTION_AMMONIUM_IMMOBILIZED_PER_DAY * sop_ammonium_gN_per_m3[i] + porporato_parameters.MAX_PROPORTION_NITRATE_IMMOBILIZED_PER_DAY * sop_nitrate_gN_per_m3[i]
        maximum_immobilization_gN_per_m3_per_day[i] = moisture_decomposition_factor[i] * available_nitrogen_for_immobilization_gN_per_m3_per_day[i]
        if net_flux_to_mineral_nitrogen_gN_per_m3_per_day[input_index] > 0:
            mineralization_gN_per_m3_per_day[i] = net_flux_to_mineral_nitrogen_gN_per_m3_per_day[input_index]
            total_immobilization_gN_per_m3_per_day[i] = 0
        else:
            mineralization_gN_per_m3_per_day[i] = 0
            total_immobilization_gN_per_m3_per_day[i] = -net_flux_to_mineral_nitrogen_gN_per_m3_per_day[input_index]
        if total_immobilization_gN_per_m3_per_day[i] > maximum_immobilization_gN_per_m3_per_day[i]:
            print("Insufficient nitrogen to meet immobilization demand, assumptions violated!")
            immobilization_assumptions_violation[i] = True

        nitrification_gN_per_m3_per_day[i] = nitrification_rate_constant[i] * sop_ammonium_gN_per_m3[i]

        # Duplicated between ammonium and nitrate
        immobilization_from_ammonium_gN_per_m3_per_day[i] = 0 if available_nitrogen_for_immobilization_gN_per_m3_per_day[i] == 0  else porporato_parameters.MAX_PROPORTION_AMMONIUM_IMMOBILIZED_PER_DAY * sop_ammonium_gN_per_m3[i] / available_nitrogen_for_immobilization_gN_per_m3_per_day[i] * total_immobilization_gN_per_m3_per_day[i]
        plant_passive_uptake_of_ammonium_gN_per_m3_per_day[i] = ammonium_passive_uptake_rate_constant[i] * sop_ammonium_gN_per_m3[i]
        maximum_active_ammonium_uptake_gN_per_m3_per_day[i] = ammonium_max_active_uptake_rate_constant[i] * sop_ammonium_gN_per_m3[i]
        plant_active_uptake_of_ammonium_gN_per_m3_per_day[i] = max(0.0, min(maximum_active_ammonium_uptake_gN_per_m3_per_day[i], plant_ammonium_demand_gN_per_m3_per_day[i] - plant_passive_uptake_of_ammonium_gN_per_m3_per_day[i]))
        ammonium_leaching_gN_per_m3_per_day[i] = ammonium_leaching_rate_constant[i] * sop_ammonium_gN_per_m3[i]

        immobilization_from_nitrate_gN_per_m3_per_day[i] = 0 if available_nitrogen_for_immobilization_gN_per_m3_per_day[i] == 0  else porporato_parameters.MAX_PROPORTION_NITRATE_IMMOBILIZED_PER_DAY * sop_nitrate_gN_per_m3[i] / available_nitrogen_for_immobilization_gN_per_m3_per_day[i] * total_immobilization_gN_per_m3_per_day[i]
        plant_passive_uptake_of_nitrate_gN_per_m3_per_day[i] = nitrate_passive_uptake_rate_constant[i] * sop_nitrate_gN_per_m3[i]
        maximum_active_nitrate_uptake_gN_per_m3_per_day[i] = nitrate_max_active_uptake_rate_constant[i] * sop_nitrate_gN_per_m3[i]
        plant_active_uptake_of_nitrate_gN_per_m3_per_day[i] = max(0.0, min(maximum_active_nitrate_uptake_gN_per_m3_per_day[i], plant_nitrate_demand_gN_per_m3_per_day[i] - plant_passive_uptake_of_nitrate_gN_per_m3_per_day[i]))
        nitrate_leaching_gN_per_m3_per_day[i] = nitrate_leaching_rate_constant[i] * sop_nitrate_gN_per_m3[i]

        # States
        ammonium_scaling_fix = 1.0
        sop_ammonium_gN_per_m3[i + 1] = sop_ammonium_gN_per_m3[i] + (mineralization_gN_per_m3_per_day[i] - immobilization_from_ammonium_gN_per_m3_per_day[i] - nitrification_gN_per_m3_per_day[i] - ammonium_leaching_gN_per_m3_per_day[i] - plant_passive_uptake_of_ammonium_gN_per_m3_per_day[i] - plant_active_uptake_of_ammonium_gN_per_m3_per_day[i]) * output_dt_days
        if sop_ammonium_gN_per_m3[i + 1] < 0:
            ammonium_scaling_fix = sop_ammonium_gN_per_m3[i] / (sop_ammonium_gN_per_m3[i] - sop_ammonium_gN_per_m3[i + 1])
            print("Ammonium underflow error! Applying scaling fix of", ammonium_scaling_fix)
            sop_ammonium_gN_per_m3[i + 1] = 0.0
        sop_nitrate_gN_per_m3[i + 1] = sop_nitrate_gN_per_m3[i] + (nitrification_gN_per_m3_per_day[i] * ammonium_scaling_fix - immobilization_from_nitrate_gN_per_m3_per_day[i] - nitrate_leaching_gN_per_m3_per_day[i] - plant_passive_uptake_of_nitrate_gN_per_m3_per_day[i] - plant_active_uptake_of_nitrate_gN_per_m3_per_day[i]) * output_dt_days

    # if any(available_nitrogen_for_immobilization_gN_per_m3_per_day < 0) or any(maximum_immobilization_gN_per_m3_per_day < 0) or any(immobilization_assumptions_violation) or any(nitrification_gN_per_m3_per_day < 0) or any(mineralization_gN_per_m3_per_day < 0) or any(total_immobilization_gN_per_m3_per_day < 0) or any(plant_passive_uptake_of_ammonium_gN_per_m3_per_day < 0) or any(plant_active_uptake_of_ammonium_gN_per_m3_per_day < 0) or any(plant_passive_uptake_of_nitrate_gN_per_m3_per_day < 0) or any(plant_active_uptake_of_nitrate_gN_per_m3_per_day < 0) or any(ammonium_leaching_gN_per_m3_per_day < 0) or any(nitrate_leaching_gN_per_m3_per_day < 0) or any(ammonium_gN_per_m3 < 0) or any(ammonium_gN_per_m3 > 500) or any(nitrate_gN_per_m3 < 0) or any(nitrate_gN_per_m3 > 500):
    #     print("Inorganic nitrogen sanity check failed!")

    # We return the start-of-period values for periods 1 to n+1, which is equivalent to end-of-period values for periods 0 to n.
    return sop_ammonium_gN_per_m3[1:], sop_nitrate_gN_per_m3[1:], ammonium_leaching_gN_per_m3_per_day, nitrate_leaching_gN_per_m3_per_day, plant_passive_uptake_of_ammonium_gN_per_m3_per_day, plant_passive_uptake_of_nitrate_gN_per_m3_per_day, plant_active_uptake_of_ammonium_gN_per_m3_per_day, plant_active_uptake_of_nitrate_gN_per_m3_per_day, immobilization_assumptions_violation, nitrification_gN_per_m3_per_day