from dataclasses import dataclass
from typing import Tuple, Callable

import numpy as np
import numpy.typing as npt

@dataclass
class KernelMetadata:
    num_trials_per_initial_condition: int
    num_control_steps: int
    accumulated_N_num_steps: int
    moisture_num_steps: int
    ammonium_num_steps: int
    nitrate_num_steps: int
    single_step_accumulated_N_num_steps: int

def find_optimal_cost_and_input(cost_to_go_from_cell_without_fertilizer: npt.NDArray[np.float64],
                                kernel_metadata: KernelMetadata, N_price_USD_per_g: float, soil_depth_m: float,
                                nitrogen_indices_to_values: Callable[[int, int], Tuple[float, float]],
                                output_cost_to_go: npt.NDArray[np.float64], output_ammonium_add_cells: npt.NDArray[np.uint8], output_nitrate_add_cells: npt.NDArray[np.uint8]) -> None:
    """

    :param cost_to_go_from_cell_without_fertilizer:
    :param kernel_metadata:
    :param N_price_USD_per_g:
    :param soil_depth_m: Soil depth in meters
    :param nitrogen_indices_to_values: Function mapping (ammonium index, nitrate index) to (ammonium quantity in g/m^3, nitrate quantity in g/m^3)
    :param output_cost_to_go: Output parameter
    :param output_ammonium_add_cells: Output parameter
    :param output_nitrate_add_cells: Output parameter
    """
    for initial_accumulated_N_index in range(kernel_metadata.accumulated_N_num_steps):
        for initial_forecast_index in range(2):
            for initial_moisture_index in range(kernel_metadata.moisture_num_steps):
                for initial_ammonium_index in range(kernel_metadata.ammonium_num_steps):
                    for initial_nitrate_index in range(kernel_metadata.nitrate_num_steps):
                        optimal_cost = cost_to_go_from_cell_without_fertilizer[initial_accumulated_N_index, initial_forecast_index, initial_moisture_index, initial_ammonium_index, initial_nitrate_index]
                        optimal_ammonium_add = 0
                        optimal_nitrate_add = 0
                        pre_add_ammonium, pre_add_nitrate = nitrogen_indices_to_values(initial_ammonium_index, initial_nitrate_index)
                        for post_add_ammonium_index in range(initial_ammonium_index, kernel_metadata.ammonium_num_steps):
                            for post_add_nitrate_index in range(initial_nitrate_index, kernel_metadata.nitrate_num_steps):
                                post_add_ammonium, post_add_nitrate = nitrogen_indices_to_values(post_add_ammonium_index, post_add_nitrate_index)
                                added_n_cost_USD_per_m2 = N_price_USD_per_g * soil_depth_m * (post_add_ammonium - pre_add_ammonium + post_add_nitrate - pre_add_nitrate)
                                total_cost = cost_to_go_from_cell_without_fertilizer[initial_accumulated_N_index, initial_forecast_index, initial_moisture_index, post_add_ammonium_index, post_add_nitrate_index] + added_n_cost_USD_per_m2
                                if total_cost < optimal_cost:
                                    optimal_cost = total_cost
                                    optimal_ammonium_add = post_add_ammonium_index - initial_ammonium_index
                                    optimal_nitrate_add = post_add_nitrate_index - initial_nitrate_index
                        output_cost_to_go[initial_accumulated_N_index, initial_forecast_index, initial_moisture_index, initial_ammonium_index, initial_nitrate_index] = optimal_cost
                        output_ammonium_add_cells[initial_accumulated_N_index, initial_forecast_index, initial_moisture_index, initial_ammonium_index, initial_nitrate_index] = optimal_ammonium_add
                        output_nitrate_add_cells[initial_accumulated_N_index, initial_forecast_index, initial_moisture_index, initial_ammonium_index, initial_nitrate_index] = optimal_nitrate_add

def calculate_terminal_step_probability_of_violation(transition_counts: npt.NDArray[np.uint8], kernel_metadata: KernelMetadata) -> npt.NDArray[np.float64]:
    return np.sum(transition_counts[-1], (5, 6, 7, 8), np.float64)[:, :, :, :, 1] / kernel_metadata.num_trials_per_initial_condition

def solve_dp_base_case(transition_counts: npt.NDArray[np.uint8], expected_plant_N_deficit_cost_USD: npt.NDArray[np.float64], kernel_metadata: KernelMetadata, N_price_USD_per_g: float, soil_depth_m: float, nitrogen_indices_to_values: Callable[[int, int], Tuple[float, float]], leaching_violation_penalty_USD_per_m2: float) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    # Dimensions are: control step, has leaching been violated, accumulated N, weather forecast, moisture, ammonium, nitrate
    optimal_cost_to_go_USD_per_m2 = np.ndarray(
        [kernel_metadata.num_control_steps, 2, kernel_metadata.accumulated_N_num_steps, 2, kernel_metadata.moisture_num_steps, kernel_metadata.ammonium_num_steps, kernel_metadata.nitrate_num_steps],
        np.float64)
    optimal_ammonium_add_in_cells = np.ndarray(
        [kernel_metadata.num_control_steps, 2, kernel_metadata.accumulated_N_num_steps, 2, kernel_metadata.moisture_num_steps, kernel_metadata.ammonium_num_steps, kernel_metadata.nitrate_num_steps],
        np.uint8)
    optimal_nitrate_add_in_cells = np.ndarray(
        [kernel_metadata.num_control_steps, 2, kernel_metadata.accumulated_N_num_steps, 2, kernel_metadata.moisture_num_steps, kernel_metadata.ammonium_num_steps, kernel_metadata.nitrate_num_steps],
        np.uint8)

    find_optimal_cost_and_input(expected_plant_N_deficit_cost_USD[-1], kernel_metadata, N_price_USD_per_g, soil_depth_m, nitrogen_indices_to_values, optimal_cost_to_go_USD_per_m2[-1, 1],
                                optimal_ammonium_add_in_cells[-1, 1], optimal_nitrate_add_in_cells[-1, 1])
    optimal_cost_to_go_USD_per_m2[
        -1, 1] += leaching_violation_penalty_USD_per_m2  # penalize violation by adding the penalty at the end

    # forecast, initial moisture, initial ammonium, initial nitrate
    terminal_step_probability_of_violation = calculate_terminal_step_probability_of_violation(transition_counts, kernel_metadata)
    leaching_terminal_cost = leaching_violation_penalty_USD_per_m2 * terminal_step_probability_of_violation
    leaching_penalty_adjusted_terminal_deficit_cost_USD_per_m2 = expected_plant_N_deficit_cost_USD[-1].copy()
    for i in range(kernel_metadata.accumulated_N_num_steps):
        leaching_penalty_adjusted_terminal_deficit_cost_USD_per_m2[i] += leaching_terminal_cost
    find_optimal_cost_and_input(leaching_penalty_adjusted_terminal_deficit_cost_USD_per_m2, kernel_metadata, N_price_USD_per_g, soil_depth_m, nitrogen_indices_to_values,
                                optimal_cost_to_go_USD_per_m2[-1, 0], optimal_ammonium_add_in_cells[-1, 0],
                                optimal_nitrate_add_in_cells[-1, 0])
    return optimal_cost_to_go_USD_per_m2, optimal_ammonium_add_in_cells, optimal_nitrate_add_in_cells

def solve_dp_recursive_cases(transition_counts: npt.NDArray[np.uint8],
                             expected_plant_N_deficit_cost_USD: npt.NDArray[np.float64],
                             kernel_metadata: KernelMetadata, N_price_USD_per_g: float, soil_depth_m: float,
                             nitrogen_indices_to_values: Callable[[int, int], Tuple[float, float]],
                             probability_of_clear_forecast: float,
                             optimal_cost_to_go_USD_per_m2: npt.NDArray[np.float64],
                             optimal_ammonium_add_in_cells: npt.NDArray[np.uint8],
                             optimal_nitrate_add_in_cells: npt.NDArray[np.uint8]) -> None:
    for i in reversed(range(kernel_metadata.num_control_steps - 1)):
        # leaching violated, accumulated N, forecast, starting moisture, starting ammonium, starting nitrate
        expected_cost_to_go_from_cell = np.ndarray(
            [2, kernel_metadata.accumulated_N_num_steps, 2, kernel_metadata.moisture_num_steps, kernel_metadata.ammonium_num_steps, kernel_metadata.nitrate_num_steps], np.float64)
        # has leaching been violated, accumulated N, moisture, ammonium, nitrate
        next_step_weather_averaged_optimal_cost_to_go_USD_per_m2 = (
                optimal_cost_to_go_USD_per_m2[i + 1, :, :, 0] * probability_of_clear_forecast +
                optimal_cost_to_go_USD_per_m2[i + 1, :, :, 1] * (1 - probability_of_clear_forecast))
        for accumulated_N_index in range(min((kernel_metadata.single_step_accumulated_N_num_steps * i + 1), kernel_metadata.accumulated_N_num_steps)):
        # for accumulated_N_index in range(kernel_metadata.accumulated_N_num_steps):
            accumulated_N_window_end = accumulated_N_index + kernel_metadata.single_step_accumulated_N_num_steps
            for forecast_index in range(2):
                for moisture_index in range(kernel_metadata.moisture_num_steps):
                    for ammonium_index in range(kernel_metadata.ammonium_num_steps):
                        for nitrate_index in range(kernel_metadata.nitrate_num_steps):
                            if accumulated_N_window_end > kernel_metadata.accumulated_N_num_steps:
                                # this array is of outcome counts, dimensions are [change in accumulated N, moisture, ammonium, nitrate]
                                windowed_violating_transition_counts = transition_counts[i, forecast_index, moisture_index, ammonium_index, nitrate_index, 1,
                                                                       :(kernel_metadata.accumulated_N_num_steps - accumulated_N_index)].copy()
                                windowed_non_violating_transition_counts = transition_counts[i, forecast_index, moisture_index, ammonium_index, nitrate_index, 0,
                                                                           :(kernel_metadata.accumulated_N_num_steps - accumulated_N_index)].copy()

                                windowed_violating_transition_counts[-1] += np.sum(transition_counts[i, forecast_index, moisture_index, ammonium_index, nitrate_index, 1,
                                                                                   (kernel_metadata.accumulated_N_num_steps - accumulated_N_index):], axis=0)
                                windowed_non_violating_transition_counts[-1] += np.sum(
                                    transition_counts[i, forecast_index, moisture_index, ammonium_index, nitrate_index,
                                    0,
                                    (kernel_metadata.accumulated_N_num_steps - accumulated_N_index):], axis=0)
                            else:
                                windowed_violating_transition_counts = transition_counts[i, forecast_index,
                                                                       moisture_index, ammonium_index, nitrate_index, 1]
                                windowed_non_violating_transition_counts = transition_counts[i, forecast_index,
                                                                           moisture_index, ammonium_index,
                                                                           nitrate_index, 0]

                            accumulated_N_window_upper_index_capped = min(accumulated_N_window_end, kernel_metadata.accumulated_N_num_steps)
                            expected_cost_to_go_from_cell[0, accumulated_N_index, forecast_index, moisture_index, ammonium_index, nitrate_index] = np.sum(windowed_non_violating_transition_counts * next_step_weather_averaged_optimal_cost_to_go_USD_per_m2[0, accumulated_N_index:accumulated_N_window_upper_index_capped] + windowed_violating_transition_counts * next_step_weather_averaged_optimal_cost_to_go_USD_per_m2[1, accumulated_N_index:accumulated_N_window_upper_index_capped]) / kernel_metadata.num_trials_per_initial_condition
                            expected_cost_to_go_from_cell[1, accumulated_N_index, forecast_index, moisture_index, ammonium_index, nitrate_index] = np.sum((windowed_violating_transition_counts + windowed_non_violating_transition_counts) * next_step_weather_averaged_optimal_cost_to_go_USD_per_m2[1, accumulated_N_index:accumulated_N_window_upper_index_capped]) / kernel_metadata.num_trials_per_initial_condition

        expected_cost_to_go_from_cell[0] += expected_plant_N_deficit_cost_USD[i]
        expected_cost_to_go_from_cell[1] += expected_plant_N_deficit_cost_USD[i]

        find_optimal_cost_and_input(expected_cost_to_go_from_cell[0], kernel_metadata, N_price_USD_per_g, soil_depth_m, nitrogen_indices_to_values, optimal_cost_to_go_USD_per_m2[i, 0],
                                    optimal_ammonium_add_in_cells[i, 0], optimal_nitrate_add_in_cells[i, 0])
        find_optimal_cost_and_input(expected_cost_to_go_from_cell[1], kernel_metadata, N_price_USD_per_g, soil_depth_m, nitrogen_indices_to_values, optimal_cost_to_go_USD_per_m2[i, 1],
                                    optimal_ammonium_add_in_cells[i, 1], optimal_nitrate_add_in_cells[i, 1])

def solve_dp(transition_counts: npt.NDArray[np.uint8], expected_plant_N_deficit_cost_USD: npt.NDArray[np.float64],
             kernel_metadata: KernelMetadata, N_price_USD_per_g: float, soil_depth_m: float,
             nitrogen_indices_to_values: Callable[[int, int], Tuple[float, float]], leaching_violation_penalty_USD_per_m2: float, probability_of_clear_forecast: float,) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    optimal_cost_to_go_USD_per_m2, optimal_ammonium_add_in_cells, optimal_nitrate_add_in_cells = solve_dp_base_case(transition_counts, expected_plant_N_deficit_cost_USD, kernel_metadata, N_price_USD_per_g, soil_depth_m, nitrogen_indices_to_values, leaching_violation_penalty_USD_per_m2)
    solve_dp_recursive_cases(transition_counts, expected_plant_N_deficit_cost_USD,  kernel_metadata, N_price_USD_per_g, soil_depth_m, nitrogen_indices_to_values, probability_of_clear_forecast, optimal_cost_to_go_USD_per_m2, optimal_ammonium_add_in_cells, optimal_nitrate_add_in_cells)
    return optimal_cost_to_go_USD_per_m2, optimal_ammonium_add_in_cells, optimal_nitrate_add_in_cells

def calculate_probabilities_of_violation(transition_counts: npt.NDArray[np.uint8],
                                         optimal_ammonium_add_in_cells: npt.NDArray[np.uint8],
                                         optimal_nitrate_add_in_cells: npt.NDArray[np.uint8],
                                         kernel_metadata: KernelMetadata,
                                         probability_of_clear_forecast: float) -> npt.NDArray[np.float64]:
    terminal_step_probability_of_violation = calculate_terminal_step_probability_of_violation(transition_counts, kernel_metadata)
    probabilities_of_violation = np.ndarray(
        [kernel_metadata.num_control_steps, kernel_metadata.accumulated_N_num_steps, 2, kernel_metadata.moisture_num_steps, kernel_metadata.ammonium_num_steps, kernel_metadata.nitrate_num_steps],
        np.float64)
    for initial_accumulated_N_index in range(kernel_metadata.accumulated_N_num_steps):
        for initial_forecast_index in range(2):
            for initial_moisture_index in range(kernel_metadata.moisture_num_steps):
                for initial_ammonium_index in range(kernel_metadata.ammonium_num_steps):
                    for initial_nitrate_index in range(kernel_metadata.nitrate_num_steps):
                        post_add_ammonium = initial_ammonium_index + optimal_ammonium_add_in_cells[
                            -1, 0, initial_accumulated_N_index, initial_forecast_index, initial_moisture_index, initial_ammonium_index, initial_nitrate_index]
                        post_add_nitrate = initial_nitrate_index + optimal_nitrate_add_in_cells[
                            -1, 0, initial_accumulated_N_index, initial_forecast_index, initial_moisture_index, initial_ammonium_index, initial_nitrate_index]
                        probabilities_of_violation[
                            -1, initial_accumulated_N_index, initial_forecast_index, initial_moisture_index, initial_ammonium_index, initial_nitrate_index] = \
                        terminal_step_probability_of_violation[
                            initial_forecast_index, initial_moisture_index, post_add_ammonium, post_add_nitrate]
    for i in reversed(range(kernel_metadata.num_control_steps - 1)):
        next_step_weather_averaged_probability_of_violation = (
                probabilities_of_violation[i + 1, :, 0] * probability_of_clear_forecast +
                probabilities_of_violation[i + 1, :, 1] * (1 - probability_of_clear_forecast))
        for initial_accumulated_N_index in range(min((kernel_metadata.single_step_accumulated_N_num_steps * i + 1), kernel_metadata.accumulated_N_num_steps)):
        # for initial_accumulated_N_index in range(kernel_metadata.accumulated_N_num_steps):
            for initial_forecast_index in range(2):
                for initial_moisture_index in range(kernel_metadata.moisture_num_steps):
                    for initial_ammonium_index in range(kernel_metadata.ammonium_num_steps):
                        for initial_nitrate_index in range(kernel_metadata.nitrate_num_steps):
                            post_add_ammonium = initial_ammonium_index + optimal_ammonium_add_in_cells[
                                i, 0, initial_accumulated_N_index, initial_forecast_index, initial_moisture_index, initial_ammonium_index, initial_nitrate_index]
                            post_add_nitrate = initial_nitrate_index + optimal_nitrate_add_in_cells[
                                i, 0, initial_accumulated_N_index, initial_forecast_index, initial_moisture_index, initial_ammonium_index, initial_nitrate_index]
                            accumulated_N_window_end = initial_accumulated_N_index + kernel_metadata.single_step_accumulated_N_num_steps
                            if accumulated_N_window_end > kernel_metadata.accumulated_N_num_steps:
                                # We only want the non-violating counts, violating is accounted for elsewhere
                                windowed_transition_counts = transition_counts[i, initial_forecast_index,
                                                             initial_moisture_index, post_add_ammonium,
                                                             post_add_nitrate, 0, :(
                                            kernel_metadata.accumulated_N_num_steps - initial_accumulated_N_index)].copy()  # accumulated N, moisture, ammonium, nitrate
                                windowed_transition_counts[-1] += np.sum(
                                    transition_counts[i, initial_forecast_index, initial_moisture_index,
                                    post_add_ammonium, post_add_nitrate, 0,
                                    (kernel_metadata.accumulated_N_num_steps - initial_accumulated_N_index):], axis=0)
                            else:
                                windowed_transition_counts = transition_counts[
                                    i, initial_forecast_index, initial_moisture_index, post_add_ammonium, post_add_nitrate, 0]
                            total_transition_counts = np.sum(windowed_transition_counts)
                            this_step_probability_of_violation = (kernel_metadata.num_trials_per_initial_condition - total_transition_counts) / kernel_metadata.num_trials_per_initial_condition
                            assert this_step_probability_of_violation >= 0
                            assert this_step_probability_of_violation <= 1
                            if total_transition_counts > 0:
                                probabilities_of_violation[
                                    i, initial_accumulated_N_index, initial_forecast_index, initial_moisture_index, initial_ammonium_index, initial_nitrate_index] = \
                                this_step_probability_of_violation + np.sum(
                                    windowed_transition_counts * next_step_weather_averaged_probability_of_violation[
                                                                 initial_accumulated_N_index:min(
                                                                     accumulated_N_window_end,
                                                                     kernel_metadata.accumulated_N_num_steps)]) / kernel_metadata.num_trials_per_initial_condition

                            else:
                                probabilities_of_violation[i, initial_accumulated_N_index, initial_forecast_index, initial_moisture_index, initial_ammonium_index, initial_nitrate_index] = 1.0
    return probabilities_of_violation