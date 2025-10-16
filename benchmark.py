from typing import List
import numpy as np
import pandas as pd
import io
import tellurium as te
from sklearn.preprocessing import StandardScaler
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.utils.metric_utils import SampleLevelMetric


def encode(series: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    clip_range = (-5, 5)
    num_bins = 4096
    bin_edges = np.linspace(clip_range[0], clip_range[1], num_bins + 1)

    normed = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    normed = np.clip(normed, *clip_range)
    token_ids = np.digitize(normed, bin_edges) - 1
    return token_ids


def format_prompt(sample: dict) -> str:
    data = sample["series"]
    prev_data = sample["prev_data"]
    fin_str = ""
    for series in data:
        encoded = encode(np.array(series))
        fin_str += " ".join(map(str, encoded)) + "\n"

    prompt = f"""
    You are a model tasked with recovering the biological network corresponding 
    to the time series data provided.
    Previous info: {prev_data}
    RETURN ONLY THE MODEL IN ANTIMONY FORMAT. YOU MUST ESTIMATE ALL PARAMETERS. 
    YOU MUST COME UP WITH NEW REACTIONS

    Time Series Data:
    {fin_str}
    """
    return prompt.strip()


def prompt_fn(sample: dict) -> str:
    return format_prompt(sample)


def compute_error(sim_output: np.ndarray, expected_output: np.ndarray) -> float:
    # Mean squared error across all species/time points (excluding time column)
    mse = np.mean((sim_output - expected_output) ** 2)
    return mse


def sample_level_fn(sample: dict, completion: str) -> float:
    try:
        r = te.loada(completion)
        sim_result = r.simulate()
        sim_output = sim_result[:, 1:]  # drop time column

        # Parse expected_output CSV string to DataFrame
        expected_csv = sample["expected_output"]
        expected_df = pd.read_csv(io.StringIO(expected_csv))
        expected_output = expected_df.to_numpy()[:, 1:]  # drop time column

        error = compute_error(sim_output, expected_output)
        if error == 0:
            return 1e6  # very high score for perfect match
        else:
            return 1.0 / error
    except Exception:
        return 0.001  # penalty for errors


custom_metric = SampleLevelMetric(
    metric_name="tellurium_inverse_error",
    higher_is_better=True,
    category="accuracy",
    sample_level_fn=sample_level_fn,
    corpus_level_fn=np.mean,
)


task = LightevalTaskConfig(
    name="tellurium-benchmark",
    prompt_function=prompt_fn,
    suite=["community"],
    hf_repo="TheBobBob2/tellurium_benchmark",  # replace with your repo name
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,  # None for zero-shot
    few_shots_select=None,  # None for zero-shot
    metrics=[custom_metric],
    generation_size=256,
    stop_sequence=None,
)
