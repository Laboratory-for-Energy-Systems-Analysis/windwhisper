"""Wind turbine specifications, model training and emission estimation."""

from typing import List, Tuple
from pathlib import Path
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import skops.io as sio
import xarray as xr


from . import DATA_DIR
from .windspeed import WindSpeed


def train_wind_turbine_model(file_path: str = None) -> Tuple[RegressorMixin, List[str]]:
    """Train the wind turbine noise emission model.

    :param file_path: Optional path to the training CSV file.
    :type file_path: str | None
    :returns: Trained scikit-learn model and the list of noise column names.
    :rtype: tuple[sklearn.base.RegressorMixin, list[str]]
    :raises FileNotFoundError: If the training dataset cannot be located.
    :raises ValueError: When the provided file is not a CSV file.
    """

    if file_path is None:
        file_path = Path(DATA_DIR / "training_data" / "noise_wind_turbines.csv")

    # File extension must be .csv
    if Path(file_path).suffix != ".csv":
        raise ValueError(f"The file extension for '{file_path}' must be '.csv'.")

    try:
        # Read the CSV file, skipping metadata rows
        df = pd.read_csv(file_path, skiprows=[1, 2])
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"The file '{file_path}' was not found.") from exc

    # List of noise columns
    noise_cols = [col for col in df.columns if "Noise" in col]

    # Select only the columns of interest
    cols_to_select = ["Power", "Diameter", "hub height [m]"] + noise_cols
    df = df[cols_to_select]

    # Remove rows where all values in noise columns are NaN
    df = df.dropna(subset=noise_cols, how="all")

    # Separate input and output data
    X = df[["Power", "Diameter", "hub height [m]"]]
    Y = df[noise_cols]
    print("Number of observations in whole set:", Y.shape[0])

    # Convert non-numeric values in 'X' to NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    # Convert non-numeric values in 'Y' to NaN
    Y = Y.apply(pd.to_numeric, errors="coerce")

    # Use mean imputation for missing input values
    imputer_X = SimpleImputer()
    X = pd.DataFrame(imputer_X.fit_transform(X), columns=X.columns)

    # Use kNN imputation for missing output values
    imputer_Y = KNNImputer(n_neighbors=5)
    Y = pd.DataFrame(imputer_Y.fit_transform(Y), columns=Y.columns)

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print("Number of observations in test set:", X_test.shape[0])
    print(f"Number of observations in training set: {X_train.shape[0]}")

    # Create and train the multi-output model
    model = MultiOutputRegressor(HistGradientBoostingRegressor())
    model.fit(X_train.values, Y_train.values)

    # Predict and evaluate the model
    Y_pred = model.predict(X_test.values)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(Y_test, Y_pred, multioutput="raw_values")
    print("Mean Squared Error (MSE):", mse)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Save the trained model for future use
    sio.dump(obj=(model, noise_cols), file=f"{Path(DATA_DIR / 'default_model')}.skops")

    # Print the location of the saved model
    print(f"Trained model saved to {Path(DATA_DIR / 'default_model')}.skops")

    return model, noise_cols


def load_model(filepath=None) -> Tuple[RegressorMixin, List[str]]:
    """Load a previously trained wind turbine emission model.

    :param filepath: Optional path to the ``.skops`` model archive.
    :type filepath: str | Path | None
    :returns: Trained model and the list of noise column names.
    :rtype: tuple[sklearn.base.RegressorMixin, list[str]]
    :raises FileNotFoundError: If the model archive does not exist.
    """

    if filepath is None:
        filepath = Path(DATA_DIR / "default_model.skops")

    # check that the model exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"The trained model file {filepath} was not found.")

    # Get the list of untrusted types
    unknown_types = sio.get_untrusted_types(file=filepath)

    # If these types are safe, load both model and noise columns
    model, noise_cols = sio.load(
        filepath, trusted = unknown_types
    )

    return model, noise_cols


def check_wind_turbine_specs(wind_turbines: dict) -> dict:
    """Validate user-provided turbine specifications.

    :param wind_turbines: Mapping of turbine identifiers to specification
        dictionaries.
    :type wind_turbines: dict
    :returns: Normalised turbine specifications with numeric values.
    :rtype: dict
    :raises KeyError: If a required field is missing.
    :raises ValueError: If numeric fields are invalid or positions are malformed.
    """

    mandatory_fields = ["power", "diameter", "hub height", "position"]

    for turbine, specs in wind_turbines.items():
        if not all(
            x in specs for x in mandatory_fields
        ):  # check that all mandatory fields are present
            raise KeyError(f"Missing mandatory field(s) in turbine {turbine}.")

    # check that `power`, `diameter` and `hub height` are numeric
    # and positive

    for turbine, specs in wind_turbines.items():
        for field in ["power", "diameter", "hub height"]:
            try:
                specs[field] = float(specs[field])
            except ValueError:
                raise ValueError(
                    f"The field '{field}' in turbine {turbine} must be numeric."
                ) from None

            if specs[field] <= 0:
                raise ValueError(
                    f"The field '{field}' in turbine {turbine} must be positive."
                )

    # check that the radius is inferior to the hub height
    for turbine, specs in wind_turbines.items():
        if specs["diameter"] / 2 > specs["hub height"]:
            raise ValueError(
                f"The radius of turbine {turbine} must be inferior to its hub height."
            )

    # check that `hub height` is inferior to 300 m and that `power`
    # is inferior to 20 MW
    for turbine, specs in wind_turbines.items():
        if specs["hub height"] > 300:
            raise ValueError(
                f"The hub height of turbine {turbine} must be inferior to 300 m."
            )
        if specs["power"] > 20000:
            raise ValueError(
                f"The power of turbine {turbine} must be inferior to 20 MW."
            )

    # check that the value for `position`is a tuple of two floats
    for turbine, specs in wind_turbines.items():
        if not isinstance(specs["position"], tuple):
            raise ValueError(f"The position of turbine {turbine} must be a tuple.")
        if len(specs["position"]) != 2:
            raise ValueError(
                f"The position of turbine {turbine} must contain two values."
            )
        if not all(isinstance(x, float) for x in specs["position"]):
            raise ValueError(
                f"The position of turbine {turbine} must contain two floats."
            )

    return wind_turbines


class WindTurbines:
    """Manage wind turbine specifications and noise emission predictions."""

    def __init__(
        self,
        wind_turbines: dict,
        model_file: str = None,
        retrain_model: bool = False,
        dataset_file: str = None,
        wind_speed_data: xr.DataArray | str = None,
        elevation_data: xr.DataArray | str = None,
        humidity: int = 70,
        temperature: int = 10,
    ):
        """Initialise the wind turbine manager and predict noise emissions.

        :param wind_turbines: Turbine specifications keyed by identifier.
        :type wind_turbines: dict
        :param model_file: Alternative ``.skops`` model archive to load.
        :type model_file: str | None
        :param retrain_model: When ``True`` retrain the model using ``dataset_file``.
        :type retrain_model: bool
        :param dataset_file: CSV dataset used for retraining the model.
        :type dataset_file: str | None
        :param wind_speed_data: Hourly wind speed dataset or path to a NetCDF file.
        :type wind_speed_data: xarray.DataArray | str | None
        :param elevation_data: Elevation dataset or path used during propagation.
        :type elevation_data: xarray.DataArray | str | None
        :param humidity: Relative humidity expressed as a percentage.
        :type humidity: int
        :param temperature: Air temperature in degrees Celsius.
        :type temperature: int
        """

        self.noise_propagation = None
        self.ws = None
        self.wind_turbines = check_wind_turbine_specs(wind_turbines)

        self.fetch_wind_speeds(wind_speed_data)

        if retrain_model:
            print("Retraining the model...")
            self.model, self.noise_cols = train_wind_turbine_model(dataset_file)
        else:
            try:
                self.model, self.noise_cols = load_model(model_file)
            except FileNotFoundError:
                self.model, self.noise_cols = train_wind_turbine_model(dataset_file)

        self.fetch_noise_level_vs_wind_speed()


    def fetch_noise_level_vs_wind_speed(self):
        """Predict emission spectra for wind speeds between 3 and 12 m/s."""

        # create xarray that stores the parameters for the list
        # of wind turbines passed by the user
        # plus the noise values predicted by the model

        pattern = re.compile(r"[-+]?\d*\.\d+|\d+")
        arr = xr.DataArray(
            np.zeros((len(self.wind_turbines), len(self.noise_cols))),
            dims=("turbine", "wind_speed"),
            coords={
                "turbine": list(self.wind_turbines.keys()),
                "wind_speed": [
                    float(re.findall(pattern, s)[0]) for s in self.noise_cols
                ],
            },
        )

        arr.coords["wind_speed"].attrs["units"] = "m/s"
        arr.coords["turbine"].attrs["units"] = None

        # convert self.wind_turbines into a numpy array
        # to be able to pass it to the model
        arr_input = np.array(
            [
                [specs["power"], specs["diameter"], specs["hub height"]]
                for turbine, specs in self.wind_turbines.items()
            ]
        )

        # predict the noise values
        arr.values = self.model.predict(arr_input)
        arr.loc[
            dict(wind_speed=arr.wind_speed < 3)
        ] = 0  # set noise to 0 for wind speeds < 3 m/s

        for turbine, specs in self.wind_turbines.items():
            specs["noise_vs_wind_speed"] = arr.loc[dict(turbine=turbine)]

    def plot_noise_curve(self):
        """Plot modelled noise levels for all turbines and wind speeds."""

        # Different line styles and markers
        line_styles = ["-", "--", "-.", ":"]
        markers = ["o", "^", "s", "p", "*", "+", "x", "D"]

        fig, ax = plt.subplots(figsize=(10, 6))

        i = 0
        for turbine, specs in self.wind_turbines.items():
            style = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            ax.plot(
                specs["noise_vs_wind_speed"],
                linestyle=style,
                marker=marker,
                label=turbine,
            )
            i += 1

        plt.title("Noise vs Wind Speed")
        plt.xlabel("Wind Speed (m/s)")
        plt.ylabel("Noise (dBa)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def fetch_wind_speeds(self, wind_speed_data: xr.DataArray | str = None):
        """Attach hourly wind speed profiles to each turbine specification.

        :param wind_speed_data: Precomputed dataset or path used to initialise
            :class:`windwhisper.windspeed.WindSpeed`.
        :type wind_speed_data: xarray.DataArray | str | None
        """

        self.wind_turbines = WindSpeed(
            wind_turbines=self.wind_turbines,
            wind_speed_data=wind_speed_data,
        ).wind_turbines

