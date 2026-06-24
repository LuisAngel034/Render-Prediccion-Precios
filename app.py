from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "pipeline_random_forest_pca.joblib")
    )
)

METADATA_PATH = Path(
    os.getenv(
        "METADATA_PATH",
        str(BASE_DIR / "metadatos_modelo.json")
    )
)

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

OPTIONS = {
    "airline": [
        ("SpiceJet", "SpiceJet"),
        ("AirAsia", "AirAsia"),
        ("Vistara", "Vistara"),
        ("GO_FIRST", "GO FIRST"),
        ("Indigo", "Indigo"),
        ("Air_India", "Air India"),
    ],
    "city": [
        ("Delhi", "Delhi"),
        ("Mumbai", "Mumbai"),
        ("Bangalore", "Bangalore"),
        ("Kolkata", "Kolkata"),
        ("Hyderabad", "Hyderabad"),
        ("Chennai", "Chennai"),
    ],
    "time": [
        ("Early_Morning", "Temprano en la mañana"),
        ("Morning", "Mañana"),
        ("Afternoon", "Tarde"),
        ("Evening", "Atardecer"),
        ("Night", "Noche"),
        ("Late_Night", "Madrugada"),
    ],
    "stops": [
        ("zero", "Vuelo directo"),
        ("one", "Una escala"),
        ("two_or_more", "Dos o más escalas"),
    ],
    "class": [
        ("Economy", "Económica"),
        ("Business", "Ejecutiva"),
    ],
}

EXPECTED_COLUMNS = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left",
]

DEFAULT_FORM = {
    "airline": "Vistara",
    "source_city": "Delhi",
    "departure_time": "Morning",
    "stops": "zero",
    "arrival_time": "Afternoon",
    "destination_city": "Mumbai",
    "class": "Economy",
    "duration": "2.17",
    "days_left": "15",
}

def load_metadata() -> dict[str, Any]:
    """Carga el archivo JSON de metadatos, si existe."""

    if not METADATA_PATH.exists():
        logger.warning(
            "No se encontró el archivo de metadatos: %s",
            METADATA_PATH
        )
        return {}

    try:
        with METADATA_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except (OSError, json.JSONDecodeError) as error:
        logger.warning(
            "No fue posible leer los metadatos: %s",
            error
        )
        return {}


def load_model():
    """Carga el pipeline completo almacenado mediante Joblib."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el pipeline en: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


METADATA = load_metadata()
MODEL = None
MODEL_LOAD_ERROR = None

try:
    MODEL = load_model()

    logger.info(
        "Pipeline cargado correctamente desde %s",
        MODEL_PATH
    )

    expected_version = METADATA.get("version_sklearn")

    if (
        expected_version
        and expected_version != sklearn.__version__
    ):
        logger.warning(
            "El pipeline fue almacenado con scikit-learn %s, "
            "pero la aplicación usa %s.",
            expected_version,
            sklearn.__version__
        )

except Exception as error:
    MODEL_LOAD_ERROR = str(error)

    logger.exception(
        "No fue posible cargar el pipeline."
    )

def allowed_values(option_name: str) -> set[str]:
    """Devuelve los valores internos válidos de una lista."""

    return {
        value
        for value, _label in OPTIONS[option_name]
    }


def validate_choice(
    form_data: dict[str, str],
    field_name: str,
    option_name: str,
    visible_name: str,
    errors: list[str],
) -> str:
    """Valida que una opción pertenezca a las categorías permitidas."""

    value = form_data.get(
        field_name,
        ""
    ).strip()

    if value not in allowed_values(option_name):
        errors.append(
            f"Selecciona un valor válido para {visible_name}."
        )

    return value


def parse_float(
    form_data: dict[str, str],
    field_name: str,
    visible_name: str,
    minimum: float,
    maximum: float,
    errors: list[str],
) -> float | None:
    """Convierte y valida un número decimal."""

    raw_value = form_data.get(
        field_name,
        ""
    ).strip()

    try:
        value = float(
            raw_value.replace(",", ".")
        )

    except ValueError:
        errors.append(
            f"{visible_name} debe ser un número válido."
        )
        return None

    if not np.isfinite(value):
        errors.append(
            f"{visible_name} debe ser un número finito."
        )
        return None

    if value < minimum or value > maximum:
        errors.append(
            f"{visible_name} debe encontrarse entre "
            f"{minimum} y {maximum}."
        )

    return value


def parse_integer(
    form_data: dict[str, str],
    field_name: str,
    visible_name: str,
    minimum: int,
    maximum: int,
    errors: list[str],
) -> int | None:
    """Convierte y valida un número entero."""

    raw_value = form_data.get(
        field_name,
        ""
    ).strip()

    try:
        numeric_value = float(raw_value)

    except ValueError:
        errors.append(
            f"{visible_name} debe ser un número entero."
        )
        return None

    if not numeric_value.is_integer():
        errors.append(
            f"{visible_name} no puede contener decimales."
        )
        return None

    value = int(numeric_value)

    if value < minimum or value > maximum:
        errors.append(
            f"{visible_name} debe encontrarse entre "
            f"{minimum} y {maximum}."
        )

    return value


def create_input_dataframe(
    form_data: dict[str, str]
) -> tuple[pd.DataFrame | None, list[str]]:
    """Valida el formulario y crea el DataFrame para el pipeline."""

    errors: list[str] = []

    airline = validate_choice(
        form_data,
        "airline",
        "airline",
        "la aerolínea",
        errors,
    )

    source_city = validate_choice(
        form_data,
        "source_city",
        "city",
        "la ciudad de origen",
        errors,
    )

    departure_time = validate_choice(
        form_data,
        "departure_time",
        "time",
        "el horario de salida",
        errors,
    )

    stops = validate_choice(
        form_data,
        "stops",
        "stops",
        "el número de escalas",
        errors,
    )

    arrival_time = validate_choice(
        form_data,
        "arrival_time",
        "time",
        "el horario de llegada",
        errors,
    )

    destination_city = validate_choice(
        form_data,
        "destination_city",
        "city",
        "la ciudad de destino",
        errors,
    )

    ticket_class = validate_choice(
        form_data,
        "class",
        "class",
        "la clase del boleto",
        errors,
    )

    duration = parse_float(
        form_data,
        "duration",
        "La duración",
        0.50,
        60.00,
        errors,
    )

    days_left = parse_integer(
        form_data,
        "days_left",
        "Los días restantes",
        1,
        49,
        errors,
    )

    if (
        source_city
        and destination_city
        and source_city == destination_city
    ):
        errors.append(
            "La ciudad de origen y la ciudad de destino "
            "deben ser diferentes."
        )

    if errors:
        return None, errors

    input_data = pd.DataFrame(
        [
            {
                "airline": airline,
                "source_city": source_city,
                "departure_time": departure_time,
                "stops": stops,
                "arrival_time": arrival_time,
                "destination_city": destination_city,
                "class": ticket_class,
                "duration": duration,
                "days_left": days_left,
            }
        ],
        columns=EXPECTED_COLUMNS,
    )

    return input_data, errors


@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():
    """Muestra el formulario y procesa las predicciones."""

    form_data = DEFAULT_FORM.copy()
    errors: list[str] = []

    prediction = None
    prediction_text = None

    if request.method == "POST":
        form_data.update(
            request.form.to_dict(
                flat=True
            )
        )

        if MODEL is None:
            errors.append(
                "El pipeline no está disponible. "
                "Revisa el archivo del modelo y las versiones instaladas."
            )

        else:
            input_data, validation_errors = (
                create_input_dataframe(
                    form_data
                )
            )

            errors.extend(
                validation_errors
            )

            if not errors and input_data is not None:
                try:
                    prediction = float(
                        MODEL.predict(
                            input_data
                        )[0]
                    )

                    if not np.isfinite(prediction):
                        raise ValueError(
                            "La predicción no es un número finito."
                        )

                    if prediction <= 0:
                        raise ValueError(
                            "El modelo produjo un precio no válido."
                        )

                    prediction_text = (
                        f"₹ {prediction:,.2f}"
                    )

                    logger.info(
                        "Predicción generada: %.2f",
                        prediction
                    )

                except Exception:
                    logger.exception(
                        "Error al generar la predicción."
                    )

                    errors.append(
                        "No fue posible generar la predicción. "
                        "Verifica los datos e intenta nuevamente."
                    )

    return render_template(
        "index.html",
        options=OPTIONS,
        form_data=form_data,
        errors=errors,
        prediction=prediction,
        prediction_text=prediction_text,
        model_name=METADATA.get(
            "nombre",
            "Random Forest con PCA"
        ),
        model_available=MODEL is not None,
        model_error=MODEL_LOAD_ERROR,
    )


@app.route(
    "/health",
    methods=["GET"]
)
def health():
    """Ruta para verificar el estado del servicio."""

    if MODEL is None:
        return jsonify(
            {
                "status": "error",
                "model_loaded": False,
                "detail": MODEL_LOAD_ERROR,
            }
        ), 503

    return jsonify(
        {
            "status": "ok",
            "model_loaded": True,
            "model": METADATA.get(
                "nombre",
                "Random Forest con PCA"
            ),
            "sklearn_version": sklearn.__version__,
        }
    ), 200

if __name__ == "__main__":
    port = int(
        os.getenv(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True,
    )
