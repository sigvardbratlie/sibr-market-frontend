# tests/test_main.py

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch
import os
from main import api

# Opprett en test-klient som kan sende forespørsler til appen
client = TestClient(api)


# 1. Test for et vellykket kall til det aggregerte endepunktet
def test_get_aggregated_data_success():
    """
    Tester et vellykket kall til /homes/{property_type}/{aggregation_level}.
    Vi "mocker" (etterligner) run_query-funksjonen for å unngå et reelt kall til BigQuery.
    """
    # Definer testdata som vår falske run_query skal returnere
    mock_dataframe = pd.DataFrame([
        {"county": "Oslo", "average_price": 5000000, "count": 120}
    ])

    # Bruk 'patch' til å midlertidig erstatte 'main.run_query'
    # Hver gang run_query blir kalt inne i denne testen, vil den returnere vår mock_dataframe
    # i stedet for å kjøre den ekte koden.
    with patch('main.run_query', return_value=mock_dataframe) as mock_run_query:
        # Utfør en GET-forespørsel til API-et
        response = client.get("/homes/houses/county?filter_value=oslo")

        # Verifiser at responsen er som forventet
        assert response.status_code == 200
        assert response.json() == [
            {"county": "Oslo", "average_price": 5000000, "count": 120}
        ]

        # (Valgfritt) Verifiser at den mockede funksjonen ble kalt
        mock_run_query.assert_called_once()
        # Du kan også sjekke nøyaktig HVA den ble kalt med
        # mock_run_query.assert_called_with(...)


# 2. Test for håndtering av ugyldig input
def test_get_aggregated_data_invalid_level():
    """
    Tester at API-et returnerer en 400-feil (Bad Request) hvis
    et ugyldig aggregeringsnivå blir brukt.
    """
    # Dette kallet skal aldri nå run_query, så vi trenger ikke mocke den.
    response = client.get("/homes/houses/invalid_level")

    # Verifiser at vi får riktig statuskode og feilmelding
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid aggregation level specified."}


# 3. Test for når ingen data blir funnet
def test_get_aggregated_data_not_found():
    """
    Tester at API-et returnerer en tom liste hvis spørringen ikke gir treff.
    """
    # Denne gangen skal vår falske run_query returnere en tom DataFrame
    with patch('main.run_query', return_value=pd.DataFrame()) as mock_run_query:
        response = client.get("/homes/apartments/municipality?filter_value=finnesikke")

        # Verifiser at responsen er vellykket, men innholdet er en tom liste
        assert response.status_code == 200
        assert response.json() == []

