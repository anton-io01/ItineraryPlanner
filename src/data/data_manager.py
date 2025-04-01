# src/data/data_manager.py
import os
import pandas as pd
from typing import Optional, List, Dict, Any

# Percorsi relativi dei file CSV
ATTRACTIONS_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets',
                                    'attrazioni_roma.csv')
TOURISTS_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets',
                                 'preferenze_turista.csv')


def load_csv_to_dataframe(file_path: str) -> Optional[pd.DataFrame]:
    """
    Carica un file CSV in un DataFrame pandas.
    Gestisce FileNotFoundError e altri errori di caricamento.

    Args:
        file_path (str): Il percorso del file CSV.

    Returns:
        pandas.DataFrame or None: Il DataFrame caricato o None se il file non esiste o non può essere caricato.
    """
    try:
        if not os.path.exists(file_path):
            print(f"Errore: File non trovato: {file_path}")
            return None

        df = pd.read_csv(file_path)
        print(f"Caricato con successo: {file_path} ({len(df)} righe)")
        return df
    except Exception as e:
        print(f"Errore durante il caricamento di {file_path}: {e}")
        return None


def load_attractions(file_path: str = ATTRACTIONS_CSV_PATH) -> Optional[pd.DataFrame]:
    """Carica il dataset delle attrazioni."""
    print("Caricamento dati attrazioni...")
    return load_csv_to_dataframe(file_path)


def load_tourists(file_path: str = TOURISTS_CSV_PATH) -> Optional[pd.DataFrame]:
    """Carica il dataset delle preferenze dei turisti."""
    print("Caricamento dati turisti...")
    return load_csv_to_dataframe(file_path)


def get_attraction_details(attractions_df: pd.DataFrame, attraction_id: Union[str, int]) -> Optional[Dict[str, Any]]:
    """
    Restituisce i dettagli di una specifica attrazione come dizionario.

    Args:
        attractions_df (pandas.DataFrame): DataFrame delle attrazioni.
        attraction_id (int o str): L'ID dell'attrazione da cercare.

    Returns:
        dict or None: Un dizionario con i dettagli dell'attrazione o None se non trovata.
    """
    if attractions_df is None:
        return None

    try:
        # Assicurati che l'ID sia cercato come numero (se è numerico nel CSV)
        attraction = attractions_df[attractions_df['id_attrazione'] == int(attraction_id)]

        if not attraction.empty:
            # .iloc[0] seleziona la prima (e unica) riga trovata
            # .to_dict() converte la riga (Series) in un dizionario Python
            return attraction.iloc[0].to_dict()
        else:
            print(f"Attenzione: Attrazione con ID {attraction_id} non trovata.")
            return None
    except ValueError:
        print(f"Errore: attraction_id '{attraction_id}' non è un intero valido.")
        return None
    except KeyError:
        print("Errore: La colonna 'id_attrazione' non esiste nel DataFrame.")
        return None


def get_tourist_profile(tourists_df: pd.DataFrame, tourist_id: Union[str, int]) -> Optional[Dict[str, Any]]:
    """
    Restituisce il profilo di un specifico turista come dizionario.

    Args:
        tourists_df (pandas.DataFrame): DataFrame dei turisti.
        tourist_id (int o str): L'ID del turista da cercare.

    Returns:
        dict or None: Un dizionario con il profilo del turista o None se non trovato.
    """
    if tourists_df is None:
        return None

    try:
        # Assicurati che l'ID sia cercato come numero
        tourist = tourists_df[tourists_df['id_turista'] == int(tourist_id)]

        if not tourist.empty:
            return tourist.iloc[0].to_dict()
        else:
            print(f"Attenzione: Turista con ID {tourist_id} non trovato.")
            return None
    except ValueError:
        print(f"Errore: tourist_id '{tourist_id}' non è un intero valido.")
        return None
    except KeyError:
        print("Errore: La colonna 'id_turista' non esiste nel DataFrame.")
        return None


def get_all_attractions_list(attractions_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Restituisce tutte le attrazioni come una lista di dizionari.

    Args:
        attractions_df (pandas.DataFrame): DataFrame delle attrazioni.

    Returns:
        list: Lista di dizionari, ognuno rappresentante un'attrazione, o lista vuota.
    """
    if attractions_df is None:
        return []

    # 'records' orient converte ogni riga in un dizionario
    return attractions_df.to_dict('records')