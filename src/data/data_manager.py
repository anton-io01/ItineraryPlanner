import pandas as pd
import os # Per controllare l'esistenza dei file

# --- Costanti per i nomi dei file (modifica se necessario) ---
ATTRACTIONS_CSV_PATH = '/home/anton/PythonProject/ICON/datasets/attrazioni_roma.csv'
TOURISTS_CSV_PATH = '/home/anton/PythonProject/ICON/datasets/preferenze_turista.csv'

# --- Funzioni di Caricamento ---

def load_csv_to_dataframe(file_path):
    """
    Carica un file CSV in un DataFrame pandas.
    Gestisce FileNotFoundError.

    Args:
        file_path (str): Il percorso del file CSV.

    Returns:
        pandas.DataFrame or None: Il DataFrame caricato o None se il file non esiste.
    """
    if not os.path.exists(file_path):
        print(f"Errore: File non trovato: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"Caricato con successo: {file_path} ({len(df)} righe)")
        return df
    except Exception as e:
        print(f"Errore durante il caricamento di {file_path}: {e}")
        return None

def load_attractions(file_path=ATTRACTIONS_CSV_PATH):
    """Carica il dataset delle attrazioni."""
    print("Caricamento dati attrazioni...")
    return load_csv_to_dataframe(file_path)

def load_tourists(file_path=TOURISTS_CSV_PATH):
    """Carica il dataset delle preferenze dei turisti."""
    print("Caricamento dati turisti...")
    return load_csv_to_dataframe(file_path)

# --- Funzioni di Accesso ai Dati ---

def get_attraction_details(attractions_df, attraction_id):
    """
    Restituisce i dettagli di una specifica attrazione come dizionario.

    Args:
        attractions_df (pandas.DataFrame): DataFrame delle attrazioni.
        attraction_id (int): L'ID dell'attrazione da cercare.

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

def get_tourist_profile(tourists_df, tourist_id):
    """
    Restituisce il profilo di un specifico turista come dizionario.

    Args:
        tourists_df (pandas.DataFrame): DataFrame dei turisti.
        tourist_id (int): L'ID del turista da cercare.

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

def get_all_attractions_list(attractions_df):
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

# --- Blocco Esempio/Test (eseguito solo se lo script è lanciato direttamente) ---
if __name__ == "__main__":
    print("-" * 30)
    print("Esecuzione Test Modulo Data Manager")
    print("-" * 30)

    # Carica i dati
    attractions_data = load_attractions()
    tourists_data = load_tourists()

    print("-" * 30)

    if attractions_data is not None:
        print("\nPrime 5 righe del DataFrame Attrazioni:")
        print(attractions_data.head())

        print("\nTest get_attraction_details:")
        details_colosseo = get_attraction_details(attractions_data, 3) # Cerca Colosseo (ID 3)
        if details_colosseo:
            print("Dettagli Colosseo (ID 3):")
            for key, value in details_colosseo.items():
                print(f"  {key}: {value}")
        else:
            print("Colosseo non trovato.")

        details_inesistente = get_attraction_details(attractions_data, 999) # ID non esistente

        print("\nTest get_all_attractions_list (prime 2):")
        all_attractions = get_all_attractions_list(attractions_data)
        if all_attractions:
            print(all_attractions[:2])
        else:
            print("Nessuna attrazione caricata.")

    print("-" * 30)

    if tourists_data is not None:
        print("\nPrime 5 righe del DataFrame Turisti:")
        print(tourists_data.head())

        print("\nTest get_tourist_profile:")
        profile_turista1 = get_tourist_profile(tourists_data, 1) # Cerca Turista 1
        if profile_turista1:
            print("Profilo Turista 1:")
            for key, value in profile_turista1.items():
                print(f"  {key}: {value}")
        else:
            print("Turista 1 non trovato.")

        profile_inesistente = get_tourist_profile(tourists_data, 50) # ID non esistente

    print("-" * 30)
    print("Fine Test Modulo Data Manager")
    print("-" * 30)