# data_manager.py
import pandas as pd
import os
from pathlib import Path

# --- Utilizzo di percorsi relativi ---
# Ottiene il percorso della directory corrente dello script
CURRENT_DIR = Path(__file__).parent.absolute()
# Risali alla directory principale del progetto (2 livelli sopra src/data)
PROJECT_ROOT = CURRENT_DIR.parent.parent
# Percorsi relativi per i file CSV
ATTRACTIONS_CSV_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'attrazioni_roma.csv')
TOURISTS_CSV_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'preferenze_turista.csv')


def load_csv_to_dataframe(file_path):
    """
    Carica un file CSV in un DataFrame pandas.
    Gestisce FileNotFoundError e altri errori in modo robusto.

    Args:
        file_path (str): Il percorso del file CSV.

    Returns:
        pandas.DataFrame or None: Il DataFrame caricato o None se il file non esiste.
    """
    try:
        # Verifica l'esistenza del file
        if not os.path.exists(file_path):
            print(f"Errore: File non trovato: {file_path}")
            print(f"Percorso assoluto cercato: {os.path.abspath(file_path)}")
            # Verifica se esiste la directory
            dir_path = os.path.dirname(file_path)
            if not os.path.exists(dir_path):
                print(f"La directory {dir_path} non esiste.")
            else:
                print(f"Contenuto della directory {dir_path}:")
                for file in os.listdir(dir_path):
                    print(f"  - {file}")
            return None

        # Carica il file CSV
        df = pd.read_csv(file_path)
        print(f"Caricato con successo: {file_path} ({len(df)} righe)")
        return df
    except pd.errors.EmptyDataError:
        print(f"Errore: Il file {file_path} è vuoto.")
        return None
    except pd.errors.ParserError:
        print(f"Errore: Il file {file_path} non è un CSV valido.")
        return None
    except Exception as e:
        print(f"Errore imprevisto durante il caricamento di {file_path}: {e}")
        return None


def load_attractions(file_path=None):
    """
    Carica il dataset delle attrazioni.

    Args:
        file_path (str, optional): Percorso personalizzato del CSV. Se None, usa il percorso predefinito.

    Returns:
        pandas.DataFrame or None: DataFrame delle attrazioni o None in caso di errore.
    """
    print("Caricamento dati attrazioni...")
    if file_path is None:
        file_path = ATTRACTIONS_CSV_PATH
    return load_csv_to_dataframe(file_path)


def load_tourists(file_path=None):
    """
    Carica il dataset delle preferenze dei turisti.

    Args:
        file_path (str, optional): Percorso personalizzato del CSV. Se None, usa il percorso predefinito.

    Returns:
        pandas.DataFrame or None: DataFrame dei turisti o None in caso di errore.
    """
    print("Caricamento dati turisti...")
    if file_path is None:
        file_path = TOURISTS_CSV_PATH
    return load_csv_to_dataframe(file_path)


def get_attraction_details(attractions_df, attraction_id):
    """
    Restituisce i dettagli di una specifica attrazione come dizionario.

    Args:
        attractions_df (pandas.DataFrame): DataFrame delle attrazioni.
        attraction_id (int or str): L'ID dell'attrazione da cercare.

    Returns:
        dict or None: Un dizionario con i dettagli dell'attrazione o None se non trovata.
    """
    if attractions_df is None:
        print("Errore: DataFrame delle attrazioni non valido.")
        return None

    try:
        # Converti l'ID in intero se è una stringa
        numeric_id = int(attraction_id)

        # Cerca l'attrazione
        attraction = attractions_df[attractions_df['id_attrazione'] == numeric_id]

        if not attraction.empty:
            return attraction.iloc[0].to_dict()
        else:
            print(f"Attenzione: Attrazione con ID {attraction_id} non trovata.")
            return None
    except ValueError:
        print(f"Errore: attraction_id '{attraction_id}' non è un intero valido.")
        return None
    except KeyError:
        print("Errore: La colonna 'id_attrazione' non esiste nel DataFrame.")
        # Mostra le colonne disponibili per il debug
        if attractions_df is not None:
            print(f"Colonne disponibili: {attractions_df.columns.tolist()}")
        return None
    except Exception as e:
        print(f"Errore imprevisto: {e}")
        return None


def get_tourist_profile(tourists_df, tourist_id):
    """
    Restituisce il profilo di un specifico turista come dizionario.

    Args:
        tourists_df (pandas.DataFrame): DataFrame dei turisti.
        tourist_id (int or str): L'ID del turista da cercare.

    Returns:
        dict or None: Un dizionario con il profilo del turista o None se non trovato.
    """
    if tourists_df is None:
        print("Errore: DataFrame dei turisti non valido.")
        return None

    try:
        # Converti l'ID in intero se è una stringa
        numeric_id = int(tourist_id)

        # Cerca il turista
        tourist = tourists_df[tourists_df['id_turista'] == numeric_id]

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
        # Mostra le colonne disponibili per il debug
        if tourists_df is not None:
            print(f"Colonne disponibili: {tourists_df.columns.tolist()}")
        return None
    except Exception as e:
        print(f"Errore imprevisto: {e}")
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
        print("Errore: DataFrame delle attrazioni non valido.")
        return []

    try:
        # 'records' orient converte ogni riga in un dizionario
        return attractions_df.to_dict('records')
    except Exception as e:
        print(f"Errore durante la conversione del DataFrame in lista: {e}")
        return []


# --- Funzione per verificare l'accesso ai dati ---
def check_data_access():
    """
    Funzione di utilità per verificare l'accesso ai dati e i percorsi.
    Utile per il debugging di problemi relativi ai percorsi dei file.

    Returns:
        bool: True se tutti i test passano, False altrimenti.
    """
    success = True
    print("=" * 60)
    print("VERIFICA ACCESSO AI DATI")
    print("=" * 60)

    # Verifica percorsi
    print(f"Directory corrente: {os.getcwd()}")
    print(f"Directory dello script: {CURRENT_DIR}")
    print(f"Directory principale del progetto: {PROJECT_ROOT}")
    print(f"Percorso attrazioni: {ATTRACTIONS_CSV_PATH}")
    print(f"Percorso turisti: {TOURISTS_CSV_PATH}")

    # Verifica esistenza file
    print("\nVerifica esistenza file:")
    for path, name in [(ATTRACTIONS_CSV_PATH, "Attrazioni"), (TOURISTS_CSV_PATH, "Turisti")]:
        if os.path.exists(path):
            print(f"✅ File {name} trovato: {path}")
        else:
            print(f"❌ File {name} NON trovato: {path}")
            success = False

    # Prova a caricare i dati
    print("\nProva caricamento dati:")
    try:
        attractions = load_attractions()
        if attractions is not None:
            print(f"✅ Caricamento attrazioni riuscito: {len(attractions)} righe")
            # Mostra le prime righe
            print("\nPrime 2 righe attrazioni:")
            print(attractions.head(2))
        else:
            print("❌ Caricamento attrazioni fallito")
            success = False

        tourists = load_tourists()
        if tourists is not None:
            print(f"✅ Caricamento turisti riuscito: {len(tourists)} righe")
            # Mostra le prime righe
            print("\nPrime 2 righe turisti:")
            print(tourists.head(2))
        else:
            print("❌ Caricamento turisti fallito")
            success = False
    except Exception as e:
        print(f"❌ Errore durante il test di caricamento: {e}")
        success = False

    print("=" * 60)
    return success


# --- Blocco Esempio/Test (eseguito solo se lo script è lanciato direttamente) ---
if __name__ == "__main__":
    print("-" * 60)
    print("Test Modulo Data Manager")
    print("-" * 60)

    # Esegui verifica accesso ai dati
    check_data_access()

    # Carica i dati
    attractions_data = load_attractions()
    tourists_data = load_tourists()

    if attractions_data is not None:
        print("\nTest get_attraction_details:")
        # Cerca Colosseo (ID 3)
        details_colosseo = get_attraction_details(attractions_data, 3)
        if details_colosseo:
            print("Dettagli Colosseo (ID 3):")
            for key, value in details_colosseo.items():
                print(f"  {key}: {value}")

        # Test con ID non esistente
        details_inesistente = get_attraction_details(attractions_data, 999)

        print("\nTest get_all_attractions_list (prime 2):")
        all_attractions = get_all_attractions_list(attractions_data)
        if all_attractions:
            print(all_attractions[:2])

    if tourists_data is not None:
        print("\nTest get_tourist_profile:")
        # Cerca Turista 1
        profile_turista1 = get_tourist_profile(tourists_data, 1)
        if profile_turista1:
            print("Profilo Turista 1:")
            for key, value in profile_turista1.items():
                print(f"  {key}: {value}")

        # Test con ID non esistente
        profile_inesistente = get_tourist_profile(tourists_data, 50)

    print("-" * 60)
    print("Fine Test Modulo Data Manager")
    print("-" * 60)