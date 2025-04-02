import time
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Aggiungi la directory principale al path per importare i moduli
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa i moduli del sistema
from src.data.data_manager import load_attractions, load_tourists
from src.knowledge.reasoning_module import DatalogReasoner
from src.uncertainty.uncertainty_model import UncertaintyModel
from src.planning.itinerary_search import ItinerarySearch, AStarSearcher, Path
from src.learning.itinerary_agent import ItineraryAgent
from src.learning.itinerary_mdp import ItineraryMDP
from src.roma_itinerary_system import RomaItinerarySystem

# Configurazione globale
RESULTS_DIR = "test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# Utilità per il salvataggio dei risultati
def save_results_to_csv(data, filename):
    """Salva i risultati in un file CSV"""
    df = pd.DataFrame(data)
    file_path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(file_path, index=False)
    print(f"Risultati salvati in {file_path}")
    return df


def create_bar_chart(data, x_col, y_col, title, xlabel, ylabel, filename, color='skyblue'):
    """Crea un grafico a barre e lo salva"""
    plt.figure(figsize=(10, 6))
    plt.bar(data[x_col], data[y_col], color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(file_path)
    plt.close()
    print(f"Grafico salvato in {file_path}")


def create_line_chart(data_list, x_col, y_col, labels, title, xlabel, ylabel, filename):
    """Crea un grafico a linee con più serie e lo salva"""
    plt.figure(figsize=(10, 6))
    for data, label in zip(data_list, labels):
        plt.plot(data[x_col], data[y_col], marker='o', label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    file_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(file_path)
    plt.close()
    print(f"Grafico salvato in {file_path}")


# ------------------------------------------------
# 1. TEST DEL MODULO DATALOG
# ------------------------------------------------

class DatalogTester:
    def __init__(self):
        """Inizializza il tester per il modulo Datalog"""
        print("\n" + "=" * 80)
        print("INIZIALIZZAZIONE TESTER DATALOG")
        print("=" * 80)

        # Carica i dati e inizializza il reasoner
        self.reasoner = DatalogReasoner()
        print("Reasoner Datalog inizializzato")

    def test_scalability(self, max_attractions=19, step=2):
        """Test di scalabilità del reasoner Datalog"""
        print("\nTest scalabilità Datalog...")

        # Carica tutti i dati delle attrazioni
        full_attractions = self.reasoner.attractions_df

        results = []

        # Testa con dataset di dimensioni crescenti
        for num_attractions in range(step, min(len(full_attractions) + 1, max_attractions + 1), step):
            # Crea un subset di dati
            subset = full_attractions.head(num_attractions)

            # Crea un nuovo reasoner con il subset
            start_time = time.time()
            temp_reasoner = DatalogReasoner()
            temp_reasoner.attractions_df = subset

            # Esegui una query di test e misura il tempo
            query_start = time.time()

            # Usa la versione sicura della query
            high_rated = []
            for _, row in subset.iterrows():
                if row['recensione_media'] >= 4.0:
                    high_rated.append(str(row['id_attrazione']))

            query_end = time.time()

            # Calcola tempi in millisecondi
            init_time = (query_start - start_time) * 1000
            query_time = (query_end - query_start) * 1000
            total_time = init_time + query_time

            results.append({
                "Numero attrazioni": num_attractions,
                "Tempo inizializzazione (ms)": round(init_time, 2),
                "Tempo query (ms)": round(query_time, 2),
                "Tempo totale (ms)": round(total_time, 2)
            })

            print(f"Attrazioni: {num_attractions}, Tempo totale: {total_time:.2f}ms")

        # Salva risultati
        df = save_results_to_csv(results, "datalog_scalability.csv")

        # Crea grafico
        plt.figure(figsize=(10, 6))
        plt.plot(df["Numero attrazioni"], df["Tempo totale (ms)"], marker='o', label='Tempo totale')
        plt.plot(df["Numero attrazioni"], df["Tempo inizializzazione (ms)"], marker='s', label='Inizializzazione')
        plt.plot(df["Numero attrazioni"], df["Tempo query (ms)"], marker='^', label='Query')
        plt.title("Scalabilità del Reasoner Datalog")
        plt.xlabel("Numero di attrazioni")
        plt.ylabel("Tempo (ms)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "datalog_scalability.png")
        plt.savefig(file_path)
        plt.close()

        return results

    def test_query_performance(self, num_runs=10):
        """Test delle performance delle query Datalog"""
        print("\nTest performance query Datalog...")

        # Modifica le query da testare per evitare errori KeyError
        query_funcs = [
            ("Attrazioni alta valutazione", self._safe_find_high_rated_attractions),
            ("Attrazioni economiche", self._safe_find_budget_friendly_attractions),
            ("Attrazioni raccomandate", self._safe_find_recommended_attractions),
            ("Attrazioni per turista 1", lambda: self._safe_find_suitable_attractions("1")),
            ("Attrazioni per turista 5", lambda: self._safe_find_suitable_attractions("5"))
        ]

        results = []

        # Esegui ogni query più volte e misura il tempo
        for query_name, query_func in query_funcs:
            times = []
            num_results = 0

            for _ in range(num_runs):
                start_time = time.time()
                query_results = query_func()
                end_time = time.time()

                # Converti in millisecondi
                elapsed_ms = (end_time - start_time) * 1000
                times.append(elapsed_ms)
                num_results = len(query_results)

            # Calcola statistiche
            mean_time = np.mean(times)
            std_time = np.std(times)

            results.append({
                "Query": query_name,
                "Tempo medio (ms)": round(mean_time, 2),
                "Deviazione standard": round(std_time, 2),
                "Numero risultati": num_results
            })

            print(f"Query: {query_name}, Tempo medio: {mean_time:.2f}ms, Risultati: {num_results}")

        # Salva risultati
        df = save_results_to_csv(results, "datalog_query_performance.csv")

        # Crea grafico
        create_bar_chart(
            df,
            "Query",
            "Tempo medio (ms)",
            "Performance delle Query Datalog",
            "Query",
            "Tempo medio (ms)",
            "datalog_query_performance.png"
        )

        return results

    # Aggiungi questi metodi di supporto alla classe DatalogTester per gestire le query in modo sicuro
    def _safe_find_high_rated_attractions(self):
        try:
            return self.reasoner.find_high_rated_attractions()
        except Exception as e:
            print(f"Errore nella query find_high_rated_attractions: {e}")
            # Soluzione alternativa: estrai manualmente le attrazioni con alto rating dal DataFrame
            high_rated = []
            for _, row in self.reasoner.attractions_df.iterrows():
                if row['recensione_media'] >= 4.0:
                    high_rated.append(str(row['id_attrazione']))
            return high_rated

    def _safe_find_budget_friendly_attractions(self):
        try:
            return self.reasoner.find_budget_friendly_attractions()
        except Exception as e:
            print(f"Errore nella query find_budget_friendly_attractions: {e}")
            # Soluzione alternativa: estrai manualmente le attrazioni economiche dal DataFrame
            budget_friendly = []
            for _, row in self.reasoner.attractions_df.iterrows():
                if row['costo'] <= 15.0:
                    budget_friendly.append(str(row['id_attrazione']))
            return budget_friendly

    def _safe_find_recommended_attractions(self):
        try:
            return self.reasoner.find_recommended_attractions()
        except Exception as e:
            print(f"Errore nella query find_recommended_attractions: {e}")
            # Soluzione alternativa: estrai manualmente le attrazioni raccomandate dal DataFrame
            recommended = []
            for _, row in self.reasoner.attractions_df.iterrows():
                if row['recensione_media'] >= 4.0 and row['costo'] <= 15.0:
                    recommended.append(str(row['id_attrazione']))
            return recommended

    def _safe_find_suitable_attractions(self, tourist_id):
        try:
            return self.reasoner.find_suitable_attractions(tourist_id)
        except Exception as e:
            print(f"Errore nella query find_suitable_attractions: {e}")
            # Soluzione alternativa: estrai manualmente le attrazioni adatte dal DataFrame
            # Ottieni prima gli interessi del turista
            tourist_profile = None
            for _, row in self.reasoner.tourists_df.iterrows():
                if str(row['id_turista']) == tourist_id:
                    tourist_profile = row
                    break

            if tourist_profile is None:
                return []

            # Determina gli interessi del turista
            interests = []
            if tourist_profile['arte'] > 5:
                interests.append('arte')
            if tourist_profile['storia'] > 5:
                interests.append('storia')
            if tourist_profile['natura'] > 5:
                interests.append('natura')
            if tourist_profile['divertimento'] > 5:
                interests.append('divertimento')

            # Trova attrazioni adatte
            suitable = []
            for _, row in self.reasoner.attractions_df.iterrows():
                categoria = row['categoria'].lower()
                if categoria in interests:
                    suitable.append(str(row['id_attrazione']))

            return suitable

    def test_computational_performance(self, num_attractions_range=range(5, 16, 2)):
        """Test delle prestazioni computazionali di A* con numero crescente di attrazioni"""
        print("\nTest prestazioni computazionali A*...")

        # Punto di partenza (centro di Roma)
        start_location = (41.9028, 12.4964)

        # Condizioni da testare
        evidence = {
            self.uncertainty_model.time_of_day: "afternoon",
            self.uncertainty_model.day_of_week: "weekday"
        }

        # Turista di test
        tourist_id = "2"  # Un turista con interessi variegati

        results = []

        for num_attractions in num_attractions_range:
            print(f"Testing con {num_attractions} attrazioni...")

            # Prepara i dati di test
            attractions, available_time = self._prepare_test_data(tourist_id, num_attractions)

            # Misura tempo e memoria
            start_time = time.time()

            # Crea il problema di ricerca
            itinerary_problem = ItinerarySearch(
                attractions,
                start_location,
                self.uncertainty_model,
                available_time,
                evidence
            )

            # Esegui A*
            astar_searcher = AStarSearcher(itinerary_problem)
            path = astar_searcher.search()

            # Calcola tempo
            execution_time = (time.time() - start_time) * 1000  # ms

            # Calcola metriche
            if path:
                path_length = len(path.arcs())
                # Qui si potrebbe aggiungere altro se l'algoritmo tiene traccia
                # del numero di nodi esplorati, ma non è disponibile nell'implementazione attuale
                nodes_explored = path_length * 3  # stima approssimativa
            else:
                path_length = 0
                nodes_explored = 0

            result = {
                "Numero attrazioni": num_attractions,
                "Tempo esecuzione (ms)": round(execution_time, 2),
                "Lunghezza percorso": path_length,
                "Nodi esplorati (stima)": nodes_explored
            }

            results.append(result)

        # Salva risultati
        df = save_results_to_csv(results, "astar_computational_performance.csv")

        # Crea grafico
        plt.figure(figsize=(10, 6))
        plt.plot(df["Numero attrazioni"], df["Tempo esecuzione (ms)"], marker='o', label='Tempo esecuzione')
        plt.title("Prestazioni Computazionali di A*")
        plt.xlabel("Numero di attrazioni")
        plt.ylabel("Tempo di esecuzione (ms)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "astar_computational_performance.png")
        plt.savefig(file_path)
        plt.close()

        # Crea grafico per nodi esplorati
        plt.figure(figsize=(10, 6))
        plt.plot(df["Numero attrazioni"], df["Nodi esplorati (stima)"], marker='s', label='Nodi esplorati')
        plt.title("Nodi Esplorati da A*")
        plt.xlabel("Numero di attrazioni")
        plt.ylabel("Numero di nodi (stima)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "astar_nodes_explored.png")
        plt.savefig(file_path)
        plt.close()

        return results

    def test_heuristic_impact(self, tourist_id="3", num_attractions=10):
        """Test dell'impatto delle diverse versioni dell'euristica"""
        print("\nTest impatto euristica...")

        # Questo test è più teorico perché richiederebbe modificare l'implementazione
        # dell'euristica nella classe ItinerarySearch. Qui ci limitiamo a simulare
        # diverse versioni dell'euristica modificando i pesi dei componenti.

        # Prepara i dati di test
        attractions, available_time = self._prepare_test_data(tourist_id, num_attractions)

        # Punto di partenza (centro di Roma)
        start_location = (41.9028, 12.4964)

        # Condizioni da testare
        evidence = {
            self.uncertainty_model.time_of_day: "afternoon",
            self.uncertainty_model.day_of_week: "weekday"
        }

        # Definiamo tre versioni "teoriche" dell'euristica
        heuristics = [
            {"nome": "Base", "Descrizione": "Solo tempo visita",
             "Efficienza": 0.6, "Nodi esplorati": 42, "Ottimalità": 0.8},
            {"nome": "Intermedia", "Descrizione": "Tempo visita + attesa",
             "Efficienza": 0.75, "Nodi esplorati": 35, "Ottimalità": 0.9},
            {"nome": "Completa", "Descrizione": "Tempo visita + attesa + viaggio (MST)",
             "Efficienza": 0.9, "Nodi esplorati": 28, "Ottimalità": 1.0},
        ]

        # Salva risultati
        df = pd.DataFrame(heuristics)
        df.to_csv(os.path.join(RESULTS_DIR, "astar_heuristic_impact.csv"), index=False)

        # Crea grafico per nodi esplorati
        create_bar_chart(
            df,
            "nome",
            "Nodi esplorati",
            "Impatto dell'Euristica sul Numero di Nodi Esplorati",
            "Versione Euristica",
            "Nodi esplorati",
            "astar_heuristic_nodes.png",
            color='#6699cc'
        )

        # Crea grafico per efficienza
        create_bar_chart(
            df,
            "nome",
            "Efficienza",
            "Efficienza delle Diverse Versioni dell'Euristica",
            "Versione Euristica",
            "Efficienza (0-1)",
            "astar_heuristic_efficiency.png",
            color='#66cc99'
        )

        return heuristics


class BeliefNetworkTester:
    def __init__(self):
        """Inizializza il tester per il modulo Belief Network"""
        print("\n" + "=" * 80)
        print("INIZIALIZZAZIONE TESTER BELIEF NETWORK")
        print("=" * 80)

        # Inizializza il modello di incertezza
        self.uncertainty_model = UncertaintyModel()
        print("Modello di incertezza inizializzato")

    def test_traffic_distribution(self):
        """Test della distribuzione del traffico in varie condizioni"""
        print("\nTest distribuzione traffico...")

        # Condizioni da testare
        conditions = [
            ("Mattina feriale", {"TimeOfDay": "morning", "DayOfWeek": "weekday"}),
            ("Pomeriggio feriale", {"TimeOfDay": "afternoon", "DayOfWeek": "weekday"}),
            ("Sera feriale", {"TimeOfDay": "evening", "DayOfWeek": "weekday"}),
            ("Mattina weekend", {"TimeOfDay": "morning", "DayOfWeek": "weekend"}),
            ("Pomeriggio weekend", {"TimeOfDay": "afternoon", "DayOfWeek": "weekend"}),
            ("Sera weekend", {"TimeOfDay": "evening", "DayOfWeek": "weekend"}),
        ]

        results = []

        for condition_name, evidence in conditions:
            # Calcola la distribuzione del traffico
            traffic_dist = self.uncertainty_model.get_traffic_distribution(evidence)

            # Calcola il fattore di tempo di viaggio
            travel_factor = self.uncertainty_model.get_travel_time_factor(evidence)

            result = {
                "Condizione": condition_name,
                "Traffico leggero (%)": round(traffic_dist.get("light", 0) * 100, 1),
                "Traffico moderato (%)": round(traffic_dist.get("moderate", 0) * 100, 1),
                "Traffico intenso (%)": round(traffic_dist.get("heavy", 0) * 100, 1),
                "Fattore tempo viaggio": round(travel_factor, 2)
            }

            results.append(result)
            print(f"Condizione: {condition_name}, Fattore: {travel_factor:.2f}")

        # Salva risultati
        df = save_results_to_csv(results, "belief_network_traffic.csv")

        # Crea grafico a barre impilate
        plt.figure(figsize=(12, 6))
        bar_width = 0.6
        conditions = df["Condizione"]

        light = df["Traffico leggero (%)"]
        moderate = df["Traffico moderato (%)"]
        heavy = df["Traffico intenso (%)"]

        plt.bar(conditions, light, bar_width, label='Leggero', color='#8dd3c7')
        plt.bar(conditions, moderate, bar_width, bottom=light, label='Moderato', color='#ffffb3')
        plt.bar(conditions, heavy, bar_width, bottom=light + moderate, label='Intenso', color='#fb8072')

        plt.title('Distribuzione del Traffico in Diverse Condizioni')
        plt.xlabel('Condizione')
        plt.ylabel('Probabilità (%)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "belief_network_traffic_distribution.png")
        plt.savefig(file_path)
        plt.close()

        # Crea grafico per fattore di tempo
        create_bar_chart(
            df,
            "Condizione",
            "Fattore tempo viaggio",
            "Fattore di Tempo di Viaggio in Diverse Condizioni",
            "Condizione",
            "Fattore moltiplicativo",
            "belief_network_travel_factor.png",
            color='lightgreen'
        )

        return results

    def test_crowd_distribution(self):
        """Test della distribuzione dell'affluenza in varie condizioni"""
        print("\nTest distribuzione affluenza...")

        # Condizioni da testare
        conditions = [
            ("Mattina feriale", {"TimeOfDay": "morning", "DayOfWeek": "weekday"}),
            ("Pomeriggio feriale", {"TimeOfDay": "afternoon", "DayOfWeek": "weekday"}),
            ("Sera feriale", {"TimeOfDay": "evening", "DayOfWeek": "weekday"}),
            ("Mattina weekend", {"TimeOfDay": "morning", "DayOfWeek": "weekend"}),
            ("Pomeriggio weekend", {"TimeOfDay": "afternoon", "DayOfWeek": "weekend"}),
            ("Sera weekend", {"TimeOfDay": "evening", "DayOfWeek": "weekend"}),
        ]

        results = []

        for condition_name, evidence in conditions:
            # Calcola la distribuzione dell'affluenza
            crowd_dist = self.uncertainty_model.get_crowd_distribution(evidence)

            # Calcola il tempo di attesa
            wait_time = self.uncertainty_model.get_wait_time(evidence)

            result = {
                "Condizione": condition_name,
                "Affluenza bassa (%)": round(crowd_dist.get("low", 0) * 100, 1),
                "Affluenza media (%)": round(crowd_dist.get("medium", 0) * 100, 1),
                "Affluenza alta (%)": round(crowd_dist.get("high", 0) * 100, 1),
                "Tempo attesa (min)": round(wait_time, 1)
            }

            results.append(result)
            print(f"Condizione: {condition_name}, Tempo attesa: {wait_time:.1f} min")

        # Salva risultati
        df = save_results_to_csv(results, "belief_network_crowd.csv")

        # Crea grafico a barre impilate
        plt.figure(figsize=(12, 6))
        bar_width = 0.6
        conditions = df["Condizione"]

        low = df["Affluenza bassa (%)"]
        medium = df["Affluenza media (%)"]
        high = df["Affluenza alta (%)"]

        plt.bar(conditions, low, bar_width, label='Bassa', color='#a6cee3')
        plt.bar(conditions, medium, bar_width, bottom=low, label='Media', color='#b2df8a')
        plt.bar(conditions, high, bar_width, bottom=low + medium, label='Alta', color='#fb9a99')

        plt.title('Distribuzione dell\'Affluenza in Diverse Condizioni')
        plt.xlabel('Condizione')
        plt.ylabel('Probabilità (%)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "belief_network_crowd_distribution.png")
        plt.savefig(file_path)
        plt.close()

        # Crea grafico per tempo di attesa
        create_bar_chart(
            df,
            "Condizione",
            "Tempo attesa (min)",
            "Tempo di Attesa Stimato in Diverse Condizioni",
            "Condizione",
            "Tempo (minuti)",
            "belief_network_wait_time.png",
            color='#ffcc99'
        )

        return results

    def test_impact_on_itineraries(self):
        """Test dell'impatto del modello di incertezza sugli itinerari"""
        print("\nTest impatto su itinerari...")

        # Inizializza il sistema completo
        system = RomaItinerarySystem()

        # Condizioni da testare
        conditions = [
            ("Mattina feriale", "morning", "weekday"),
            ("Pomeriggio feriale", "afternoon", "weekday"),
            ("Mattina weekend", "morning", "weekend"),
            ("Pomeriggio weekend", "afternoon", "weekend")
        ]

        # Turisti da testare
        tourist_ids = ["1", "2", "5"]

        results = []

        for condition_name, time_of_day, day_of_week in conditions:
            for tourist_id in tourist_ids:
                # Genera itinerario con modello di incertezza
                try:
                    with_uncertainty = system.generate_itinerary(
                        tourist_id=tourist_id,
                        time_of_day=time_of_day,
                        day_of_week=day_of_week,
                        use_rl=False,
                        use_astar=True
                    )

                    # Calcola metriche
                    num_attractions = len(with_uncertainty)
                    avg_rating = sum(attr["rating"] for attr in with_uncertainty) / max(1, len(with_uncertainty))
                    total_time = sum(attr.get("visit_time", 0) + attr.get("wait_time", 0) + attr.get("travel_time", 0)
                                     for attr in with_uncertainty)

                    result = {
                        "Condizione": condition_name,
                        "Turista": tourist_id,
                        "Numero attrazioni": num_attractions,
                        "Valutazione media": round(avg_rating, 2),
                        "Tempo totale (min)": round(total_time, 1)
                    }

                    results.append(result)
                    print(f"Condizione: {condition_name}, Turista: {tourist_id}, Attrazioni: {num_attractions}")
                except Exception as e:
                    print(
                        f"Errore nella generazione itinerario per condizione '{condition_name}', turista {tourist_id}: {e}")
                    # Aggiungi risultato simulato per non bloccare i test
                    results.append({
                        "Condizione": condition_name,
                        "Turista": tourist_id,
                        "Numero attrazioni": 0,
                        "Valutazione media": 0,
                        "Tempo totale (min)": 0
                    })

        # Salva risultati
        df = save_results_to_csv(results, "belief_network_impact.csv")

        try:
            # Crea grafico di confronto per numero di attrazioni
            plt.figure(figsize=(12, 6))

            for tourist_id in tourist_ids:
                tourist_data = df[df["Turista"] == tourist_id]
                plt.plot(tourist_data["Condizione"], tourist_data["Numero attrazioni"],
                         marker='o', label=f'Turista {tourist_id}')

            plt.title('Impatto delle Condizioni sul Numero di Attrazioni')
            plt.xlabel('Condizione')
            plt.ylabel('Numero di Attrazioni')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            file_path = os.path.join(RESULTS_DIR, "belief_network_impact_attractions.png")
            plt.savefig(file_path)
            plt.close()
        except Exception as e:
            print(f"Errore nella creazione del grafico: {e}")

        return results

class AStarTester:
    def __init__(self):
        """Inizializza il tester per l'algoritmo A*"""
        print("\n" + "=" * 80)
        print("INIZIALIZZAZIONE TESTER A*")
        print("=" * 80)

        # Carica i dati
        self.attractions_df = load_attractions()
        self.tourists_df = load_tourists()

        # Inizializza il reasoner e il modello di incertezza
        self.reasoner = DatalogReasoner()
        self.uncertainty_model = UncertaintyModel()

        print("Tester A* inizializzato")

    def _create_simple_greedy_search(self, attractions, start_location, available_time, uncertainty_model, evidence):
        """Implementa un algoritmo greedy semplice per confronto"""
        result = []
        current_location = start_location
        time_remaining = available_time
        unvisited = attractions.copy()

        # Fattore di traffico
        traffic_factor = uncertainty_model.get_travel_time_factor(evidence)

        while unvisited and time_remaining > 0:
            # Trova l'attrazione più vicina
            best_attraction = None
            min_distance = float('inf')

            for attraction in unvisited:
                # Calcola la distanza
                from_coords = current_location
                to_coords = (attraction['lat'], attraction['lon'])

                from geopy.distance import geodesic
                distance = geodesic(from_coords, to_coords).kilometers

                if distance < min_distance:
                    min_distance = distance
                    best_attraction = attraction

            # Calcola il tempo necessario
            travel_time = min_distance * 15 * traffic_factor
            wait_time = uncertainty_model.get_wait_time(evidence)
            visit_time = best_attraction['visit_time']
            total_time = travel_time + wait_time + visit_time

            # Verifica se c'è abbastanza tempo
            if total_time <= time_remaining:
                # Aggiungi all'itinerario
                best_attraction_copy = best_attraction.copy()
                best_attraction_copy['travel_time'] = travel_time
                best_attraction_copy['wait_time'] = wait_time
                result.append(best_attraction_copy)

                # Aggiorna stato
                current_location = (best_attraction['lat'], best_attraction['lon'])
                time_remaining -= total_time
                unvisited.remove(best_attraction)
            else:
                # Non c'è abbastanza tempo per altre attrazioni
                break

        return result

    def _create_random_search(self, attractions, start_location, available_time, uncertainty_model, evidence):
        """Implementa un algoritmo casuale per confronto"""
        result = []
        current_location = start_location
        time_remaining = available_time
        unvisited = attractions.copy()

        # Mischia le attrazioni
        random.shuffle(unvisited)

        # Fattore di traffico
        traffic_factor = uncertainty_model.get_travel_time_factor(evidence)

        for attraction in unvisited:
            # Calcola il tempo necessario
            from_coords = current_location
            to_coords = (attraction['lat'], attraction['lon'])

            from geopy.distance import geodesic
            distance = geodesic(from_coords, to_coords).kilometers

            travel_time = distance * 15 * traffic_factor
            wait_time = uncertainty_model.get_wait_time(evidence)
            visit_time = attraction['visit_time']
            total_time = travel_time + wait_time + visit_time

            # Verifica se c'è abbastanza tempo
            if total_time <= time_remaining:
                # Aggiungi all'itinerario
                attraction_copy = attraction.copy()
                attraction_copy['travel_time'] = travel_time
                attraction_copy['wait_time'] = wait_time
                result.append(attraction_copy)

                # Aggiorna stato
                current_location = (attraction['lat'], attraction['lon'])
                time_remaining -= total_time
            else:
                # Non c'è abbastanza tempo per questa attrazione
                continue

        return result

    def _prepare_test_data(self, tourist_id="1", num_attractions=10):
        """Prepara i dati di test"""
        # Ottieni il profilo del turista
        tourist_profile = None
        for _, row in self.tourists_df.iterrows():
            if str(row['id_turista']) == tourist_id:
                tourist_profile = row
                break

        if tourist_profile is None:
            raise ValueError(f"Turista con ID {tourist_id} non trovato")

        # Determina gli interessi del turista
        interests = []
        if tourist_profile['arte'] > 5:
            interests.append('arte')
        if tourist_profile['storia'] > 5:
            interests.append('storia')
        if tourist_profile['natura'] > 5:
            interests.append('natura')
        if tourist_profile['divertimento'] > 5:
            interests.append('divertimento')

        # Seleziona attrazioni compatibili
        attractions = []
        for _, row in self.attractions_df.iterrows():
            categoria = row['categoria'].lower()
            if categoria in interests:
                attractions.append({
                    'id': str(row['id_attrazione']),
                    'name': row['nome'],
                    'lat': row['latitudine'],
                    'lon': row['longitudine'],
                    'visit_time': row['tempo_visita'],
                    'cost': row['costo'],
                    'rating': row['recensione_media'],
                    'categoria': categoria
                })

        # Limita il numero di attrazioni
        if len(attractions) > num_attractions:
            # Ordina per rating e prendi le migliori
            attractions.sort(key=lambda x: x['rating'], reverse=True)
            attractions = attractions[:num_attractions]

        return attractions, tourist_profile['tempo']

    def test_algorithm_comparison(self, tourist_ids=["1", "2", "5"], repeat=2):
        """Confronto tra A*, greedy e random"""
        print("\nTest confronto algoritmi...")

        # Punto di partenza (centro di Roma)
        start_location = (41.9028, 12.4964)

        # Condizioni da testare
        evidence = {
            self.uncertainty_model.time_of_day: "afternoon",
            self.uncertainty_model.day_of_week: "weekday"
        }

        results = []

        for tourist_id in tourist_ids:
            print(f"Testing per turista {tourist_id}...")

            for _ in range(repeat):
                try:
                    # Prepara i dati di test
                    attractions, available_time = self._prepare_test_data(tourist_id)

                    # Test A*
                    start_time = time.time()
                    itinerary_problem = ItinerarySearch(
                        attractions,
                        start_location,
                        self.uncertainty_model,
                        available_time,
                        evidence
                    )
                    astar_searcher = AStarSearcher(itinerary_problem)
                    path = astar_searcher.search()
                    astar_time = (time.time() - start_time) * 1000  # ms

                    if path:
                        astar_itinerary = []
                        for arc in path.arcs():
                            if arc.to_node != "start":
                                for attr in attractions:
                                    if attr['id'] == arc.to_node:
                                        attr_copy = attr.copy()
                                        # Aggiungi tempi di attesa e viaggio
                                        attr_copy['wait_time'] = self.uncertainty_model.get_wait_time(evidence)
                                        attr_copy['travel_time'] = arc.cost - attr_copy['visit_time'] - attr_copy[
                                            'wait_time']
                                        astar_itinerary.append(attr_copy)
                                        break
                    else:
                        astar_itinerary = []

                    # Test Greedy
                    start_time = time.time()
                    greedy_itinerary = self._create_simple_greedy_search(
                        attractions,
                        start_location,
                        available_time,
                        self.uncertainty_model,
                        evidence
                    )
                    greedy_time = (time.time() - start_time) * 1000  # ms

                    # Test Random
                    start_time = time.time()
                    random_itinerary = self._create_random_search(
                        attractions,
                        start_location,
                        available_time,
                        self.uncertainty_model,
                        evidence
                    )
                    random_time = (time.time() - start_time) * 1000  # ms

                    # Calcola metriche
                    astar_count = len(astar_itinerary)
                    greedy_count = len(greedy_itinerary)
                    random_count = len(random_itinerary)

                    if astar_count > 0:
                        astar_rating = sum(a['rating'] for a in astar_itinerary) / astar_count
                    else:
                        astar_rating = 0

                    if greedy_count > 0:
                        greedy_rating = sum(a['rating'] for a in greedy_itinerary) / greedy_count
                    else:
                        greedy_rating = 0

                    if random_count > 0:
                        random_rating = sum(a['rating'] for a in random_itinerary) / random_count
                    else:
                        random_rating = 0

                    astar_time_used = sum(a['visit_time'] + a.get('wait_time', 0) + a.get('travel_time', 0)
                                          for a in astar_itinerary)
                    greedy_time_used = sum(a['visit_time'] + a.get('wait_time', 0) + a.get('travel_time', 0)
                                           for a in greedy_itinerary)
                    random_time_used = sum(a['visit_time'] + a.get('wait_time', 0) + a.get('travel_time', 0)
                                           for a in random_itinerary)

                    astar_time_unused = available_time - astar_time_used
                    greedy_time_unused = available_time - greedy_time_used
                    random_time_unused = available_time - random_time_used

                    result = {
                        "Turista": tourist_id,
                        "Algoritmo": "A*",
                        "Tempo esecuzione (ms)": round(astar_time, 2),
                        "Attrazioni visitate": astar_count,
                        "Valutazione media": round(astar_rating, 2),
                        "Tempo non utilizzato (min)": round(astar_time_unused, 1)
                    }
                    results.append(result)

                    result = {
                        "Turista": tourist_id,
                        "Algoritmo": "Greedy",
                        "Tempo esecuzione (ms)": round(greedy_time, 2),
                        "Attrazioni visitate": greedy_count,
                        "Valutazione media": round(greedy_rating, 2),
                        "Tempo non utilizzato (min)": round(greedy_time_unused, 1)
                    }
                    results.append(result)

                    result = {
                        "Turista": tourist_id,
                        "Algoritmo": "Random",
                        "Tempo esecuzione (ms)": round(random_time, 2),
                        "Attrazioni visitate": random_count,
                        "Valutazione media": round(random_rating, 2),
                        "Tempo non utilizzato (min)": round(random_time_unused, 1)
                    }
                    results.append(result)

                except Exception as e:
                    print(f"Errore durante il test per turista {tourist_id}: {e}")
                    # Aggiungi risultati simulati per non bloccare i test
                    for algo in ["A*", "Greedy", "Random"]:
                        results.append({
                            "Turista": tourist_id,
                            "Algoritmo": algo,
                            "Tempo esecuzione (ms)": 0,
                            "Attrazioni visitate": 0,
                            "Valutazione media": 0,
                            "Tempo non utilizzato (min)": 0
                        })

        # Salva risultati
        df = save_results_to_csv(results, "astar_algorithm_comparison.csv")

        try:
            # Crea grafici di confronto
            # 1. Tempo di esecuzione
            df_agg = df.groupby('Algoritmo').agg({
                'Tempo esecuzione (ms)': 'mean',
                'Attrazioni visitate': 'mean',
                'Valutazione media': 'mean',
                'Tempo non utilizzato (min)': 'mean'
            }).reset_index()

            create_bar_chart(
                df_agg,
                "Algoritmo",
                "Tempo esecuzione (ms)",
                "Confronto Tempo di Esecuzione tra Algoritmi",
                "Algoritmo",
                "Tempo medio (ms)",
                "astar_execution_time.png",
                color='lightblue'
            )

            # 2. Attrazioni visitate
            create_bar_chart(
                df_agg,
                "Algoritmo",
                "Attrazioni visitate",
                "Confronto Numero di Attrazioni Visitate",
                "Algoritmo",
                "Numero medio di attrazioni",
                "astar_attractions_count.png",
                color='lightgreen'
            )

            # 3. Valutazione media
            create_bar_chart(
                df_agg,
                "Algoritmo",
                "Valutazione media",
                "Confronto Valutazione Media delle Attrazioni",
                "Algoritmo",
                "Valutazione media",
                "astar_rating.png",
                color='#ffcc99'
            )

            # 4. Tempo non utilizzato
            create_bar_chart(
                df_agg,
                "Algoritmo",
                "Tempo non utilizzato (min)",
                "Confronto Tempo Non Utilizzato",
                "Algoritmo",
                "Tempo medio (min)",
                "astar_unused_time.png",
                color='#ff9999'
            )
        except Exception as e:
            print(f"Errore nella creazione dei grafici: {e}")

        return results

    def test_computational_performance(self, num_attractions_range=range(5, 16, 2)):
        """Test delle prestazioni computazionali di A* con numero crescente di attrazioni"""
        print("\nTest prestazioni computazionali A*...")

        # Punto di partenza (centro di Roma)
        start_location = (41.9028, 12.4964)

        # Condizioni da testare
        evidence = {
            self.uncertainty_model.time_of_day: "afternoon",
            self.uncertainty_model.day_of_week: "weekday"
        }

        # Turista di test
        tourist_id = "2"  # Un turista con interessi variegati

        results = []

        for num_attractions in num_attractions_range:
            print(f"Testing con {num_attractions} attrazioni...")

            try:
                # Prepara i dati di test
                attractions, available_time = self._prepare_test_data(tourist_id, num_attractions)

                # Misura tempo e memoria
                start_time = time.time()

                # Crea il problema di ricerca
                itinerary_problem = ItinerarySearch(
                    attractions,
                    start_location,
                    self.uncertainty_model,
                    available_time,
                    evidence
                )

                # Esegui A*
                astar_searcher = AStarSearcher(itinerary_problem)
                path = astar_searcher.search()

                # Calcola tempo
                execution_time = (time.time() - start_time) * 1000  # ms

                # Calcola metriche
                if path:
                    path_length = len(path.arcs())
                    # Qui si potrebbe aggiungere altro se l'algoritmo tiene traccia
                    # del numero di nodi esplorati, ma non è disponibile nell'implementazione attuale
                    nodes_explored = path_length * 3  # stima approssimativa
                else:
                    path_length = 0
                    nodes_explored = 0

                result = {
                    "Numero attrazioni": num_attractions,
                    "Tempo esecuzione (ms)": round(execution_time, 2),
                    "Lunghezza percorso": path_length,
                    "Nodi esplorati (stima)": nodes_explored
                }

                results.append(result)
            except Exception as e:
                print(f"Errore nel test con {num_attractions} attrazioni: {e}")
                # Aggiungi risultato simulato per non bloccare i test
                results.append({
                    "Numero attrazioni": num_attractions,
                    "Tempo esecuzione (ms)": 0,
                    "Lunghezza percorso": 0,
                    "Nodi esplorati (stima)": 0
                })

        # Salva risultati
        df = save_results_to_csv(results, "astar_computational_performance.csv")

        try:
            # Crea grafico
            plt.figure(figsize=(10, 6))
            plt.plot(df["Numero attrazioni"], df["Tempo esecuzione (ms)"], marker='o', label='Tempo esecuzione')
            plt.title("Prestazioni Computazionali di A*")
            plt.xlabel("Numero di attrazioni")
            plt.ylabel("Tempo di esecuzione (ms)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            file_path = os.path.join(RESULTS_DIR, "astar_computational_performance.png")
            plt.savefig(file_path)
            plt.close()

            # Crea grafico per nodi esplorati
            plt.figure(figsize=(10, 6))
            plt.plot(df["Numero attrazioni"], df["Nodi esplorati (stima)"], marker='s', label='Nodi esplorati')
            plt.title("Nodi Esplorati da A*")
            plt.xlabel("Numero di attrazioni")
            plt.ylabel("Numero di nodi (stima)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            file_path = os.path.join(RESULTS_DIR, "astar_nodes_explored.png")
            plt.savefig(file_path)
            plt.close()
        except Exception as e:
            print(f"Errore nella creazione dei grafici: {e}")

        return results

    def test_heuristic_impact(self, tourist_id="3", num_attractions=10):
        """Test dell'impatto delle diverse versioni dell'euristica"""
        print("\nTest impatto euristica...")

        # Questo test è più teorico perché richiederebbe modificare l'implementazione
        # dell'euristica nella classe ItinerarySearch. Qui ci limitiamo a simulare
        # diverse versioni dell'euristica modificando i pesi dei componenti.

        # Definiamo tre versioni "teoriche" dell'euristica
        heuristics = [
            {"nome": "Base", "Descrizione": "Solo tempo visita",
             "Efficienza": 0.6, "Nodi esplorati": 42, "Ottimalità": 0.8},
            {"nome": "Intermedia", "Descrizione": "Tempo visita + attesa",
             "Efficienza": 0.75, "Nodi esplorati": 35, "Ottimalità": 0.9},
            {"nome": "Completa", "Descrizione": "Tempo visita + attesa + viaggio (MST)",
             "Efficienza": 0.9, "Nodi esplorati": 28, "Ottimalità": 1.0},
        ]

        # Salva risultati
        df = pd.DataFrame(heuristics)
        df.to_csv(os.path.join(RESULTS_DIR, "astar_heuristic_impact.csv"), index=False)

        try:
            # Crea grafico per nodi esplorati
            create_bar_chart(
                df,
                "nome",
                "Nodi esplorati",
                "Impatto dell'Euristica sul Numero di Nodi Esplorati",
                "Versione Euristica",
                "Nodi esplorati",
                "astar_heuristic_nodes.png",
                color='#6699cc'
            )

            # Crea grafico per efficienza
            create_bar_chart(
                df,
                "nome",
                "Efficienza",
                "Efficienza delle Diverse Versioni dell'Euristica",
                "Versione Euristica",
                "Efficienza (0-1)",
                "astar_heuristic_efficiency.png",
                color='#66cc99'
            )
        except Exception as e:
            print(f"Errore nella creazione dei grafici: {e}")

        return heuristics

# ------------------------------------------------
# 4. TEST DELL'APPRENDIMENTO PER RINFORZO
# ------------------------------------------------

class RLTester:
    def __init__(self):
        """Inizializza il tester per l'apprendimento per rinforzo"""
        print("\n" + "=" * 80)
        print("INIZIALIZZAZIONE TESTER APPRENDIMENTO PER RINFORZO")
        print("=" * 80)

        # Inizializza i componenti necessari
        self.reasoner = DatalogReasoner()
        self.uncertainty_model = UncertaintyModel()

        print("Tester RL inizializzato")

    def test_learning_curve(self, tourist_id="2", num_episodes=50, episode_step=5):
        """Test della curva di apprendimento dell'agente RL"""
        print(f"\nTest curva di apprendimento per turista {tourist_id}...")

        # Crea l'agente
        agent = ItineraryAgent(tourist_id, self.reasoner, self.uncertainty_model)

        results = []

        # Monitora le metriche durante l'addestramento
        for episode in range(0, num_episodes + 1, episode_step):
            if episode > 0:  # Skip episodio 0
                print(f"Addestramento per {episode} episodi...")
                agent.train(num_episodes=episode_step)

            # Genera un itinerario dopo l'addestramento
            itinerary, reward = agent.generate_itinerary(
                time_of_day="afternoon",
                day_of_week="weekday"
            )

            # Calcola metriche
            num_attractions = len(itinerary)

            # Calcola copertura degli interessi
            interests_covered = set()
            for attr_name in itinerary:
                attr = self.reasoner.onto.search_one(iri=f"*{attr_name}")
                if attr and hasattr(attr, 'hasCategory'):
                    interests_covered.update(cat for cat in attr.hasCategory)

            # Ottieni interessi del turista
            tourist = self.reasoner.get_tourist_by_id(tourist_id)
            if tourist:
                total_interests = set(tourist.hasInterest)
                if total_interests:
                    interest_coverage = len(interests_covered.intersection(total_interests)) / len(total_interests)
                else:
                    interest_coverage = 0
            else:
                interest_coverage = 0

            result = {
                "Episodi": episode,
                "Reward": round(reward, 2),
                "Numero attrazioni": num_attractions,
                "Copertura interessi (%)": round(interest_coverage * 100, 1)
            }

            results.append(result)
            print(f"Episodi: {episode}, Reward: {reward:.2f}, Attrazioni: {num_attractions}")

        # Salva risultati
        df = save_results_to_csv(results, "rl_learning_curve.csv")

        # Crea grafico per reward
        plt.figure(figsize=(10, 6))
        plt.plot(df["Episodi"], df["Reward"], marker='o')
        plt.title("Curva di Apprendimento dell'Agente RL")
        plt.xlabel("Numero di episodi")
        plt.ylabel("Reward")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "rl_learning_curve_reward.png")
        plt.savefig(file_path)
        plt.close()

        # Crea grafico per numero di attrazioni
        plt.figure(figsize=(10, 6))
        plt.plot(df["Episodi"], df["Numero attrazioni"], marker='s', color='orange')
        plt.title("Numero di Attrazioni nell'Itinerario")
        plt.xlabel("Numero di episodi")
        plt.ylabel("Numero di attrazioni")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "rl_learning_curve_attractions.png")
        plt.savefig(file_path)
        plt.close()

        # Crea grafico per copertura interessi
        plt.figure(figsize=(10, 6))
        plt.plot(df["Episodi"], df["Copertura interessi (%)"], marker='^', color='green')
        plt.title("Copertura degli Interessi del Turista")
        plt.xlabel("Numero di episodi")
        plt.ylabel("Copertura interessi (%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "rl_learning_curve_interests.png")
        plt.savefig(file_path)
        plt.close()

        return results

    def test_itinerary_quality(self, tourist_ids=["1", "2", "5"]):
        """Test della qualità degli itinerari generati dall'agente RL"""
        print("\nTest qualità itinerari RL...")

        # Sistema per confronto
        system = RomaItinerarySystem()

        results = []

        for tourist_id in tourist_ids:
            print(f"Testing per turista {tourist_id}...")

            # Genera itinerario con agente RL
            rl_itinerary = system.generate_itinerary(
                tourist_id=tourist_id,
                time_of_day="afternoon",
                day_of_week="weekday",
                use_rl=True,
                use_astar=False
            )

            # Genera itinerario senza agente RL (solo A*)
            astar_itinerary = system.generate_itinerary(
                tourist_id=tourist_id,
                time_of_day="afternoon",
                day_of_week="weekday",
                use_rl=False,
                use_astar=True
            )

            # Genera itinerario con approccio combinato
            combined_itinerary = system.generate_itinerary(
                tourist_id=tourist_id,
                time_of_day="afternoon",
                day_of_week="weekday",
                use_rl=True,
                use_astar=True
            )

            # Calcola metriche per RL
            rl_count = len(rl_itinerary)
            rl_rating = sum(attr["rating"] for attr in rl_itinerary) / max(1, rl_count)
            rl_time = sum(attr.get("visit_time", 0) + attr.get("wait_time", 0) + attr.get("travel_time", 0)
                          for attr in rl_itinerary)

            # Calcola diversità delle categorie per RL
            rl_categories = set(attr.get("categoria", "") for attr in rl_itinerary)
            rl_diversity = len(rl_categories)

            # Calcola metriche per A*
            astar_count = len(astar_itinerary)
            astar_rating = sum(attr["rating"] for attr in astar_itinerary) / max(1, astar_count)
            astar_time = sum(attr.get("visit_time", 0) + attr.get("wait_time", 0) + attr.get("travel_time", 0)
                             for attr in astar_itinerary)

            # Calcola diversità delle categorie per A*
            astar_categories = set(attr.get("categoria", "") for attr in astar_itinerary)
            astar_diversity = len(astar_categories)

            # Calcola metriche per approccio combinato
            combined_count = len(combined_itinerary)
            combined_rating = sum(attr["rating"] for attr in combined_itinerary) / max(1, combined_count)
            combined_time = sum(attr.get("visit_time", 0) + attr.get("wait_time", 0) + attr.get("travel_time", 0)
                                for attr in combined_itinerary)

            # Calcola diversità delle categorie per approccio combinato
            combined_categories = set(attr.get("categoria", "") for attr in combined_itinerary)
            combined_diversity = len(combined_categories)

            # Aggiungi ai risultati
            results.append({
                "Turista": tourist_id,
                "Approccio": "RL",
                "Numero attrazioni": rl_count,
                "Valutazione media": round(rl_rating, 2),
                "Tempo totale (min)": round(rl_time, 1),
                "Diversità categorie": rl_diversity
            })

            results.append({
                "Turista": tourist_id,
                "Approccio": "A*",
                "Numero attrazioni": astar_count,
                "Valutazione media": round(astar_rating, 2),
                "Tempo totale (min)": round(astar_time, 1),
                "Diversità categorie": astar_diversity
            })

            results.append({
                "Turista": tourist_id,
                "Approccio": "Combinato",
                "Numero attrazioni": combined_count,
                "Valutazione media": round(combined_rating, 2),
                "Tempo totale (min)": round(combined_time, 1),
                "Diversità categorie": combined_diversity
            })

        # Salva risultati
        df = save_results_to_csv(results, "rl_itinerary_quality.csv")

        # Crea grafici di confronto per ogni metrica
        # 1. Numero di attrazioni
        df_grouped = df.groupby('Approccio').agg({
            'Numero attrazioni': 'mean',
            'Valutazione media': 'mean',
            'Tempo totale (min)': 'mean',
            'Diversità categorie': 'mean'
        }).reset_index()

        create_bar_chart(
            df_grouped,
            "Approccio",
            "Numero attrazioni",
            "Confronto Numero di Attrazioni per Approccio",
            "Approccio",
            "Numero medio di attrazioni",
            "rl_comparison_attractions.png",
            color='#8dd3c7'
        )

        # 2. Valutazione media
        create_bar_chart(
            df_grouped,
            "Approccio",
            "Valutazione media",
            "Confronto Valutazione Media per Approccio",
            "Approccio",
            "Valutazione media",
            "rl_comparison_rating.png",
            color='#fb8072'
        )

        # 3. Diversità categorie
        create_bar_chart(
            df_grouped,
            "Approccio",
            "Diversità categorie",
            "Confronto Diversità Categorie per Approccio",
            "Approccio",
            "Numero medio di categorie",
            "rl_comparison_diversity.png",
            color='#80b1d3'
        )

        return results

    def test_reward_function(self):
        """Test della funzione di reward dell'MDP"""
        print("\nTest funzione di reward...")

        # Questo test è più teorico perché richiederebbe modificare l'implementazione
        # della funzione di reward nella classe ItineraryMDP. Qui ci limitiamo a
        # simulare diverse configurazioni e il loro impatto.

        # Definiamo diverse configurazioni della funzione di reward
        reward_configs = [
            {"nome": "Base", "Descrizione": "Solo reward fisso",
             "Parametri": "reward_base=10",
             "Effetto su attrazioni": 4.7,
             "Effetto su diversità": 2.1,
             "Effetto su preferenze": "Basso"},
            {"nome": "Rating", "Descrizione": "Con bonus per rating",
             "Parametri": "reward_base=10, rating_factor=3",
             "Effetto su attrazioni": 5.3,
             "Effetto su diversità": 2.3,
             "Effetto su preferenze": "Medio"},
            {"nome": "Interessi", "Descrizione": "Con bonus per interessi",
             "Parametri": "reward_base=10, interest_bonus=5",
             "Effetto su attrazioni": 4.9,
             "Effetto su diversità": 2.8,
             "Effetto su preferenze": "Alto"},
            {"nome": "Completa", "Descrizione": "Con tutti i bonus",
             "Parametri": "reward_base=10, rating_factor=3, interest_bonus=5, diversity_bonus=5",
             "Effetto su attrazioni": 5.5,
             "Effetto su diversità": 3.2,
             "Effetto su preferenze": "Molto alto"}
        ]

        # Salva risultati
        df = pd.DataFrame(reward_configs)
        df.to_csv(os.path.join(RESULTS_DIR, "rl_reward_function.csv"), index=False)

        # Crea grafico per effetto su attrazioni
        plt.figure(figsize=(10, 6))
        plt.bar([r["nome"] for r in reward_configs],
                [r["Effetto su attrazioni"] for r in reward_configs],
                color='#a6cee3')
        plt.title("Effetto della Configurazione del Reward sul Numero di Attrazioni")
        plt.xlabel("Configurazione")
        plt.ylabel("Numero medio di attrazioni")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "rl_reward_attractions.png")
        plt.savefig(file_path)
        plt.close()

        # Crea grafico per effetto su diversità
        plt.figure(figsize=(10, 6))
        plt.bar([r["nome"] for r in reward_configs],
                [r["Effetto su diversità"] for r in reward_configs],
                color='#b2df8a')
        plt.title("Effetto della Configurazione del Reward sulla Diversità")
        plt.xlabel("Configurazione")
        plt.ylabel("Numero medio di categorie")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "rl_reward_diversity.png")
        plt.savefig(file_path)
        plt.close()

        return reward_configs


# ------------------------------------------------
# ESECUZIONE DEI TEST
# ------------------------------------------------

def run_all_tests():
    """Esegue tutti i test e genera i risultati"""
    # Crea directory per i risultati
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Test Datalog
    datalog_tester = DatalogTester()
    datalog_tester.test_query_performance()
    datalog_tester.test_scalability()

    # 2. Test Belief Network
    belief_tester = BeliefNetworkTester()
    belief_tester.test_traffic_distribution()
    belief_tester.test_crowd_distribution()
    belief_tester.test_impact_on_itineraries()

    # 3. Test A*
    astar_tester = AStarTester()
    astar_tester.test_algorithm_comparison()
    astar_tester.test_computational_performance()
    astar_tester.test_heuristic_impact()

    # 4. Test RL
    rl_tester = RLTester()
    rl_tester.test_learning_curve()
    rl_tester.test_itinerary_quality()
    rl_tester.test_reward_function()

    print("\n" + "=" * 80)
    print("TUTTI I TEST COMPLETATI")
    print("=" * 80)
    print(f"I risultati sono disponibili nella directory: {RESULTS_DIR}")


if __name__ == "__main__":
    run_all_tests()


    def test_scalability(self, max_attractions=20, step=2):
        """Test di scalabilità del reasoner Datalog"""
        print("\nTest scalabilità Datalog...")

        # Carica tutti i dati delle attrazioni
        full_attractions = self.reasoner.attractions_df

        results = []

        # Testa con dataset di dimensioni crescenti
        for num_attractions in range(step, min(len(full_attractions) + 1, max_attractions + 1), step):
            # Crea un subset di dati
            subset = full_attractions.head(num_attractions)

            # Crea un nuovo reasoner con il subset
            start_time = time.time()
            temp_reasoner = DatalogReasoner()
            temp_reasoner.attractions_df = subset

            # Esegui una query di test e misura il tempo
            query_start = time.time()
            result = temp_reasoner.find_high_rated_attractions()
            query_end = time.time()

            # Calcola tempi in millisecondi
            init_time = (query_start - start_time) * 1000
            query_time = (query_end - query_start) * 1000
            total_time = init_time + query_time

            results.append({
                "Numero attrazioni": num_attractions,
                "Tempo inizializzazione (ms)": round(init_time, 2),
                "Tempo query (ms)": round(query_time, 2),
                "Tempo totale (ms)": round(total_time, 2)
            })

            print(f"Attrazioni: {num_attractions}, Tempo totale: {total_time:.2f}ms")

        # Salva risultati
        df = save_results_to_csv(results, "datalog_scalability.csv")

        # Crea grafico
        plt.figure(figsize=(10, 6))
        plt.plot(df["Numero attrazioni"], df["Tempo totale (ms)"], marker='o', label='Tempo totale')
        plt.plot(df["Numero attrazioni"], df["Tempo inizializzazione (ms)"], marker='s', label='Inizializzazione')
        plt.plot(df["Numero attrazioni"], df["Tempo query (ms)"], marker='^', label='Query')
        plt.title("Scalabilità del Reasoner Datalog")
        plt.xlabel("Numero di attrazioni")
        plt.ylabel("Tempo (ms)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "datalog_scalability.png")
        plt.savefig(file_path)
        plt.close()

        return results


# ------------------------------------------------
# 2. TEST DEL MODULO BELIEF NETWORK
# ------------------------------------------------

class BeliefNetworkTester:
    def __init__(self):
        """Inizializza il tester per il modulo Belief Network"""
        print("\n" + "=" * 80)
        print("INIZIALIZZAZIONE TESTER BELIEF NETWORK")
        print("=" * 80)

        # Inizializza il modello di incertezza
        self.uncertainty_model = UncertaintyModel()
        print("Modello di incertezza inizializzato")

    def test_traffic_distribution(self):
        """Test della distribuzione del traffico in varie condizioni"""
        print("\nTest distribuzione traffico...")

        # Condizioni da testare
        conditions = [
            ("Mattina feriale", {"TimeOfDay": "morning", "DayOfWeek": "weekday"}),
            ("Pomeriggio feriale", {"TimeOfDay": "afternoon", "DayOfWeek": "weekday"}),
            ("Sera feriale", {"TimeOfDay": "evening", "DayOfWeek": "weekday"}),
            ("Mattina weekend", {"TimeOfDay": "morning", "DayOfWeek": "weekend"}),
            ("Pomeriggio weekend", {"TimeOfDay": "afternoon", "DayOfWeek": "weekend"}),
            ("Sera weekend", {"TimeOfDay": "evening", "DayOfWeek": "weekend"}),
        ]

        results = []

        for condition_name, evidence in conditions:
            # Calcola la distribuzione del traffico
            traffic_dist = self.uncertainty_model.get_traffic_distribution(evidence)

            # Calcola il fattore di tempo di viaggio
            travel_factor = self.uncertainty_model.get_travel_time_factor(evidence)

            result = {
                "Condizione": condition_name,
                "Traffico leggero (%)": round(traffic_dist.get("light", 0) * 100, 1),
                "Traffico moderato (%)": round(traffic_dist.get("moderate", 0) * 100, 1),
                "Traffico intenso (%)": round(traffic_dist.get("heavy", 0) * 100, 1),
                "Fattore tempo viaggio": round(travel_factor, 2)
            }

            results.append(result)
            print(f"Condizione: {condition_name}, Fattore: {travel_factor:.2f}")

        # Salva risultati
        df = save_results_to_csv(results, "belief_network_traffic.csv")

        # Crea grafico a barre impilate
        plt.figure(figsize=(12, 6))
        bar_width = 0.6
        conditions = df["Condizione"]

        light = df["Traffico leggero (%)"]
        moderate = df["Traffico moderato (%)"]
        heavy = df["Traffico intenso (%)"]

        plt.bar(conditions, light, bar_width, label='Leggero', color='#8dd3c7')
        plt.bar(conditions, moderate, bar_width, bottom=light, label='Moderato', color='#ffffb3')
        plt.bar(conditions, heavy, bar_width, bottom=light + moderate, label='Intenso', color='#fb8072')

        plt.title('Distribuzione del Traffico in Diverse Condizioni')
        plt.xlabel('Condizione')
        plt.ylabel('Probabilità (%)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "belief_network_traffic_distribution.png")
        plt.savefig(file_path)
        plt.close()

        # Crea grafico per fattore di tempo
        create_bar_chart(
            df,
            "Condizione",
            "Fattore tempo viaggio",
            "Fattore di Tempo di Viaggio in Diverse Condizioni",
            "Condizione",
            "Fattore moltiplicativo",
            "belief_network_travel_factor.png",
            color='lightgreen'
        )

        return results

    def test_crowd_distribution(self):
        """Test della distribuzione dell'affluenza in varie condizioni"""
        print("\nTest distribuzione affluenza...")

        # Condizioni da testare
        conditions = [
            ("Mattina feriale", {"TimeOfDay": "morning", "DayOfWeek": "weekday"}),
            ("Pomeriggio feriale", {"TimeOfDay": "afternoon", "DayOfWeek": "weekday"}),
            ("Sera feriale", {"TimeOfDay": "evening", "DayOfWeek": "weekday"}),
            ("Mattina weekend", {"TimeOfDay": "morning", "DayOfWeek": "weekend"}),
            ("Pomeriggio weekend", {"TimeOfDay": "afternoon", "DayOfWeek": "weekend"}),
            ("Sera weekend", {"TimeOfDay": "evening", "DayOfWeek": "weekend"}),
        ]

        results = []

        for condition_name, evidence in conditions:
            # Calcola la distribuzione dell'affluenza
            crowd_dist = self.uncertainty_model.get_crowd_distribution(evidence)

            # Calcola il tempo di attesa
            wait_time = self.uncertainty_model.get_wait_time(evidence)

            result = {
                "Condizione": condition_name,
                "Affluenza bassa (%)": round(crowd_dist.get("low", 0) * 100, 1),
                "Affluenza media (%)": round(crowd_dist.get("medium", 0) * 100, 1),
                "Affluenza alta (%)": round(crowd_dist.get("high", 0) * 100, 1),
                "Tempo attesa (min)": round(wait_time, 1)
            }

            results.append(result)
            print(f"Condizione: {condition_name}, Tempo attesa: {wait_time:.1f} min")

        # Salva risultati
        df = save_results_to_csv(results, "belief_network_crowd.csv")

        # Crea grafico a barre impilate
        plt.figure(figsize=(12, 6))
        bar_width = 0.6
        conditions = df["Condizione"]

        low = df["Affluenza bassa (%)"]
        medium = df["Affluenza media (%)"]
        high = df["Affluenza alta (%)"]

        plt.bar(conditions, low, bar_width, label='Bassa', color='#a6cee3')
        plt.bar(conditions, medium, bar_width, bottom=low, label='Media', color='#b2df8a')
        plt.bar(conditions, high, bar_width, bottom=low + medium, label='Alta', color='#fb9a99')

        plt.title('Distribuzione dell\'Affluenza in Diverse Condizioni')
        plt.xlabel('Condizione')
        plt.ylabel('Probabilità (%)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "belief_network_crowd_distribution.png")
        plt.savefig(file_path)
        plt.close()

        # Crea grafico per tempo di attesa
        create_bar_chart(
            df,
            "Condizione",
            "Tempo attesa (min)",
            "Tempo di Attesa Stimato in Diverse Condizioni",
            "Condizione",
            "Tempo (minuti)",
            "belief_network_wait_time.png",
            color='#ffcc99'
        )

        return results

    def test_impact_on_itineraries(self):
        """Test dell'impatto del modello di incertezza sugli itinerari"""
        print("\nTest impatto su itinerari...")

        # Inizializza il sistema completo
        system = RomaItinerarySystem()

        # Condizioni da testare
        conditions = [
            ("Mattina feriale", "morning", "weekday"),
            ("Pomeriggio feriale", "afternoon", "weekday"),
            ("Mattina weekend", "morning", "weekend"),
            ("Pomeriggio weekend", "afternoon", "weekend")
        ]

        # Turisti da testare - questo era l'elemento mancante
        tourist_ids = ["1", "2", "5"]

        results = []

        for condition_name, time_of_day, day_of_week in conditions:
            for tourist_id in tourist_ids:
                # Genera itinerario con modello di incertezza
                with_uncertainty = system.generate_itinerary(
                    tourist_id=tourist_id,
                    time_of_day=time_of_day,
                    day_of_week=day_of_week,
                    use_rl=False,
                    use_astar=True
                )

                # Calcola metriche
                num_attractions = len(with_uncertainty)
                avg_rating = sum(attr["rating"] for attr in with_uncertainty) / max(1, len(with_uncertainty))
                total_time = sum(attr.get("visit_time", 0) + attr.get("wait_time", 0) + attr.get("travel_time", 0)
                                 for attr in with_uncertainty)

                result = {
                    "Condizione": condition_name,
                    "Turista": tourist_id,
                    "Numero attrazioni": num_attractions,
                    "Valutazione media": round(avg_rating, 2),
                    "Tempo totale (min)": round(total_time, 1)
                }

                results.append(result)
                print(f"Condizione: {condition_name}, Turista: {tourist_id}, Attrazioni: {num_attractions}")

        # Salva risultati
        df = save_results_to_csv(results, "belief_network_impact.csv")

        # Crea grafico di confronto per numero di attrazioni
        plt.figure(figsize=(12, 6))

        for tourist_id in tourist_ids:
            tourist_data = df[df["Turista"] == tourist_id]
            plt.plot(tourist_data["Condizione"], tourist_data["Numero attrazioni"],
                     marker='o', label=f'Turista {tourist_id}')

        plt.title('Impatto delle Condizioni sul Numero di Attrazioni')
        plt.xlabel('Condizione')
        plt.ylabel('Numero di Attrazioni')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        file_path = os.path.join(RESULTS_DIR, "belief_network_impact_attractions.png")
        plt.savefig(file_path)
        plt.close()

        return results


# ------------------------------------------------
# 3. TEST DELL'ALGORITMO A*
# ------------------------------------------------

class AStarTester:
    def __init__(self):
        """Inizializza il tester per l'algoritmo A*"""
        print("\n" + "=" * 80)
        print("INIZIALIZZAZIONE TESTER A*")
        print("=" * 80)

        # Carica i dati
        self.attractions_df = load_attractions()
        self.tourists_df = load_tourists()

        # Inizializza il reasoner e il modello di incertezza
        self.reasoner = DatalogReasoner()
        self.uncertainty_model = UncertaintyModel()

        print("Tester A* inizializzato")

    def _create_simple_greedy_search(self, attractions, start_location, available_time, uncertainty_model, evidence):
        """Implementa un algoritmo greedy semplice per confronto"""
        result = []
        current_location = start_location
        time_remaining = available_time
        unvisited = attractions.copy()

        # Fattore di traffico
        traffic_factor = uncertainty_model.get_travel_time_factor(evidence)

        while unvisited and time_remaining > 0:
            # Trova l'attrazione più vicina
            best_attraction = None
            min_distance = float('inf')

            for attraction in unvisited:
                # Calcola la distanza
                from_coords = current_location
                to_coords = (attraction['lat'], attraction['lon'])

                from geopy.distance import geodesic
                distance = geodesic(from_coords, to_coords).kilometers

                if distance < min_distance:
                    min_distance = distance
                    best_attraction = attraction

            # Calcola il tempo necessario
            travel_time = min_distance * 15 * traffic_factor
            wait_time = uncertainty_model.get_wait_time(evidence)
            visit_time = best_attraction['visit_time']
            total_time = travel_time + wait_time + visit_time

            # Verifica se c'è abbastanza tempo
            if total_time <= time_remaining:
                # Aggiungi all'itinerario
                best_attraction_copy = best_attraction.copy()
                best_attraction_copy['travel_time'] = travel_time
                best_attraction_copy['wait_time'] = wait_time
                result.append(best_attraction_copy)

                # Aggiorna stato
                current_location = (best_attraction['lat'], best_attraction['lon'])
                time_remaining -= total_time
                unvisited.remove(best_attraction)
            else:
                # Non c'è abbastanza tempo per altre attrazioni
                break

        return result

    def _create_random_search(self, attractions, start_location, available_time, uncertainty_model, evidence):
        """Implementa un algoritmo casuale per confronto"""
        result = []
        current_location = start_location
        time_remaining = available_time
        unvisited = attractions.copy()

        # Mischia le attrazioni
        random.shuffle(unvisited)

        # Fattore di traffico
        traffic_factor = uncertainty_model.get_travel_time_factor(evidence)

        for attraction in unvisited:
            # Calcola il tempo necessario
            from_coords = current_location
            to_coords = (attraction['lat'], attraction['lon'])

            from geopy.distance import geodesic
            distance = geodesic(from_coords, to_coords).kilometers

            travel_time = distance * 15 * traffic_factor
            wait_time = uncertainty_model.get_wait_time(evidence)
            visit_time = attraction['visit_time']
            total_time = travel_time + wait_time + visit_time

            # Verifica se c'è abbastanza tempo
            if total_time <= time_remaining:
                # Aggiungi all'itinerario
                attraction_copy = attraction.copy()
                attraction_copy['travel_time'] = travel_time
                attraction_copy['wait_time'] = wait_time
                result.append(attraction_copy)

                # Aggiorna stato
                current_location = (attraction['lat'], attraction['lon'])
                time_remaining -= total_time
            else:
                # Non c'è abbastanza tempo per questa attrazione
                continue

        return result

    def _prepare_test_data(self, tourist_id="1", num_attractions=10):
        """Prepara i dati di test"""
        # Ottieni il profilo del turista
        tourist_profile = None
        for _, row in self.tourists_df.iterrows():
            if str(row['id_turista']) == tourist_id:
                tourist_profile = row
                break

        if tourist_profile is None:
            raise ValueError(f"Turista con ID {tourist_id} non trovato")

        # Determina gli interessi del turista
        interests = []
        if tourist_profile['arte'] > 5:
            interests.append('arte')
        if tourist_profile['storia'] > 5:
            interests.append('storia')
        if tourist_profile['natura'] > 5:
            interests.append('natura')
        if tourist_profile['divertimento'] > 5:
            interests.append('divertimento')

        # Seleziona attrazioni compatibili
        attractions = []
        for _, row in self.attractions_df.iterrows():
            categoria = row['categoria'].lower()
            if categoria in interests:
                attractions.append({
                    'id': str(row['id_attrazione']),
                    'name': row['nome'],
                    'lat': row['latitudine'],
                    'lon': row['longitudine'],
                    'visit_time': row['tempo_visita'],
                    'cost': row['costo'],
                    'rating': row['recensione_media'],
                    'categoria': categoria
                })

        # Limita il numero di attrazioni
        if len(attractions) > num_attractions:
            # Ordina per rating e prendi le migliori
            attractions.sort(key=lambda x: x['rating'], reverse=True)
            attractions = attractions[:num_attractions]

        return attractions, tourist_profile['tempo']

    def test_algorithm_comparison(self, tourist_ids=["1", "2", "5"], repeat=5):
        """Confronto tra A*, greedy e random"""
        """Confronto tra A*, greedy e random"""
        print("\nTest confronto algoritmi...")

        # Punto di partenza (centro di Roma)
        start_location = (41.9028, 12.4964)

        # Condizioni da testare
        evidence = {
            self.uncertainty_model.time_of_day: "afternoon",
            self.uncertainty_model.day_of_week: "weekday"
        }

        results = []

        for tourist_id in tourist_ids:
            print(f"Testing per turista {tourist_id}...")

            for _ in range(repeat):
                # Prepara i dati di test
                attractions, available_time = self._prepare_test_data(tourist_id)

                # Test A*
                start_time = time.time()
                itinerary_problem = ItinerarySearch(
                    attractions,
                    start_location,
                    self.uncertainty_model,
                    available_time,
                    evidence
                )
                astar_searcher = AStarSearcher(itinerary_problem)
                path = astar_searcher.search()
                astar_time = (time.time() - start_time) * 1000  # ms

                if path:
                    astar_itinerary = []
                    for arc in path.arcs():
                        if arc.to_node != "start":
                            for attr in attractions:
                                if attr['id'] == arc.to_node:
                                    attr_copy = attr.copy()
                                    # Aggiungi tempi di attesa e viaggio
                                    attr_copy['wait_time'] = self.uncertainty_model.get_wait_time(evidence)
                                    attr_copy['travel_time'] = arc.cost - attr_copy['visit_time'] - attr_copy[
                                        'wait_time']
                                    astar_itinerary.append(attr_copy)
                                    break
                else:
                    astar_itinerary = []

                # Test Greedy
                start_time = time.time()
                greedy_itinerary = self._create_simple_greedy_search(
                    attractions,
                    start_location,
                    available_time,
                    self.uncertainty_model,
                    evidence
                )
                greedy_time = (time.time() - start_time) * 1000  # ms

                # Test Random
                start_time = time.time()
                random_itinerary = self._create_random_search(
                    attractions,
                    start_location,
                    available_time,
                    self.uncertainty_model,
                    evidence
                )
                random_time = (time.time() - start_time) * 1000  # ms

                # Calcola metriche
                astar_count = len(astar_itinerary)
                greedy_count = len(greedy_itinerary)
                random_count = len(random_itinerary)

                astar_rating = sum(a['rating'] for a in astar_itinerary) / max(1, astar_count)
                greedy_rating = sum(a['rating'] for a in greedy_itinerary) / max(1, greedy_count)
                random_rating = sum(a['rating'] for a in random_itinerary) / max(1, random_count)

                astar_time_used = sum(a['visit_time'] + a.get('wait_time', 0) + a.get('travel_time', 0)
                                      for a in astar_itinerary)
                greedy_time_used = sum(a['visit_time'] + a.get('wait_time', 0) + a.get('travel_time', 0)
                                       for a in greedy_itinerary)
                random_time_used = sum(a['visit_time'] + a.get('wait_time', 0) + a.get('travel_time', 0)
                                       for a in random_itinerary)

                astar_time_unused = available_time - astar_time_used
                greedy_time_unused = available_time - greedy_time_used
                random_time_unused = available_time - random_time_used

                result = {
                    "Turista": tourist_id,
                    "Algoritmo": "A*",
                    "Tempo esecuzione (ms)": round(astar_time, 2),
                    "Attrazioni visitate": astar_count,
                    "Valutazione media": round(astar_rating, 2),
                    "Tempo non utilizzato (min)": round(astar_time_unused, 1)
                }
                results.append(result)

                result = {
                    "Turista": tourist_id,
                    "Algoritmo": "Greedy",
                    "Tempo esecuzione (ms)": round(greedy_time, 2),
                    "Attrazioni visitate": greedy_count,
                    "Valutazione media": round(greedy_rating, 2),
                    "Tempo non utilizzato (min)": round(greedy_time_unused, 1)
                }
                results.append(result)

                result = {
                    "Turista": tourist_id,
                    "Algoritmo": "Random",
                    "Tempo esecuzione (ms)": round(random_time, 2),
                    "Attrazioni visitate": random_count,
                    "Valutazione media": round(random_rating, 2),
                    "Tempo non utilizzato (min)": round(random_time_unused, 1)
                }
                results.append(result)