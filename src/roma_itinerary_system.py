# src/roma_itinerary_system.py
import os
import time

# Import dei moduli interni
from src.data.data_manager import load_attractions, load_tourists, get_tourist_profile, get_attraction_details
from src.knowledge.ontology_population import populate_ontology
from src.knowledge.reasoning_module import DatalogReasoner
from src.uncertainty.uncertainty_model import UncertaintyModel
from src.learning.itinerary_agent import ItineraryAgent
from src.planning.itinerary_search import ItinerarySearch, AStarSearcher
from geopy.distance import geodesic

class RomaItinerarySystem:
    """Sistema completo per la generazione di itinerari turistici a Roma"""

    def __init__(self):
        """Inizializza il sistema"""
        start_time = time.time()

        # Carica i dati CSV
        print("Caricamento dati...")
        self.attractions_df = load_attractions()
        self.tourists_df = load_tourists()

        # Inizializza il reasoner Datalog (non richiede Java)
        print("Inizializzazione reasoner Datalog...")
        self.reasoner = DatalogReasoner()

        # Inizializza modello di incertezza
        print("Creazione modello di incertezza...")
        self.uncertainty_model = UncertaintyModel()

        # Dizionario per agenti RL addestrati
        self.agents = {}

        print(f"Sistema inizializzato in {time.time() - start_time:.2f} secondi!")

    def generate_itinerary(self, tourist_id, time_of_day="afternoon", day_of_week="weekday",
                           use_rl=True, use_astar=True):
        """
        Genera un itinerario per un turista

        Args:
            tourist_id: ID del turista
            time_of_day: Momento della giornata ("morning", "afternoon", "evening")
            day_of_week: Giorno della settimana ("weekday", "weekend")
            use_rl: Se True, usa RL per selezionare le attrazioni
            use_astar: Se True, usa A* per ottimizzare l'ordine

        Returns:
            Lista di dizionari con informazioni sulle attrazioni nell'itinerario
        """
        print(f"\nGenerazione itinerario per turista {tourist_id}...")
        print(f"Parametri: {time_of_day}, {day_of_week}, RL={use_rl}, A*={use_astar}")

        # Ottieni il profilo del turista
        tourist_profile = get_tourist_profile(self.tourists_df, tourist_id)
        if not tourist_profile:
            print(f"Turista con ID {tourist_id} non trovato!")
            return []

        print(f"Profilo turista: arte={tourist_profile['arte']}, storia={tourist_profile['storia']}, " +
              f"natura={tourist_profile['natura']}, divertimento={tourist_profile['divertimento']}, " +
              f"tempo={tourist_profile['tempo']} minuti")

        # Evidenze per il modello probabilistico
        evidence = {
            self.uncertainty_model.time_of_day: time_of_day,
            self.uncertainty_model.day_of_week: day_of_week
        }

        # Fase 1: Selezione delle attrazioni (con RL o con query ontologiche)
        selected_attractions = []

        if use_rl:
            print("Selezione attrazioni con Reinforcement Learning...")

            # Verifica se l'agente è già addestrato
            if tourist_id not in self.agents:
                print(f"Addestramento agente RL per il turista {tourist_id}...")
                self.agents[tourist_id] = ItineraryAgent(tourist_id, self.reasoner, self.uncertainty_model)
                self.agents[tourist_id].train(num_episodes=1)

            # Genera itinerario
            attraction_ids, reward = self.agents[tourist_id].generate_itinerary(time_of_day, day_of_week)
            print(f"Itinerario generato con reward: {reward}")

            # Converti in formato utilizzabile
            for attr_id in attraction_ids:
                # Cerca di ottenere l'ID numerico
                try:
                    # Verifica se l'ID è nel formato "attraction_X"
                    if '_' in attr_id:
                        num_id = attr_id.split('_')[1]
                    else:
                        # Se è un nome di attrazione, cerca nell'elenco attrazioni
                        found = False
                        for idx, row in self.attractions_df.iterrows():
                            if row['nome'] == attr_id:
                                num_id = str(row['id_attrazione'])
                                found = True
                                break

                        if not found:
                            print(f"ATTENZIONE: Non riesco a trovare l'ID per {attr_id}, ignoro questa attrazione")
                            continue

                    # Ottieni dettagli
                    details = get_attraction_details(self.attractions_df, num_id)
                    if details:
                        selected_attractions.append({
                            'id': num_id,
                            'name': details['nome'],
                            'lat': details['latitudine'],
                            'lon': details['longitudine'],
                            'visit_time': details['tempo_visita'],
                            'cost': details['costo'],
                            'rating': details['recensione_media']
                        })
                except Exception as e:
                    print(f"Errore nell'elaborazione dell'attrazione {attr_id}: {e}")
        else:
            print("Selezione attrazioni con ragionamento ontologico...")

            # Determina gli interessi del turista
            interests = []
            if tourist_profile['arte'] > 2:
                interests.append('arte')  # In italiano minuscolo
            if tourist_profile['storia'] > 2:
                interests.append('storia')  # In italiano minuscolo
            if tourist_profile['natura'] > 2:
                interests.append('natura')  # In italiano minuscolo
            if tourist_profile['divertimento'] > 2:
                interests.append('divertimento')  # In italiano minuscolo

            print(f"Interessi identificati: {interests}")

            # Query all'ontologia
            attractions = self.reasoner.find_attractions_by_interest(interests)
            print(f"Trovate {len(attractions)} attrazioni che corrispondono agli interessi")

            # Filtro per rating e costo
            filtered_attractions = []
            for attr in attractions:
                attr_id = attr.name.split('_')[1]
                details = get_attraction_details(self.attractions_df, attr_id)

                # Considera solo attrazioni con rating sufficiente e costo ragionevole
                if details and details['recensione_media'] >= 3.0:
                    filtered_attractions.append({
                        'id': attr_id,
                        'name': details['nome'],
                        'lat': details['latitudine'],
                        'lon': details['longitudine'],
                        'visit_time': details['tempo_visita'],
                        'cost': details['costo'],
                        'rating': details['recensione_media']
                    })

            # Ordina per rating e prendi le migliori
            filtered_attractions.sort(key=lambda a: a['rating'], reverse=True)
            selected_attractions = filtered_attractions[:10]  # Limita a 10 per efficienza

        print(f"Selezionate {len(selected_attractions)} attrazioni")

        # Se non ci sono attrazioni selezionate, termina
        if not selected_attractions:
            print("Nessuna attrazione selezionata!")
            return []

        # Fase 2: Ottimizzazione dell'ordine (con A*)
        final_itinerary = []

        if use_astar and len(selected_attractions) > 1:
            print("Ottimizzazione dell'ordine con A*...")

            # Punto di partenza (centro di Roma)
            start_location = (41.9028, 12.4964)

            # Crea il problema di ricerca
            itinerary_problem = ItinerarySearch(
                selected_attractions,
                start_location,
                self.uncertainty_model,
                tourist_profile['tempo'],
                evidence
            )

            # Esegui A*
            searcher = AStarSearcher(itinerary_problem)
            path = searcher.search()

            if path:
                print(f"A* ha trovato un percorso ottimale con {len(path.arcs())} attrazioni")

                # Estrai l'itinerario dal percorso
                attraction_ids = [arc.to_node for arc in path.arcs() if arc.to_node != "start"]

                # Crea l'itinerario finale
                for attr_id in attraction_ids:
                    for attr in selected_attractions:
                        if attr['id'] == attr_id:
                            # Calcola tempi di attesa e viaggio
                            wait_time = self.uncertainty_model.get_wait_time(evidence)

                            # Calcola tempo di viaggio (per il primo elemento o tra elementi consecutivi)
                            travel_time = 0
                            if len(final_itinerary) > 0:
                                prev_attr = final_itinerary[-1]

                                # Usa geopy per calcolo distanza
                                from_coords = (prev_attr['lat'], prev_attr['lon'])
                                to_coords = (attr['lat'], attr['lon'])
                                distance = geodesic(from_coords, to_coords).kilometers

                                # Calcola tempo di viaggio con fattore traffico
                                traffic_factor = self.uncertainty_model.get_travel_time_factor(evidence)
                                travel_time = distance * 15 * traffic_factor

                            # Aggiungi all'itinerario finale
                            attr_copy = attr.copy()
                            attr_copy['wait_time'] = int(wait_time)
                            attr_copy['travel_time'] = int(travel_time)
                            final_itinerary.append(attr_copy)
                            break
            else:
                print("A* non ha trovato un percorso valido. Uso l'ordine originale.")
                final_itinerary = selected_attractions
        else:
            print("Uso l'ordine originale delle attrazioni...")
            # Usa l'ordine originale
            final_itinerary = selected_attractions

        print(f"Itinerario finale creato con {len(final_itinerary)} attrazioni")
        return final_itinerary

    def print_itinerary(self, itinerary):
        """Stampa un itinerario in formato leggibile"""
        if not itinerary:
            print("Itinerario vuoto!")
            return

        print(f"\nItinerario con {len(itinerary)} attrazioni:")
        print("-" * 60)

        total_time = 0
        total_cost = 0

        for i, attr in enumerate(itinerary):
            print(f"{i + 1}. {attr['name']}")
            print(f"   Valutazione: {attr['rating']}/5")
            print(f"   Costo: €{attr['cost']}")

            # Tempo di visita
            visit_time = attr['visit_time']
            print(f"   Tempo di visita: {visit_time} minuti")

            # Tempo di attesa (se disponibile)
            wait_time = attr.get('wait_time', 0)
            if wait_time > 0:
                print(f"   Tempo di attesa stimato: {wait_time} minuti")

            # Tempo di viaggio (se disponibile)
            travel_time = attr.get('travel_time', 0)
            if travel_time > 0:
                print(f"   Tempo di viaggio: {int(travel_time)} minuti")

            print()

            # Aggiorna totali
            total_time += visit_time + wait_time + travel_time
            total_cost += attr['cost']

        print("-" * 60)
        print(f"Tempo totale stimato: {int(total_time)} minuti")
        print(f"Costo totale: €{total_cost:.2f}")
        print("-" * 60)


# Esempio di utilizzo
if __name__ == "__main__":
    # Crea il sistema
    start_time = time.time()
    system = RomaItinerarySystem()
    print(f"Tempo di inizializzazione: {time.time() - start_time:.2f} secondi")

    # Genera itinerario per il turista 1 (weekend pomeriggio)
    itinerary1 = system.generate_itinerary(
        tourist_id="1",
        time_of_day="afternoon",
        day_of_week="weekend",
        use_rl=True,
        use_astar=True
    )

    # Stampa l'itinerario
    system.print_itinerary(itinerary1)

    # Genera un altro itinerario (giorno feriale mattina)
    itinerary2 = system.generate_itinerary(
        tourist_id="2",
        time_of_day="morning",
        day_of_week="weekday",
        use_rl=False,
        use_astar=True
    )

    # Stampa l'itinerario
    system.print_itinerary(itinerary2)