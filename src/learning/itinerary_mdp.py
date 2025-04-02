import random
import math
import time
from geopy.distance import geodesic


class ItineraryMDP:
    """Ambiente MDP per la generazione di itinerari turistici"""

    def __init__(self, tourist_id, reasoner, uncertainty_model):
        """
        Inizializza l'ambiente MDP
        tourist_id: ID del turista
        reasoner: Istanza di OntologyReasoner
        uncertainty_model: Istanza di UncertaintyModel
        """
        self.tourist_id = tourist_id
        self.reasoner = reasoner
        self.uncertainty_model = uncertainty_model

        # Ottieni il profilo del turista
        print(f"Ottenendo profilo del turista {tourist_id}...")
        self.tourist = reasoner.get_tourist_by_id(tourist_id)
        if not self.tourist:
            print(f"ATTENZIONE: Turista {tourist_id} non trovato!")
            self.actions = []
            return

        # Ottieni le attrazioni rilevanti (filtro iniziale)
        print("Cercando attrazioni rilevanti...")
        interest_types = self.tourist.hasInterest
        matching_attractions = []

        try:
            # Ottieni tutte le attrazioni dalla ontologia
            all_attractions = self.reasoner.onto.Attraction.instances()
            print(f"Trovate {len(all_attractions)} attrazioni totali")

            # Filtra attrazioni per interessi
            for attraction in all_attractions:
                if not hasattr(attraction, 'hasCategory'):
                    continue

                for category in attraction.hasCategory:
                    if category in interest_types:  # Confronta stringhe direttamente
                        matching_attractions.append(attraction)
                        break

            print(f"Trovate {len(matching_attractions)} attrazioni corrispondenti agli interessi")

            # Se ci sono meno di 3 attrazioni, cerca attrazioni con tempo di visita breve
            if len(matching_attractions) < 3:
                print("Trovate poche attrazioni, cercando attrazioni con tempo di visita breve...")
                # Tempo disponibile del turista
                available_time = self.tourist.hasAvailableTime[0] if hasattr(self.tourist,
                                                                             'hasAvailableTime') and self.tourist.hasAvailableTime else 480

                # Cerca attrazioni con tempo di visita <= 120 minuti (2 ore)
                short_attraction_ids = self.reasoner.find_attractions_by_max_time(120)

                for attr_id in short_attraction_ids:
                    # Converti a intero se necessario
                    numeric_id = int(attr_id) if attr_id.isdigit() else attr_id

                    # Cerca l'attrazione per ID
                    short_attr = self.reasoner.onto.search_one(f"attraction_{numeric_id}")
                    if short_attr and short_attr not in matching_attractions:
                        print(f"Aggiunta attrazione breve: {short_attr.name}")
                        matching_attractions.append(short_attr)
        except Exception as e:
            print(f"Errore nella ricerca delle attrazioni: {e}")
            matching_attractions = []

        # Le azioni sono le attrazioni che possono essere aggiunte all'itinerario
        self.actions = [attr.name for attr in matching_attractions]
        print(f"Azioni disponibili: {len(self.actions)}")

        # Stato iniziale: tempo disponibile e itinerario vuoto
        self.available_time = self.tourist.hasAvailableTime[0] if hasattr(self.tourist,
                                                                          'hasAvailableTime') and self.tourist.hasAvailableTime else 480  # 8 ore
        self.current_location = "start"  # Posizione iniziale
        self.itinerary = []  # Lista di attrazioni nell'itinerario
        self.reward = 0  # Reward accumulato

        # Evidenze per il modello probabilistico (default)
        self.evidence = {
            self.uncertainty_model.time_of_day: "afternoon",
            self.uncertainty_model.day_of_week: "weekday"
        }

        # Calcola il fattore di traffico
        self.traffic_factor = self.uncertainty_model.get_travel_time_factor(self.evidence)

        # Codifica dello stato iniziale
        self.state = self._encode_state()

    def _encode_state(self):
        """Codifica lo stato come una stringa"""
        visited = ",".join(self.itinerary) if self.itinerary else "none"
        return f"{self.available_time}|{self.current_location}|{visited}"

    def _decode_state(self, state_str):
        """Decodifica lo stato da una stringa"""
        parts = state_str.split('|')
        time_remaining = int(parts[0])
        current_location = parts[1]
        visited = parts[2].split(',') if parts[2] != "none" else []
        return time_remaining, current_location, visited

    def do(self, action):
        """Esegue un'azione e restituisce reward e nuovo stato"""
        start_time = time.time()
        print(f"MDP: Eseguendo azione {action}")

        # Verifica se l'azione è valida
        if action not in self.actions:
            print(f"MDP: Azione non valida")
            return -10, self.state  # Penalità per azione non valida

        # Verifica se l'attrazione è già nell'itinerario
        if action in self.itinerary:
            print(f"MDP: Attrazione già nell'itinerario")
            # Rimuovi l'azione dalle opzioni disponibili
            if action in self.actions:
                self.actions.remove(action)
                print(f"Rimossa azione {action} dalle azioni disponibili (già nell'itinerario)")
            return -5, self.state  # Penalità per ripetizione

        # Ottieni l'attrazione
        print(f"MDP: Cercando attrazione {action}")
        attraction = self.reasoner.onto.search_one(iri=f"*{action}")
        print(f"MDP: Attrazione trovata: {attraction is not None}")

        if not attraction:
            print(f"MDP: Attrazione non trovata")
            # Rimuovi l'azione dalle opzioni disponibili
            if action in self.actions:
                self.actions.remove(action)
                print(f"Rimossa azione {action} dalle azioni disponibili (non trovata)")
            return -10, self.state

        # Calcola il tempo per la visita
        visit_time = attraction.hasEstimatedVisitTime[0] if hasattr(attraction,
                                                                    'hasEstimatedVisitTime') and attraction.hasEstimatedVisitTime else 60
        print(f"MDP: Tempo visita: {visit_time}")

        # Calcola il tempo di attesa basato sul modello di incertezza
        wait_time = self.uncertainty_model.get_wait_time(self.evidence)
        print(f"MDP: Tempo attesa: {wait_time}")

        # Calcola il tempo di viaggio
        travel_time = 30  # Default 30 minuti
        if self.current_location != "start":
            # Se non è la prima attrazione, calcoliamo il tempo di viaggio
            prev_attraction = self.reasoner.onto.search_one(iri=f"*{self.current_location}")
            if prev_attraction and hasattr(attraction, 'hasLatitude') and hasattr(attraction,
                                                                                  'hasLongitude') and attraction.hasLatitude and attraction.hasLongitude:
                try:
                    # Calcola distanza con geopy
                    from_coords = (prev_attraction.hasLatitude[0], prev_attraction.hasLongitude[0])
                    to_coords = (attraction.hasLatitude[0], attraction.hasLongitude[0])

                    distance = geodesic(from_coords, to_coords).kilometers

                    # Converti in tempo di viaggio (15 minuti per km, moltiplicato per il fattore di traffico)
                    travel_time = distance * 15 * self.traffic_factor
                    print(f"MDP: Distanza: {distance:.2f} km, tempo viaggio: {travel_time:.2f} min")
                except Exception as e:
                    print(f"MDP: Errore nel calcolo del tempo di viaggio: {e}")
                    travel_time = 30  # Default in caso di errore
            else:
                print(f"MDP: Mancano coordinate, usato tempo di viaggio di default")

        # Tempo totale necessario
        total_time = visit_time + wait_time + travel_time
        print(f"MDP: Tempo totale necessario: {total_time:.2f} min")

        # Verifica se c'è abbastanza tempo
        if self.available_time < total_time:
            print(f"MDP: Tempo insufficiente. Disponibile: {self.available_time}, necessario: {total_time}")
            # Rimuovi l'azione dalle opzioni disponibili
            if action in self.actions:
                self.actions.remove(action)
                print(f"Rimossa azione {action} dalle azioni disponibili (tempo insufficiente)")
            return -5, self.state  # Penalità per tempo insufficiente

        # Aggiorna lo stato
        self.available_time -= total_time
        self.current_location = action
        self.itinerary.append(action)
        self.state = self._encode_state()

        # Calcola il reward
        reward = self._calculate_reward(attraction)
        self.reward += reward  # Accumula reward

        elapsed_time = time.time() - start_time
        print(f"MDP: Azione completata in {elapsed_time:.2f} secondi. Reward: {reward}")

        return reward, self.state

    def _calculate_reward(self, attraction):
        """Calcola il reward per aver aggiunto un'attrazione"""
        # Base reward
        reward = 10

        # Bonus per alta valutazione
        if hasattr(attraction, 'hasAverageRating') and attraction.hasAverageRating:
            rating = attraction.hasAverageRating[0]
            reward += (rating - 3) * 3  # Da -6 a +6

        # Bonus per categoria corrispondente agli interessi
        interest_types = [type(interest) for interest in self.tourist.hasInterest]
        for category in attraction.hasCategory:
            if type(category) in interest_types:
                reward += 5
                break

        # Bonus per diversità (categorie non ancora visitate)
        if self._adds_diversity(attraction):
            reward += 5

        return reward

    def _adds_diversity(self, attraction):
        """Verifica se l'attrazione aggiunge diversità all'itinerario"""
        if not self.itinerary:
            return True

        # Categorie già visitate
        visited_categories = set()
        for attr_name in self.itinerary:
            attr = self.reasoner.onto.search_one(iri=f"*{attr_name}")
            if attr and hasattr(attr, 'hasCategory'):
                visited_categories.update(type(cat) for cat in attr.hasCategory)

        # Categorie della nuova attrazione
        new_categories = set(type(cat) for cat in attraction.hasCategory)

        # Verifica se c'è almeno una categoria non ancora visitata
        return bool(new_categories - visited_categories)