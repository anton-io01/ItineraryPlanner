# itinerary_mdp.py
from lib.rlProblem import RL_env
import math
from geopy.distance import geodesic
import random


class ItineraryMDP(RL_env):
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
        self.tourist = reasoner.get_tourist_by_id(tourist_id)

        # Ottieni le attrazioni rilevanti (filtro iniziale)
        interest_types = [type(interest) for interest in self.tourist.hasInterest]
        matching_attractions = []

        for attraction in self.reasoner.onto.Attraction.instances():
            for category in attraction.hasCategory:
                if type(category) in interest_types:
                    matching_attractions.append(attraction)
                    break

        # Le azioni sono le attrazioni che possono essere aggiunte all'itinerario
        self.actions = [attr.name for attr in matching_attractions]

        # Stato iniziale: tempo disponibile e itinerario vuoto
        self.available_time = self.tourist.hasAvailableTime[0] if self.tourist.hasAvailableTime else 480  # 8 ore
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

        # Codifica lo stato iniziale
        self.state = self._encode_state()

        # Nome dell'ambiente
        self.name = f"ItineraryMDP_Tourist_{tourist_id}"

    def _encode_state(self):
        """Codifica lo stato come una stringa"""
        # Formato: tempo_rimanente|posizione_attuale|attr1,attr2,...
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
        # Verifica se l'azione è valida
        if action not in self.actions:
            return -10, self.state  # Penalità per azione non valida

        # Verifica se l'attrazione è già nell'itinerario
        if action in self.itinerary:
            return -5, self.state  # Penalità per ripetizione

        # Ottieni l'attrazione
        attraction = self.reasoner.onto.search_one(iri=f"*{action}")

        if not attraction:
            return -10, self.state

        # Calcola il tempo per la visita
        visit_time = attraction.hasEstimatedVisitTime[0] if attraction.hasEstimatedVisitTime else 60

        # Calcola il tempo di attesa basato sul modello di incertezza
        wait_time = self.uncertainty_model.get_wait_time(self.evidence)

        # Calcola il tempo di viaggio
        travel_time = 30  # Default 30 minuti
        if self.current_location != "start":
            # Se non è la prima attrazione, calcoliamo il tempo di viaggio
            prev_attraction = self.reasoner.onto.search_one(iri=f"*{self.current_location}")
            if prev_attraction and attraction.hasLatitude and attraction.hasLongitude:
                # Calcola distanza con geopy
                from_coords = (prev_attraction.hasLatitude[0], prev_attraction.hasLongitude[0])
                to_coords = (attraction.hasLatitude[0], attraction.hasLongitude[0])

                distance = geodesic(from_coords, to_coords).kilometers

                # Converti in tempo di viaggio (15 minuti per km, moltiplicato per il fattore di traffico)
                travel_time = distance * 15 * self.traffic_factor

        # Tempo totale necessario
        total_time = visit_time + wait_time + travel_time

        # Verifica se c'è abbastanza tempo
        if self.available_time < total_time:
            return -5, self.state  # Penalità per tempo insufficiente

        # Aggiorna lo stato
        self.available_time -= total_time
        self.current_location = action
        self.itinerary.append(action)
        self.state = self._encode_state()

        # Calcola il reward
        reward = self._calculate_reward(attraction)
        self.reward += reward  # Accumula reward

        return reward, self.state

    def _calculate_reward(self, attraction):
        """Calcola il reward per aver aggiunto un'attrazione"""
        # Base reward
        reward = 10

        # Bonus per alta valutazione
        if attraction.hasAverageRating:
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
            if attr and attr.hasCategory:
                visited_categories.update(type(cat) for cat in attr.hasCategory)

        # Categorie della nuova attrazione
        new_categories = set(type(cat) for cat in attraction.hasCategory)

        # Verifica se c'è almeno una categoria non ancora visitata
        return bool(new_categories - visited_categories)