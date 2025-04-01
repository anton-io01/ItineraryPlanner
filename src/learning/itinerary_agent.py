import random
from src.learning.itinerary_mdp import ItineraryMDP


class ItineraryAgent:
    """Agente per la generazione di itinerari turistici"""

    def __init__(self, tourist_id, reasoner, uncertainty_model):
        """
        Inizializza l'agente
        tourist_id: ID del turista
        reasoner: Istanza di OntologyReasoner
        uncertainty_model: Istanza di UncertaintyModel
        """
        self.tourist_id = tourist_id
        self.reasoner = reasoner
        self.uncertainty_model = uncertainty_model

        # Crea l'ambiente MDP
        self.mdp = ItineraryMDP(tourist_id, reasoner, uncertainty_model)

        # Dizionario per memorizzare le policy
        self.policy = {}

    def train(self, num_episodes=100):
        """Addestra l'agente per un numero specificato di episodi"""
        for episode in range(num_episodes):
            # Reinizializza l'ambiente MDP
            self.mdp = ItineraryMDP(
                self.tourist_id,
                self.reasoner,
                self.uncertainty_model
            )

            # Esegui episodio
            total_reward = 0
            done = False

            while not done:
                # Selezione azione
                action = self._select_action()

                # Esegui azione
                reward, new_state = self.mdp.do(action)
                total_reward += reward

                # Aggiorna policy
                self._update_policy(action, reward)

                # Verifica condizione di fine
                if len(self.mdp.itinerary) >= 10 or self.mdp.available_time <= 0:
                    done = True

            # Stampa progresso
            if (episode + 1) % 20 == 0:
                print(f"Episodio {episode + 1}, Reward totale: {total_reward}")

    def _select_action(self):
        """Selezione dell'azione basata sulla policy corrente"""
        # Se non ci sono azioni precedenti, scegli casualmente
        if not self.mdp.actions:
            return random.choice(self.mdp.actions)

        # Esplora o sfrutta
        if random.random() < 0.1:  # esplorazione
            return random.choice(self.mdp.actions)
        else:  # sfruttamento
            return max(self.mdp.actions, key=lambda a: self.policy.get(a, 0))

    def _update_policy(self, action, reward):
        """Aggiorna la policy basata sul reward"""
        if action not in self.policy:
            self.policy[action] = 0

        # Aggiornamento della policy con media mobile
        self.policy[action] = 0.9 * self.policy[action] + 0.1 * reward

    def generate_itinerary(self, time_of_day="afternoon", day_of_week="weekday"):
        """Genera un itinerario usando la policy appresa"""
        # Reinizializza l'ambiente
        self.mdp = ItineraryMDP(
            self.tourist_id,
            self.reasoner,
            self.uncertainty_model
        )

        # Aggiorna evidenze
        self.mdp.evidence = {
            self.mdp.uncertainty_model.time_of_day: time_of_day,
            self.mdp.uncertainty_model.day_of_week: day_of_week
        }

        # Aggiorna fattore di traffico
        self.mdp.traffic_factor = self.mdp.uncertainty_model.get_travel_time_factor(self.mdp.evidence)

        # Genera itinerario
        total_reward = 0
        done = False

        while not done:
            action = self._select_action()
            reward, new_state = self.mdp.do(action)
            total_reward += reward

            if len(self.mdp.itinerary) >= 10 or self.mdp.available_time <= 0:
                done = True

        return self.mdp.itinerary, total_reward