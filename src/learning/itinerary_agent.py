import random
import time
from src.learning.itinerary_mdp import ItineraryMDP


class ItineraryAgent:
    """Agente per la generazione di itinerari turistici"""

    def __init__(self, tourist_id, reasoner, uncertainty_model):
        """
        Inizializza l'agente
        tourist_id: ID del turista
        reasoner: Istanza di DatalogReasoner
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
            start_time = time.time()

            # Reinizializza l'ambiente MDP
            self.mdp = ItineraryMDP(
                self.tourist_id,
                self.reasoner,
                self.uncertainty_model
            )

            # Esegui episodio
            total_reward = 0
            done = False
            step = 0
            consecutive_negative_rewards = 0  # Conta reward negativi consecutivi

            # Salva azioni già tentate con reward negativo
            failed_actions = set()

            while not done:
                step += 1

                # Termina se troppe iterazioni
                if step > 100:  # Limite massimo di step per episodio
                    break

                # Verifica se ci sono ancora azioni valide disponibili
                valid_actions = [a for a in self.mdp.actions if a not in failed_actions]
                if not valid_actions:
                    done = True
                    continue

                # Selezione azione
                action = self._select_action(valid_actions)
                if action is None:
                    done = True
                    continue


                # Esegui azione
                reward, new_state = self.mdp.do(action)
                total_reward += reward


                # Aggiorna policy
                self._update_policy(action, reward)

                # Se l'azione ha reward negativo, aggiungi alle azioni fallite
                if reward < 0:
                    failed_actions.add(action)
                    consecutive_negative_rewards += 1
                    if consecutive_negative_rewards > 5:  # Termina dopo 5 reward negativi consecutivi
                        done = True
                        continue
                else:
                    consecutive_negative_rewards = 0  # Reset se otteniamo un reward positivo

                # Verifica condizione di fine
                if len(self.mdp.itinerary) >= 10 or self.mdp.available_time <= 0:
                    done = True

    def _select_action(self, valid_actions=None):
        """Selezione dell'azione basata sulla policy corrente"""
        # Usa la lista completa di azioni o solo quelle valide se specificate
        available_actions = valid_actions if valid_actions is not None else self.mdp.actions


        # Se non ci sono azioni disponibili, restituisci None
        if not available_actions:
            return None

        # Esplora o sfrutta
        if random.random() < 0.1:  # esplorazione
            chosen_action = random.choice(available_actions)
            return chosen_action
        else:  # sfruttamento
            # Trova l'azione con il valore policy più alto tra quelle disponibili
            best_action = max(available_actions, key=lambda a: self.policy.get(a, 0))
            return best_action

    def _update_policy(self, action, reward):
        """Aggiorna la policy basata sul reward"""
        if action not in self.policy:
            self.policy[action] = 0

        # Aggiornamento della policy con media mobile
        old_value = self.policy[action]
        self.policy[action] = 0.9 * old_value + 0.1 * reward

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
        step = 0
        failed_actions = set()

        while not done:
            step += 1

            # Termina se troppe iterazioni
            if step > 100:
                break

            # Filtra solo azioni valide
            valid_actions = [a for a in self.mdp.actions if a not in failed_actions]
            if not valid_actions:
                break

            action = self._select_action(valid_actions)
            if action is None:
                break

            reward, new_state = self.mdp.do(action)

            # Se azione fallita, aggiungi a failed_actions
            if reward < 0:
                failed_actions.add(action)
                continue

            total_reward += reward

            if len(self.mdp.itinerary) >= 10 or self.mdp.available_time <= 0:
                print(
                    f"Itinerario completo: {len(self.mdp.itinerary)} elementi, tempo rimasto: {self.mdp.available_time}")
                done = True

        print(f"Itinerario generato con reward totale: {total_reward}")
        return self.mdp.itinerary, total_reward