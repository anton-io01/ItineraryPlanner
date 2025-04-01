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
        print(f"Creazione ambiente MDP per turista {tourist_id}...")
        self.mdp = ItineraryMDP(tourist_id, reasoner, uncertainty_model)
        print(f"Ambiente MDP creato con {len(self.mdp.actions)} azioni possibili")

        # Dizionario per memorizzare le policy
        self.policy = {}

    def train(self, num_episodes=100):
        """Addestra l'agente per un numero specificato di episodi"""
        print(f"Inizio addestramento con {num_episodes} episodi...")
        for episode in range(num_episodes):
            print(f"Iniziando episodio {episode + 1}...")
            start_time = time.time()

            # Reinizializza l'ambiente MDP
            print(f"Reinizializzazione ambiente MDP...")
            self.mdp = ItineraryMDP(
                self.tourist_id,
                self.reasoner,
                self.uncertainty_model
            )
            print(f"Ambiente reinizializzato con {len(self.mdp.actions)} azioni possibili")

            # Esegui episodio
            total_reward = 0
            done = False
            step = 0
            consecutive_negative_rewards = 0  # Conta reward negativi consecutivi

            # Salva azioni già tentate con reward negativo
            failed_actions = set()

            while not done:
                step += 1
                print(f"Episodio {episode + 1}, Step {step}: Selezionando azione...")

                # Termina se troppe iterazioni
                if step > 100:  # Limite massimo di step per episodio
                    print("Troppi step, terminazione forzata dell'episodio")
                    break

                # Verifica se ci sono ancora azioni valide disponibili
                valid_actions = [a for a in self.mdp.actions if a not in failed_actions]
                if not valid_actions:
                    print(f"Nessuna azione valida rimasta, terminando episodio")
                    done = True
                    continue

                # Selezione azione
                action = self._select_action(valid_actions)
                if action is None:
                    print(f"Nessuna azione valida disponibile, terminando episodio")
                    done = True
                    continue

                print(f"Azione selezionata: {action}, eseguendo...")

                # Esegui azione
                reward, new_state = self.mdp.do(action)
                total_reward += reward

                print(f"Azione completata, reward: {reward}, reward totale: {total_reward}")

                # Aggiorna policy
                self._update_policy(action, reward)

                # Se l'azione ha reward negativo, aggiungi alle azioni fallite
                if reward < 0:
                    failed_actions.add(action)
                    consecutive_negative_rewards += 1
                    if consecutive_negative_rewards > 5:  # Termina dopo 5 reward negativi consecutivi
                        print("Troppe azioni con reward negativo, terminazione forzata")
                        done = True
                        continue
                else:
                    consecutive_negative_rewards = 0  # Reset se otteniamo un reward positivo

                # Verifica condizione di fine
                if len(self.mdp.itinerary) >= 10 or self.mdp.available_time <= 0:
                    print(
                        f"Episodio terminato: itinerario di {len(self.mdp.itinerary)} elementi, tempo rimasto: {self.mdp.available_time}")
                    done = True

            episode_time = time.time() - start_time
            # Stampa progresso
            print(f"Episodio {episode + 1} completato in {episode_time:.2f} secondi, Reward totale: {total_reward}")
            print(f"Itinerario generato: {self.mdp.itinerary}")

    def _select_action(self, valid_actions=None):
        """Selezione dell'azione basata sulla policy corrente"""
        # Usa la lista completa di azioni o solo quelle valide se specificate
        available_actions = valid_actions if valid_actions is not None else self.mdp.actions

        print(f"Selezionando un'azione tra {len(available_actions)} possibili")

        # Se non ci sono azioni disponibili, restituisci None
        if not available_actions:
            print("Nessuna azione disponibile!")
            return None

        # Esplora o sfrutta
        if random.random() < 0.1:  # esplorazione
            chosen_action = random.choice(available_actions)
            print(f"Strategia: esplorazione, azione selezionata: {chosen_action}")
            return chosen_action
        else:  # sfruttamento
            # Trova l'azione con il valore policy più alto tra quelle disponibili
            best_action = max(available_actions, key=lambda a: self.policy.get(a, 0))
            print(f"Strategia: sfruttamento, azione selezionata: {best_action}")
            return best_action

    def _update_policy(self, action, reward):
        """Aggiorna la policy basata sul reward"""
        if action not in self.policy:
            self.policy[action] = 0

        # Aggiornamento della policy con media mobile
        old_value = self.policy[action]
        self.policy[action] = 0.9 * old_value + 0.1 * reward
        print(f"Policy aggiornata per azione {action}: {old_value:.2f} -> {self.policy[action]:.2f}")

    def generate_itinerary(self, time_of_day="afternoon", day_of_week="weekday"):
        """Genera un itinerario usando la policy appresa"""
        print(f"Generazione itinerario per time_of_day={time_of_day}, day_of_week={day_of_week}")

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
        print(f"Fattore di traffico: {self.mdp.traffic_factor}")

        # Genera itinerario
        total_reward = 0
        done = False
        step = 0
        failed_actions = set()

        while not done:
            step += 1
            print(f"Generazione itinerario, step {step}")

            # Termina se troppe iterazioni
            if step > 100:
                print("Troppi step, terminazione forzata")
                break

            # Filtra solo azioni valide
            valid_actions = [a for a in self.mdp.actions if a not in failed_actions]
            if not valid_actions:
                print("Nessuna azione valida rimasta, terminando generazione")
                break

            action = self._select_action(valid_actions)
            if action is None:
                print("Nessuna azione valida disponibile, terminando generazione")
                break

            reward, new_state = self.mdp.do(action)

            # Se azione fallita, aggiungi a failed_actions
            if reward < 0:
                failed_actions.add(action)
                print(f"Azione {action} fallita con reward {reward}, sarà ignorata")
                continue

            total_reward += reward
            print(f"Aggiunta attrazione {action}, reward: {reward}")

            if len(self.mdp.itinerary) >= 10 or self.mdp.available_time <= 0:
                print(
                    f"Itinerario completo: {len(self.mdp.itinerary)} elementi, tempo rimasto: {self.mdp.available_time}")
                done = True

        print(f"Itinerario generato con reward totale: {total_reward}")
        return self.mdp.itinerary, total_reward