# itinerary_agent.py
from lib.rlStochasticPolicy import StochasticPIAgent
from lib.rlProblem import Simulate
from ..itinerary_mdp import ItineraryMDP


class ItineraryAgent(StochasticPIAgent):
    """Agente RL con policy stocastica per la generazione di itinerari turistici"""

    def __init__(self, tourist_id, reasoner, uncertainty_model):
        """
        Inizializza l'agente RL
        tourist_id: ID del turista
        reasoner: Istanza di OntologyReasoner
        uncertainty_model: Istanza di UncertaintyModel
        """
        # Crea l'ambiente MDP
        self.mdp = ItineraryMDP(tourist_id, reasoner, uncertainty_model)

        # Inizializza l'agente con policy stocastica
        StochasticPIAgent.__init__(
            self,
            f"ItineraryAgent_{tourist_id}",
            self.mdp.actions,
            discount=0.9,
            pi_init=1  # Valore iniziale per la distribuzione di Dirichlet
        )

    def train(self, num_episodes=100):
        """Addestra l'agente per un numero specificato di episodi"""
        for episode in range(num_episodes):
            # Reinizializza l'ambiente MDP
            self.mdp = ItineraryMDP(
                self.mdp.tourist_id,
                self.mdp.reasoner,
                self.mdp.uncertainty_model
            )

            # Esegui una simulazione
            sim = Simulate(self, self.mdp).start()
            sim.go(10)  # Max 10 attrazioni per itinerario

            # Stampa progresso
            if (episode + 1) % 20 == 0 or episode == 0:
                print(f"Episodio {episode + 1}/{num_episodes}, Reward: {sim.sum_rewards}")

    def generate_itinerary(self, time_of_day="afternoon", day_of_week="weekday"):
        """Genera un itinerario usando la policy appresa"""
        # Reinizializza l'ambiente con le evidenze specificate
        self.mdp = ItineraryMDP(
            self.mdp.tourist_id,
            self.mdp.reasoner,
            self.mdp.uncertainty_model
        )

        # Aggiorna le evidenze
        self.mdp.evidence = {
            self.mdp.uncertainty_model.time_of_day: time_of_day,
            self.mdp.uncertainty_model.day_of_week: day_of_week
        }

        # Aggiorna il fattore di traffico
        self.mdp.traffic_factor = self.mdp.uncertainty_model.get_travel_time_factor(self.mdp.evidence)

        # Esegui una simulazione usando la policy appresa
        sim = Simulate(self, self.mdp).start()
        sim.go(10)

        # Estrai l'itinerario finale
        _, _, itinerary = self.mdp._decode_state(self.mdp.state)

        return itinerary, self.mdp.reward