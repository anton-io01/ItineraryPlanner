# src/planning/itinerary_search.py
from src.knowledge.reasoning_module import OntologyReasoner
from src.uncertainty.uncertainty_model import UncertaintyModel
from geopy.distance import geodesic
from typing import List, Dict, Any

class Arc:
    """Rappresenta un arco tra due nodi con un costo"""
    def __init__(self, from_node, to_node, cost=1):
        self.from_node = from_node
        self.to_node = to_node
        self.cost = cost

class Path:
    """Rappresenta un percorso attraverso una sequenza di archi"""
    def __init__(self, problem, initial, arc=None):
        self.problem = problem
        self.initial = initial
        self.arc = arc
        self.cost = arc.cost if arc else 0

    def end(self):
        """Restituisce il nodo finale del percorso"""
        return self.arc.to_node if self.arc else self.initial

    def arcs(self):
        """Restituisce gli archi del percorso"""
        current = self
        arcs = []
        while current.arc:
            arcs.append(current.arc)
            current = current.initial
        return list(reversed(arcs))

class Search_problem:
    """Classe base per problemi di ricerca"""
    def start_node(self):
        """Restituisce il nodo iniziale"""
        raise NotImplementedError()

    def is_goal(self, node):
        """Verifica se un nodo è un goal"""
        raise NotImplementedError()

    def neighbors(self, node):
        """Restituisce i vicini di un nodo"""
        raise NotImplementedError()

    def heuristic(self, node):
        """Euristica per la ricerca"""
        return 0

class AStarSearcher:
    """Implementazione base di A* Search"""
    def __init__(self, problem):
        self.problem = problem

    def search(self):
        """Esegue la ricerca A*"""
        start = self.problem.start_node()
        frontier = [(start, start.cost + self.problem.heuristic(start))]
        explored = set()

        while frontier:
            # Ordina per costo + euristica
            frontier.sort(key=lambda x: x[1])
            current_path, _ = frontier.pop(0)

            # Verifica goal
            if self.problem.is_goal(current_path):
                return current_path

            # Marca come esplorato
            explored.add(current_path.end())

            # Esplora vicini
            for neighbor in self.problem.neighbors(current_path):
                if neighbor.end() not in explored:
                    new_cost = neighbor.cost + self.problem.heuristic(neighbor)
                    frontier.append((neighbor, new_cost))

        return None

class ItinerarySearch(Search_problem):
    """Problema di ricerca per ottimizzare l'ordine di visita delle attrazioni"""

    def __init__(self, attractions: List[Dict[str, Any]],
                 start_location: tuple,
                 uncertainty_model: UncertaintyModel,
                 available_time: int,
                 evidence: Dict[str, Any] = {}):
        """
        Inizializza il problema di ricerca
        attractions: Lista di dizionari con informazioni sulle attrazioni
        start_location: Coordinate (lat, lon) di partenza
        uncertainty_model: Istanza di UncertaintyModel
        available_time: Tempo disponibile in minuti
        evidence: Evidenze per il modello probabilistico
        """
        self.attractions = attractions
        self.start_location = start_location
        self.uncertainty_model = uncertainty_model
        self.available_time = available_time
        self.evidence = evidence

        # Mappa per le coordinate e i tempi di visita
        self.locations = {}
        self.visit_times = {}
        self.wait_times = {}

        for attr in attractions:
            self.locations[attr['id']] = (attr['lat'], attr['lon'])
            self.visit_times[attr['id']] = attr['visit_time']

            # Calcola tempo di attesa stimato per ogni attrazione
            self.wait_times[attr['id']] = self.uncertainty_model.get_wait_time(evidence)

        # Aggiungi la posizione di partenza
        self.locations['start'] = start_location

        # Calcola il fattore di tempo di viaggio
        self.traffic_factor = self.uncertainty_model.get_travel_time_factor(evidence)

    def start_node(self):
        """Restituisce il nodo iniziale"""
        return Path(self, "start", None)

    def is_goal(self, node):
        """Verifica se un nodo è un goal"""
        # Un percorso è un goal quando visita tutte le attrazioni o quando non
        # è possibile aggiungere altre attrazioni senza superare il tempo disponibile
        visited = set(arc.to_node for arc in node.arcs())
        return len(visited) == len(self.attractions) + 1  # +1 per il nodo di partenza

    def neighbors(self, node):
        """Restituisce i nodi vicini (attrazioni raggiungibili)"""
        neighbors = []

        # Nodo corrente
        current = node.end()

        # Costo accumulato finora (tempo utilizzato)
        current_cost = node.cost

        # Attrazioni già visitate
        visited = set(arc.to_node for arc in node.arcs())

        # Per ogni attrazione non ancora visitata
        for attr in self.attractions:
            attr_id = attr['id']

            if attr_id not in visited:
                # Calcola il tempo di viaggio
                travel_time = self._calculate_travel_time(current, attr_id)

                # Calcola il tempo totale (viaggio + visita + attesa)
                visit_time = self.visit_times[attr_id]
                wait_time = self.wait_times[attr_id]
                total_time = travel_time + visit_time + wait_time

                # Verifica se c'è abbastanza tempo
                if current_cost + total_time <= self.available_time:
                    # Crea un arco al nodo vicino
                    arc = Arc(current, attr_id, total_time)

                    # Crea un nuovo percorso
                    new_path = Path(self, node, arc)

                    neighbors.append(new_path)

        return neighbors

    def heuristic(self, node):
        """Euristica per A*: tempo minimo necessario per visitare le attrazioni rimanenti"""
        # Attrazioni già visitate
        visited = set(arc.to_node for arc in node.arcs())

        # Attrazioni rimanenti
        remaining = [attr for attr in self.attractions if attr['id'] not in visited]

        if not remaining:
            return 0

        # Tempo minimo di visita per le attrazioni rimanenti
        min_visit_time = sum(attr['visit_time'] for attr in remaining)

        # Tempo minimo di attesa
        min_wait_time = sum(self.wait_times[attr['id']] for attr in remaining)

        # Stima ottimistica del tempo di viaggio tra le attrazioni rimanenti
        min_travel_time = 0
        if len(remaining) > 1:
            # Utilizziamo l'euristica MST (Minimum Spanning Tree) semplificata
            # Assumendo che il percorso ottimale sia almeno la lunghezza del MST
            all_distances = []
            for i, attr1 in enumerate(remaining):
                for attr2 in remaining[i + 1:]:
                    dist = geodesic(
                        self.locations[attr1['id']],
                        self.locations[attr2['id']]
                    ).kilometers
                    all_distances.append(dist)

            if all_distances:
                # Approssimazione rapida del MST (non un vero MST)
                all_distances.sort()
                # Distanza totale per connettere tutte le attrazioni
                mst_distance = sum(all_distances[:len(remaining) - 1])
                # Converti in tempo di viaggio
                min_travel_time = mst_distance * 15 * self.traffic_factor

        return min_visit_time + min_wait_time + min_travel_time

    def _calculate_travel_time(self, from_id, to_id):
        """Calcola il tempo di viaggio tra due attrazioni"""
        if from_id == "start" or to_id == "start":
            # Tempo di default dal punto di partenza
            return 20

        # Calcola la distanza con geopy
        from_loc = self.locations[from_id]
        to_loc = self.locations[to_id]

        distance = geodesic(from_loc, to_loc).kilometers

        # Converti in tempo di viaggio (minuti)
        return distance * 15 * self.traffic_factor