# reasoning_module.py
from owlready2 import *
from src.knowledge.ontology_definitions import create_ontology
from src.knowledge.ontology_rules import create_logic_rules
from geopy.distance import geodesic


class OntologyReasoner:
    """Classe per il ragionamento e le query sull'ontologia"""

    def __init__(self, ontology_path="ontologies/roma_tourism.owl"):
        """Inizializza il reasoner con l'ontologia specificata"""
        try:
            # Carica l'ontologia esistente
            self.onto = get_ontology(ontology_path).load()
        except Exception as e:
            print(f"Errore nel caricamento dell'ontologia: {e}")
            # Crea una nuova ontologia
            self.onto = create_ontology()

        # Inizializza il reasoner
        with self.onto:
            sync_reasoner()

        # Carica le regole datalog
        self.logic_kb = create_logic_rules()

    def find_attractions_by_interest(self, interest_names):
        """Trova attrazioni in base agli interessi"""
        matching_attractions = []

        for attraction in self.onto.Attraction.instances():
            # Verifica se l'attrazione ha almeno una delle categorie richieste
            for interest_name in interest_names:
                if hasattr(self.onto, interest_name):
                    interest_class = getattr(self.onto, interest_name)
                    # Verifica se qualche categoria dell'attrazione è del tipo di interesse
                    if any(isinstance(category, interest_class) for category in attraction.hasCategory):
                        matching_attractions.append(attraction)
                        break

        return matching_attractions

    def find_budget_friendly_attractions(self, max_cost=15):
        """Trova attrazioni economiche usando Datalog"""
        # Prepara la query Datalog
        results = []
        for answer in self.logic_kb.ask([self.logic_kb.Atom('budget_friendly', ['X'])]):
            attr_id = answer['X']
            attraction = self.get_attraction_by_id(attr_id)
            if attraction:
                results.append(attraction)

        return results

    def find_high_rated_attractions(self):
        """Trova attrazioni con valutazione alta usando Datalog"""
        results = []
        for answer in self.logic_kb.ask([self.logic_kb.Atom('high_rated', ['X'])]):
            attr_id = answer['X']
            attraction = self.get_attraction_by_id(attr_id)
            if attraction:
                results.append(attraction)

        return results

    def find_suitable_attractions(self, tourist_id):
        """Trova attrazioni adatte a un turista specifico usando Datalog"""
        results = []
        for answer in self.logic_kb.ask([self.logic_kb.Atom('suitable_for', ['X', tourist_id])]):
            attr_id = answer['X']
            attraction = self.get_attraction_by_id(attr_id)
            if attraction:
                results.append(attraction)

        return results

    def find_nearby_attractions(self, attraction_id, max_distance=1.0):
        """Trova attrazioni vicine a una data attrazione"""
        source_attraction = self.get_attraction_by_id(attraction_id)
        if not source_attraction:
            return []

        nearby = []
        src_coords = (source_attraction.hasLatitude[0], source_attraction.hasLongitude[0])

        for attraction in self.onto.Attraction.instances():
            if attraction != source_attraction and attraction.hasLatitude and attraction.hasLongitude:
                dst_coords = (attraction.hasLatitude[0], attraction.hasLongitude[0])
                distance = geodesic(src_coords, dst_coords).kilometers

                if distance <= max_distance:
                    nearby.append((attraction, distance))

        # Ordina per distanza
        nearby.sort(key=lambda x: x[1])
        return [attr for attr, _ in nearby]

    def get_attraction_by_id(self, attraction_id):
        """Recupera un'attrazione tramite ID"""
        attraction_name = f"attraction_{attraction_id}"
        return self.onto.search_one(iri=f"*{attraction_name}")

    def get_tourist_by_id(self, tourist_id):
        """Recupera un profilo turistico tramite ID"""
        tourist_name = f"tourist_{tourist_id}"
        return self.onto.search_one(iri=f"*{tourist_name}")

    def recommend_attractions_for_tourist(self, tourist_id, max_attractions=5):
        """Raccomanda attrazioni per un turista specifico"""
        tourist = self.get_tourist_by_id(tourist_id)
        if not tourist:
            return []

        # Ottieni interessi del turista
        interest_classes = [type(interest) for interest in tourist.hasInterest]

        matching_attractions = []
        for attraction in self.onto.Attraction.instances():
            # Verifica se l'attrazione ha categorie che corrispondono agli interessi
            for category in attraction.hasCategory:
                if type(category) in interest_classes:
                    matching_attractions.append(attraction)
                    break

        # Ordina per valutazione e restituisci i migliori
        return sorted(
            matching_attractions,
            key=lambda a: a.hasAverageRating[0] if a.hasAverageRating else 0,
            reverse=True
        )[:max_attractions]