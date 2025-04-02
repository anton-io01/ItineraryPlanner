from typing import Dict, List, Any, Union

class Variable:
    """Rappresenta una variabile in un modello probabilistico"""
    def __init__(self, name: str, values: List[str]):
        self.name = name
        self.values = values

class Prob:
    """Rappresenta una distribuzione di probabilità condizionale"""
    def __init__(self, variable: Variable, parents: List[Variable], probabilities: Dict[Union[str, tuple], Dict[str, float]]):
        self.variable = variable
        self.parents = parents
        self.probabilities = probabilities

class BeliefNetwork:
    """Rappresenta una Belief Network probabilistica"""
    def __init__(self, name: str, variables: set, factors: set):
        self.name = name
        self.variables = variables
        self.factors = factors

class ProbRC:
    """Implementazione semplificata di Recursive Conditioning per inferenza"""
    def __init__(self, belief_network):
        self.bn = belief_network

    def query(self, variable, evidence: Dict[str, str] = {}):
        """
        Query probabilistica semplificata
        Restituisce una distribuzione di probabilità approssimata
        """
        # Implementazione base - da raffinare
        default_distribution = {val: 1.0/len(variable.values) for val in variable.values}
        return default_distribution

class UncertaintyModel:
    """Modello di incertezza basato su Belief Network"""

    def __init__(self):
        """Inizializza il modello di incertezza"""
        # Definisci le variabili
        self.time_of_day = Variable("TimeOfDay", ["morning", "afternoon", "evening"])
        self.day_of_week = Variable("DayOfWeek", ["weekday", "weekend"])
        self.weather = Variable("Weather", ["sunny", "cloudy", "rainy"])
        self.traffic = Variable("Traffic", ["light", "moderate", "heavy"])
        self.crowd = Variable("Crowd", ["low", "medium", "high"])

        # Definisci le distribuzioni di probabilità
        # P(TimeOfDay)
        f_time = Prob(self.time_of_day, [], {"morning": 0.3, "afternoon": 0.5, "evening": 0.2})

        # P(DayOfWeek)
        f_day = Prob(self.day_of_week, [], {"weekday": 0.7, "weekend": 0.3})

        # P(Weather)
        f_weather = Prob(self.weather, [], {"sunny": 0.6, "cloudy": 0.3, "rainy": 0.1})

        # P(Traffic | TimeOfDay, DayOfWeek)
        traffic_cpt = {
            ("morning", "weekday"): {"light": 0.2, "moderate": 0.3, "heavy": 0.5},
            ("morning", "weekend"): {"light": 0.6, "moderate": 0.3, "heavy": 0.1},
            ("afternoon", "weekday"): {"light": 0.3, "moderate": 0.4, "heavy": 0.3},
            ("afternoon", "weekend"): {"light": 0.4, "moderate": 0.4, "heavy": 0.2},
            ("evening", "weekday"): {"light": 0.3, "moderate": 0.3, "heavy": 0.4},
            ("evening", "weekend"): {"light": 0.5, "moderate": 0.3, "heavy": 0.2}
        }
        f_traffic = Prob(self.traffic, [self.time_of_day, self.day_of_week], traffic_cpt)

        # P(Crowd | TimeOfDay, DayOfWeek)
        crowd_cpt = {
            ("morning", "weekday"): {"low": 0.6, "medium": 0.3, "high": 0.1},
            ("morning", "weekend"): {"low": 0.3, "medium": 0.4, "high": 0.3},
            ("afternoon", "weekday"): {"low": 0.4, "medium": 0.4, "high": 0.2},
            ("afternoon", "weekend"): {"low": 0.2, "medium": 0.3, "high": 0.5},
            ("evening", "weekday"): {"low": 0.5, "medium": 0.3, "high": 0.2},
            ("evening", "weekend"): {"low": 0.3, "medium": 0.4, "high": 0.3}
        }
        f_crowd = Prob(self.crowd, [self.time_of_day, self.day_of_week], crowd_cpt)

        # Crea la Belief Network
        variables = {self.time_of_day, self.day_of_week, self.weather,
                     self.traffic, self.crowd}
        factors = {f_time, f_day, f_weather, f_traffic, f_crowd}

        self.bn = BeliefNetwork("Tourism Uncertainty Model", variables, factors)

        # Inferenza usando ProbRC (recursive conditioning)
        self.inference = ProbRC(self.bn)

    def get_traffic_distribution(self, evidence: Dict[str, str] = {}):
        """Calcola la distribuzione del traffico date le evidenze"""
        return self.inference.query(self.traffic, evidence)

    def get_crowd_distribution(self, evidence: Dict[str, str] = {}):
        """Calcola la distribuzione dell'affluenza date le evidenze"""
        return self.inference.query(self.crowd, evidence)

    def get_travel_time_factor(self, evidence: Dict[str, str] = {}):
        """Calcola un fattore di tempo di viaggio basato sul traffico"""
        traffic_dist = self.get_traffic_distribution(evidence)

        # Fattori moltiplicativi per ogni livello di traffico
        factors = {"light": 0.8, "moderate": 1.0, "heavy": 1.5}

        # Calcola il fattore atteso
        expected_factor = sum(prob * factors[level] for level, prob in traffic_dist.items())

        return expected_factor

    def get_wait_time(self, evidence: Dict[str, str] = {}):
        """Calcola il tempo di attesa atteso basato sull'affluenza"""
        crowd_dist = self.get_crowd_distribution(evidence)

        # Tempo di attesa in minuti per ogni livello di affluenza
        wait_times = {"low": 5, "medium": 15, "high": 30}

        # Calcola il tempo di attesa atteso
        expected_wait = sum(prob * wait_times[level] for level, prob in crowd_dist.items())

        return expected_wait