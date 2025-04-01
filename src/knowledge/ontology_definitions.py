# ontology_definitions.py
from owlready2 import *
import os


def create_ontology():
    """Crea e restituisce l'ontologia di base per il sistema turistico di Roma"""
    # Crea un'ontologia di base
    onto = get_ontology("http://roma-tourism.org/onto.owl")

    with onto:
        # Definizione delle classi principali
        class Attraction(Thing): pass

        class Museum(Attraction): pass

        class Monument(Attraction): pass

        class ArchaeologicalSite(Attraction): pass

        class Restaurant(Attraction): pass

        class Park(Attraction): pass

        # Categorie di interesse
        class Interest(Thing): pass

        class Art(Interest): pass

        class History(Interest): pass

        class Nature(Interest): pass

        class Entertainment(Interest): pass

        class Tourist(Thing): pass

        class Neighborhood(Thing): pass

        # Proprietà degli oggetti
        class hasCategory(ObjectProperty):
            domain = [Attraction]
            range = [Interest]

        class hasInterest(ObjectProperty):
            domain = [Tourist]
            range = [Interest]

        class isNearTo(ObjectProperty):
            domain = [Attraction]
            range = [Attraction]
            symmetric = True

        class isLocatedIn(ObjectProperty):
            domain = [Attraction]
            range = [Neighborhood]

        # Proprietà dei dati
        class hasName(DataProperty):
            domain = [Attraction, Tourist, Neighborhood]
            range = [str]

        class hasLatitude(DataProperty):
            domain = [Attraction]
            range = [float]

        class hasLongitude(DataProperty):
            domain = [Attraction]
            range = [float]

        class hasAverageRating(DataProperty):
            domain = [Attraction]
            range = [float]

        class hasCost(DataProperty):
            domain = [Attraction]
            range = [float]

        class hasEstimatedVisitTime(DataProperty):
            domain = [Attraction]
            range = [int]

        class hasAvailableTime(DataProperty):
            domain = [Tourist]
            range = [int]

        class hasInterestScore(DataProperty):
            domain = [Tourist, Interest]
            range = [int]

        # Regole SWRL (saranno implementate tramite Datalog)

    return onto