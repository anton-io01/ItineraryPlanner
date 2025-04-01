import os
from owlready2 import *
from geopy.distance import geodesic
from src.data.data_manager import load_attractions, load_tourists, get_all_attractions_list
from src.knowledge.ontology_definitions import create_ontology

def populate_ontology():
    """Popola l'ontologia con istanze dai dati CSV e OpenStreetMap"""
    # Carica o crea l'ontologia
    onto = create_ontology()

    # Carica i dati CSV
    attractions_df = load_attractions()
    tourists_df = load_tourists()
    attractions_list = get_all_attractions_list(attractions_df)

    # Quartieri principali di Roma
    neighborhoods = {
        "centro_storico": {"name": "Centro Storico", "coords": (41.9027, 12.4963)},
        "vaticano": {"name": "Vaticano", "coords": (41.9022, 12.4539)},
        "trastevere": {"name": "Trastevere", "coords": (41.8891, 12.4658)},
        "colosseo": {"name": "Colosseo", "coords": (41.8902, 12.4924)},
        "villa_borghese": {"name": "Villa Borghese", "coords": (41.9137, 12.4919)}
    }

    # Crea i quartieri nell'ontologia
    neighborhood_instances = {}
    for id, data in neighborhoods.items():
        neigh = onto.Neighborhood(id)
        neigh.hasName = [data["name"]]
        neighborhood_instances[id] = neigh

    # Mappa categorie CSV a classi dell'ontologia
    category_mapping = {
        'museo': onto.Museum,
        'monumento': onto.Monument,
        'sito_archeologico': onto.ArchaeologicalSite,
        'ristorante': onto.Restaurant,
        'parco': onto.Park
    }

    # Funzione per determinare il quartiere più vicino
    def get_nearest_neighborhood(lat, lon):
        min_dist = float('inf')
        nearest = None

        for id, data in neighborhoods.items():
            dist = geodesic((lat, lon), data["coords"]).kilometers
            if dist < min_dist:
                min_dist = dist
                nearest = neighborhood_instances[id]

        return nearest

    # Popola le attrazioni
    attractions_instances = {}
    for attraction in attractions_list:
        # Determina la classe appropriata
        category = attraction['categoria'].lower()
        attraction_class = category_mapping.get(category, onto.Attraction)

        # Crea l'istanza
        attr_id = str(attraction['id_attrazione'])
        attr_instance = attraction_class(f"attraction_{attr_id}")
        attractions_instances[attr_id] = attr_instance

        # Aggiungi proprietà di base
        attr_instance.hasName = [attraction['nome']]
        attr_instance.hasLatitude = [float(attraction['latitudine'])]
        attr_instance.hasLongitude = [float(attraction['longitudine'])]
        attr_instance.hasAverageRating = [float(attraction['recensione_media'])]
        attr_instance.hasCost = [float(attraction['costo'])]
        attr_instance.hasEstimatedVisitTime = [int(attraction['tempo_visita'])]

        # Assegna il quartiere più vicino
        attr_instance.isLocatedIn = [get_nearest_neighborhood(
            attraction['latitudine'], attraction['longitudine']
        )]

        # Assegna categorie di interesse basate sulla descrizione
        descrizione = attraction['descrizione'].lower()
        if 'arte' in descrizione:
            attr_instance.hasCategory.append(onto.Art())
        if 'storia' in descrizione:
            attr_instance.hasCategory.append(onto.History())
        if 'natura' in descrizione:
            attr_instance.hasCategory.append(onto.Nature())
        if 'divertimento' in descrizione:
            attr_instance.hasCategory.append(onto.Entertainment())

    # Popola i turisti
    for _, row in tourists_df.iterrows():
        tourist_id = str(row['id_turista'])
        tourist = onto.Tourist(f"tourist_{tourist_id}")

        # Proprietà di base
        tourist.hasName = [f"Turista {tourist_id}"]
        tourist.hasAvailableTime = [int(row['tempo'])]

        # Aggiungi interessi basati sui punteggi
        if row['arte'] > 0:
            interest = onto.Art()
            tourist.hasInterest.append(interest)
            tourist.hasInterestScore = [int(row['arte'])]

        if row['storia'] > 0:
            interest = onto.History()
            tourist.hasInterest.append(interest)
            tourist.hasInterestScore = [int(row['storia'])]

        if row['natura'] > 0:
            interest = onto.Nature()
            tourist.hasInterest.append(interest)
            tourist.hasInterestScore = [int(row['natura'])]

        if row['divertimento'] > 0:
            interest = onto.Entertainment()
            tourist.hasInterest.append(interest)
            tourist.hasInterestScore = [int(row['divertimento'])]

    # Salva l'ontologia
    os.makedirs("ontologies", exist_ok=True)
    onto.save("ontologies/roma_tourism.owl")
    return onto