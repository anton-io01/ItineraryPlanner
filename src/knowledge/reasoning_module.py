from lib.logicRelation import KB, Var, Atom, Clause, unify, apply
from src.data.data_manager import load_attractions, load_tourists, get_all_attractions_list, get_tourist_profile, \
    get_attraction_details
from geopy.distance import geodesic


class DatalogReasoner:
    """Reasoner basato su Datalog per il sistema turistico"""

    def __init__(self):
        """Inizializza il reasoner Datalog"""
        # Crea la knowledge base
        self.kb = KB([])
        print("Inizializzazione reasoner Datalog...")

        # Carica i dati
        attractions_df = load_attractions()
        self.attractions_df = attractions_df  # Salva il DataFrame per usi futuri
        attractions_list = get_all_attractions_list(attractions_df)

        # Variabili Datalog
        X = Var('X')
        Y = Var('Y')
        Z = Var('Z')
        Cost = Var('Cost')
        Rating = Var('Rating')

        # Aggiungi fatti alla knowledge base
        for attr in attractions_list:
            attr_id = str(attr['id_attrazione'])

            # Aggiungi fatto: attraction(attr_id)
            self.kb.add_clause(Clause(Atom('attraction', [attr_id])))

            # Aggiungi fatto: has_cost(attr_id, cost)
            self.kb.add_clause(Clause(Atom('has_cost', [attr_id, attr['costo']])))

            # Aggiungi fatto: has_rating(attr_id, rating)
            self.kb.add_clause(Clause(Atom('has_rating', [attr_id, attr['recensione_media']])))

            # Aggiungi fatto: has_location(attr_id, lat, lon)
            self.kb.add_clause(Clause(Atom('has_location', [attr_id, attr['latitudine'], attr['longitudine']])))

            # Aggiungi categorie in base alla descrizione
            if 'arte' in attr['descrizione'].lower():
                self.kb.add_clause(Clause(Atom('has_category', [attr_id, 'arte'])))
            if 'storia' in attr['descrizione'].lower():
                self.kb.add_clause(Clause(Atom('has_category', [attr_id, 'storia'])))
            if 'natura' in attr['descrizione'].lower():
                self.kb.add_clause(Clause(Atom('has_category', [attr_id, 'natura'])))
            if 'divertimento' in attr['descrizione'].lower():
                self.kb.add_clause(Clause(Atom('has_category', [attr_id, 'divertimento'])))

        # Aggiungi regole

        # Un'attrazione è considerata di alto rating se rating >= 4
        self.kb.add_clause(Clause(
            Atom('high_rated', [X]),
            [Atom('attraction', [X]), Atom('has_rating', [X, Rating]), Atom('lt', [4.0, Rating])]
        ))

        # Un'attrazione è economica se costo <= 15
        self.kb.add_clause(Clause(
            Atom('budget_friendly', [X]),
            [Atom('attraction', [X]), Atom('has_cost', [X, Cost]), Atom('lt', [Cost, 15.0])]
        ))

        # Un'attrazione è consigliata se ha un buon rating e un costo contenuto
        self.kb.add_clause(Clause(
            Atom('recommended', [X]),
            [Atom('attraction', [X]), Atom('high_rated', [X]), Atom('budget_friendly', [X])]
        ))

        # Un'attrazione è adatta a un turista se ha una categoria che piace al turista
        self.kb.add_clause(Clause(
            Atom('suitable_for', [X, Z]),
            [Atom('attraction', [X]), Atom('tourist_likes', [Z, Y]), Atom('has_category', [X, Y])]
        ))

        # Carica i dati dei turisti
        self._load_tourist_data()

        print("Reasoner Datalog inizializzato con successo")

    def _load_tourist_data(self):
        """Carica i dati dei turisti nella knowledge base"""
        # Carica i dati dei turisti
        tourists_df = load_tourists()
        self.tourists_df = tourists_df  # Salva il DataFrame per usi futuri

        if tourists_df is not None:
            for _, row in tourists_df.iterrows():
                tourist_id = str(row['id_turista'])

                # Aggiungi interessi basati sui punteggi
                if row['arte'] > 5:  # Soglia per considerare un interesse rilevante
                    self.kb.add_clause(Clause(Atom('tourist_likes', [tourist_id, 'arte'])))

                if row['storia'] > 5:
                    self.kb.add_clause(Clause(Atom('tourist_likes', [tourist_id, 'storia'])))

                if row['natura'] > 5:
                    self.kb.add_clause(Clause(Atom('tourist_likes', [tourist_id, 'natura'])))

                if row['divertimento'] > 5:
                    self.kb.add_clause(Clause(Atom('tourist_likes', [tourist_id, 'divertimento'])))

    def find_high_rated_attractions(self):
        """Trova attrazioni con valutazione alta"""
        results = self.kb.ask_all([Atom('high_rated', [Var('X')])])
        return [result['X'] for result in results]

    def find_budget_friendly_attractions(self):
        """Trova attrazioni economiche"""
        results = self.kb.ask_all([Atom('budget_friendly', [Var('X')])])
        return [result['X'] for result in results]

    def find_recommended_attractions(self):
        """Trova attrazioni consigliate (alto rating e budget friendly)"""
        results = self.kb.ask_all([Atom('recommended', [Var('X')])])
        return [result['X'] for result in results]

    def find_suitable_attractions(self, tourist_id):
        """Trova attrazioni adatte a un turista specifico"""
        results = self.kb.ask_all([Atom('suitable_for', [Var('X'), tourist_id])])
        return [result['X'] for result in results]

    def find_attractions_by_interest(self, interests):
        """Trova attrazioni in base agli interessi con ricerca flessibile"""
        attraction_ids = set()

        # Debug - mostra interessi ricevuti
        print(f"Ricerca attrazioni per interessi: {interests}")

        # Se non ci sono interessi, restituisci tutte le attrazioni
        if not interests:
            print("Nessun interesse specificato, restituisco tutte le attrazioni")
            return [str(id) for id in self.attractions_df['id_attrazione']]

        # Normalizza tutti gli interessi a minuscolo per la ricerca
        normalized_interests = [interest.lower() for interest in interests]

        # Mappatura flessibile degli interessi a termini di ricerca
        interest_terms = {
            'arte': ['arte', 'museo', 'galleria', 'cappella', 'basilica'],
            'storia': ['storia', 'antico', 'storico', 'romano', 'imperiale', 'foro', 'rovina'],
            'natura': ['natura', 'villa', 'parco', 'giardino', 'verde'],
            'divertimento': ['divertimento', 'svago', 'parco', 'world', 'bambini']
        }

        # Espandi gli interessi specificati ai termini di ricerca
        search_terms = set()
        for interest in normalized_interests:
            if interest in interest_terms:
                search_terms.update(interest_terms[interest])
            else:
                search_terms.add(interest)

        print(f"Termini di ricerca: {search_terms}")

        # Cerca nelle descrizioni
        for _, attr in self.attractions_df.iterrows():
            description = attr['descrizione'].lower()
            name = attr['nome'].lower()

            # Verifica se i termini di ricerca sono presenti nella descrizione o nel nome
            for term in search_terms:
                if term in description or term in name:
                    attraction_ids.add(str(attr['id_attrazione']))
                    break

        print(f"Trovate {len(attraction_ids)} attrazioni basate su descrizioni e nomi")

        # Se non è stato trovato nulla, restituisci le attrazioni con il rating più alto
        if not attraction_ids:
            print(
                "Nessuna attrazione trovata in base agli interessi, restituisco le attrazioni con valutazioni migliori")
            top_attractions = self.attractions_df.sort_values('recensione_media', ascending=False).head(5)
            attraction_ids = set(str(id) for id in top_attractions['id_attrazione'])

        return list(attraction_ids)

    def get_tourist_by_id(self, tourist_id):
        """
        Restituisce un oggetto che rappresenta un turista con i suoi attributi
        """
        # Carica il profilo del turista
        tourist_profile = get_tourist_profile(self.tourists_df, tourist_id)

        if not tourist_profile:
            return None

        # Crea un oggetto "simile" a quello restituito dall'ontologia
        class TouristInfo:
            def __init__(self, profile):
                self.id = str(profile['id_turista'])
                self.hasAvailableTime = [profile['tempo']]
                self.hasInterest = []

                if profile['arte'] > 5:
                    self.hasInterest.append('arte')
                if profile['storia'] > 5:
                    self.hasInterest.append('storia')
                if profile['natura'] > 5:
                    self.hasInterest.append('natura')
                if profile['divertimento'] > 5:
                    self.hasInterest.append('divertimento')

        return TouristInfo(tourist_profile)

    def search_one(self, iri=None):
        """
        Simula la funzione search_one dell'ontologia
        Cerca un'attrazione per ID o per nome
        """
        if not iri:
            return None

        try:
            # Gestisci il caso di IRI nel formato "attraction_X"
            if iri.startswith('attraction_'):
                try:
                    # Estrai l'ID numerico
                    attr_id = int(iri.split('_')[1])
                    # Cerca l'attrazione con questo ID
                    attr = self.attractions_df[self.attractions_df['id_attrazione'] == attr_id]
                    if not attr.empty:
                        # Crea un oggetto attrazione
                        class AttractionInfo:
                            def __init__(self, details):
                                self.id = str(details['id_attrazione'])
                                self.name = details['nome']
                                self.hasLatitude = [details['latitudine']]
                                self.hasLongitude = [details['longitudine']]
                                self.hasEstimatedVisitTime = [details['tempo_visita']]
                                self.hasAverageRating = [details['recensione_media']]
                                self.hasCategory = []

                                # Aggiungi categorie in base alla descrizione
                                desc = details['descrizione'].lower()
                                if 'arte' in desc:
                                    self.hasCategory.append('arte')
                                if 'storia' in desc:
                                    self.hasCategory.append('storia')
                                if 'natura' in desc:
                                    self.hasCategory.append('natura')
                                if 'divertimento' in desc:
                                    self.hasCategory.append('divertimento')

                        return AttractionInfo(attr.iloc[0])
                except (IndexError, ValueError) as e:
                    print(f"Errore nell'elaborazione dell'IRI {iri}: {e}")
                    return None

            # Gestisci il caso di ricerca per nome (con o senza *)
            if iri.startswith('*'):
                # Rimuovi il carattere '*'
                name = iri[1:]
            else:
                name = iri

            # Prova a trovare un'attrazione con questo nome
            match = self.attractions_df[self.attractions_df['nome'] == name]
            if not match.empty:
                # Crea un oggetto attrazione
                class AttractionInfo:
                    def __init__(self, details):
                        self.id = str(details['id_attrazione'])
                        self.name = details['nome']
                        self.hasLatitude = [details['latitudine']]
                        self.hasLongitude = [details['longitudine']]
                        self.hasEstimatedVisitTime = [details['tempo_visita']]
                        self.hasAverageRating = [details['recensione_media']]
                        self.hasCategory = []

                        # Aggiungi categorie in base alla descrizione
                        desc = details['descrizione'].lower()
                        if 'arte' in desc:
                            self.hasCategory.append('arte')
                        if 'storia' in desc:
                            self.hasCategory.append('storia')
                        if 'natura' in desc:
                            self.hasCategory.append('natura')
                        if 'divertimento' in desc:
                            self.hasCategory.append('divertimento')

                return AttractionInfo(match.iloc[0])
        except Exception as e:
            print(f"Errore nella ricerca dell'attrazione: {e}")

        return None

    def get_attractions_near(self, attraction_id, max_distance=1.0):
        """
        Trova attrazioni vicine a quella specificata
        """
        # Ottieni i dettagli dell'attrazione di origine
        source_attr = get_attraction_details(self.attractions_df, attraction_id)

        if not source_attr:
            return []

        source_coords = (source_attr['latitudine'], source_attr['longitudine'])

        # Trova attrazioni vicine
        nearby_attractions = []
        for _, attr in self.attractions_df.iterrows():
            if attr['id_attrazione'] != int(attraction_id):
                attr_coords = (attr['latitudine'], attr['longitudine'])
                distance = geodesic(source_coords, attr_coords).kilometers

                if distance <= max_distance:
                    nearby_attractions.append(str(attr['id_attrazione']))

        return nearby_attractions

    @property
    def onto(self):
        """
        Restituisce un riferimento a se stesso per compatibilità
        (in originale restituirebbe l'oggetto ontologia)
        """

        class MockOntology:
            def __init__(self, reasoner):
                self.reasoner = reasoner

                class AttractionClass:
                    def __init__(self, r):
                        self.r = r

                    def instances(self):
                        """Restituisce tutte le istanze di attrazioni"""
                        result = []
                        print(f"Cercando attrazioni nel DataFrame di lunghezza {len(self.r.attractions_df)}")
                        for attr_id in range(1, len(self.r.attractions_df) + 1):
                            print(f"Cercando attraction_{attr_id}")
                            attr = self.r.search_one(f"attraction_{attr_id}")
                            if attr:
                                print(f"Trovata attrazione: {attr.name}")
                                result.append(attr)
                            else:
                                print(f"Attrazione attraction_{attr_id} non trovata")
                        return result

                self.Attraction = AttractionClass(reasoner)

            def search_one(self, iri=None):
                """
                Delega la ricerca al reasoner
                """
                return self.reasoner.search_one(iri)

        return MockOntology(self)