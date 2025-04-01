# enhanced_reasoning_module.py
from lib.logicRelation import KB, Var, Atom, Clause, unify, apply
from src.data.data_manager import get_all_attractions_list, load_attractions


class DatalogReasoner:
    """Reasoner basato su Datalog per il sistema turistico"""

    def __init__(self):
        """Inizializza il reasoner Datalog"""
        # Crea la knowledge base
        self.kb = KB([])

        # Carica i dati
        attractions_df = load_attractions()
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

    def _load_tourist_data(self):
        """Carica i dati dei turisti nella knowledge base"""
        # Carica i dati dei turisti
        tourists_df = load_tourists()

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
        """Trova attrazioni in base agli interessi"""
        attraction_ids = set()

        for interest in interests:
            results = self.kb.ask_all([Atom('has_category', [Var('X'), interest])])
            attraction_ids.update([result['X'] for result in results])

        return list(attraction_ids)