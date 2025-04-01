# ontology_rules.py
from lib.logicRelation import Atom, Var, KB, Clause
from src.data.data_manager import get_all_attractions_list, load_attractions


def create_logic_rules():
    """Crea regole Datalog per il sistema turistico"""
    X = Var('X')
    Y = Var('Y')
    Z = Var('Z')
    Cost = Var('Cost')
    Rating = Var('Rating')

    # Definizione dei predicati
    def attraction(a):
        return Atom('attraction', [a])

    def high_rated(a):
        return Atom('high_rated', [a])

    def budget_friendly(a):
        return Atom('budget_friendly', [a])

    def has_cost(a, c):
        return Atom('has_cost', [a, c])

    def has_rating(a, r):
        return Atom('has_rating', [a, r])

    def recommended(a):
        return Atom('recommended', [a])

    def near(a, b):
        return Atom('near', [a, b])

    def has_category(a, c):
        return Atom('has_category', [a, c])

    def tourist_likes(t, c):
        return Atom('tourist_likes', [t, c])

    def suitable_for(a, t):
        return Atom('suitable_for', [a, t])

    # Crea la knowledge base con le regole logiche
    kb = KB([
        # Un'attrazione è consigliata se ha un buon rating e un costo contenuto
        Clause(recommended(X), [attraction(X), high_rated(X), budget_friendly(X)]),

        # Un'attrazione è considerata di alto rating se rating >= 4
        Clause(high_rated(X), [attraction(X), has_rating(X, Rating), Atom('lt', [4, Rating])]),

        # Un'attrazione è economica se costo <= 15
        Clause(budget_friendly(X), [attraction(X), has_cost(X, Cost), Atom('lt', [Cost, 15])]),

        # Un'attrazione è adatta a un turista se ha una categoria che piace al turista
        Clause(suitable_for(X, Z), [attraction(X), tourist_likes(Z, Y), has_category(X, Y)]),

        # Relazione di transitività per la vicinanza
        Clause(near(X, Z), [near(X, Y), near(Y, Z)])
    ])

    # Aggiungi fatti dalla base di dati
    attractions_df = load_attractions()
    attractions_list = get_all_attractions_list(attractions_df)

    for attr in attractions_list:
        attr_id = str(attr['id_attrazione'])
        # Aggiungi fatti per ogni attrazione
        kb.add_clause(Clause(attraction(attr_id)))
        kb.add_clause(Clause(has_cost(attr_id, attr['costo'])))
        kb.add_clause(Clause(has_rating(attr_id, attr['recensione_media'])))

        # Aggiungi categorie
        if 'arte' in attr['descrizione'].lower():
            kb.add_clause(Clause(has_category(attr_id, 'arte')))
        if 'storia' in attr['descrizione'].lower():
            kb.add_clause(Clause(has_category(attr_id, 'storia')))
        if 'natura' in attr['descrizione'].lower():
            kb.add_clause(Clause(has_category(attr_id, 'natura')))
        if 'divertimento' in attr['descrizione'].lower():
            kb.add_clause(Clause(has_category(attr_id, 'divertimento')))

    return kb