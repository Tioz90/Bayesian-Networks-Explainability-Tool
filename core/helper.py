import pandas as pd

from range_key_dict import RangeKeyDict

# used when formatting in f-strings
WIDTH = 0
PRECISION = 5

prob_dict = RangeKeyDict( {(0, 0.2):   "highly unlikely",
                           (0.2, 0.3): "very unlikely",
                           (0.3, 0.4): "unlikely",
                           (0.4, 0.5): "not plausibly",
                           (0.5, 0.6): "plausibly",
                           (0.6, 0.7): "possibly",
                           (0.7, 0.8): "likely",
                           (0.8, 0.9): "very likely",
                           (0.9, 1):   "highly likely",
                           (1, 1.1):   "certain"
                           } )


def read_dataset(path, type, sheet=None, names=None):
    if type == "excel":
        df = pd.read_excel( path, sheet_name=sheet )
    if type == "txt" or type == "csv":
        df = pd.read_csv( path, names, header=None )

    return df


def correlation(df, X, Y):
    return df[X].corr(df[Y])


def clean_dataset(df):
    # remove datapoints containing '?'
    index = df.isin( ['?'] ).any( axis=1 )
    df = df[~index]

    # TODO mettere un "drop n/a"?

    return df


def categorical_dataset(df):
    # check coerenza dati categorici e binarizzazione
    for col in list( df.columns ):
        if df[col].dtype == "object":
            # df[col] = pd.Categorical(df[col])
            # df[col] = pd.Categorical(df[col], categories=df[col].unique())
            # print("Values of category {} \n{}\n\n".format(col, df[col].unique())) # print values
            df[col] = pd.Categorical( df[col],
                                      categories=df[col].unique() ).codes  # convert object to categorical codes
        if df[col].dtype == "float64" or df[col].dtype == "int64":
            # df[col] = pd.cut(df[col].astype(int), 2, labels=[0, 1]) # 0: 195, 1: 82
            df[col] = pd.qcut( df[col].astype( int ), 2, labels=False,
                               duplicates='drop' )  # binarise numerical variables # TODO attenzione a come si categorizza

    return df


def inspect_dataset(df):
    print( "\n" )
    print( df.head() )  # first five data
    print( "Types \n{}".format( df.dtypes ) )


def map_codes_states(df_codes, df_names):
    codes_to_states_dict = {}

    for col in list( df_codes.columns ):
        codes_to_states_dict[col] = {}
        for i in df_codes[col].unique():
            codes_to_states_dict[col][i] = df_names[col].unique()[i]

    states_to_codes_dict = {}

    for col in list( df_names.columns ):
        states_to_codes_dict[col] = {}
        for key, value in codes_to_states_dict[col].items():
            states_to_codes_dict[col][value] = key

    return (codes_to_states_dict, states_to_codes_dict)


# print a list in a more beautiful way
def printl(list):
    str = ", ".join( repr( e ) for e in list )

    return str


# given dataframe create a list of dictionaries of its column with eventual default ones
def make_list_of_dict(df, defaults=None):
    list = []

    for col in df.columns:
        if defaults and col in defaults:
            list.append(
                    {
                        "name":    col,
                        'checked': True
                    }
            )
        else:
            list.append(
                    {
                        "name": col
                    }
            )

    return list


def question_prompt(names, questions):
    # setup PyInquirer
    from PyInquirer import prompt

    choices = {}

    for i, name in enumerate( names ):
        while True:
            choice = prompt( questions[i] )
            try:
                choices[name] = choice[name]
                break
            except KeyError:
                continue

    return choices


def probability(num1, num2):
    try:
        return (num1 / num2)
    except ZeroDivisionError:
        return None


def print_accuracy(y_true, y_pred, var_list):
    from sklearn.metrics import accuracy_score
    from colorama import Fore

    for var in var_list:
        print( Fore.GREEN +
               f"Accuracy of prediction for {var}: {accuracy_score( y_true.loc[:, var], y_pred.loc[:, var] ):{WIDTH}.{PRECISION}}" )


def print_probability_table(states, code_to_value_map):
    from prettytable import PrettyTable

    table = PrettyTable()
    table.title = "Posteriors"
    table.field_names = ["State", "Value", "Probability", "Entropy", "Efficiency"]

    try:
        for state in states:
            table.add_row(
                    [state["state"], code_to_value_map[state["state"]][state["value"]],
                     round( state["probability"], 2 ), round( state["entropy"], 2 ), round( state["efficiency"], 2 )] )
        print( table )
    except TypeError:
        return


# print entropy of each variable in list in ascending order
def print_entropies_table(df, list=None):
    from prettytable import PrettyTable
    from preprocessing import compute_entropy
    import helper as hp

    table = PrettyTable()
    table.title = "Entropies"
    table.field_names = ["State", "Entropy", "Efficiency"]

    if not list:
        entropies = compute_entropy( df, df.columns )
    else:
        entropies = compute_entropy( df, list )

    sorted_entropies = sorted( entropies.items(), key=lambda kv: kv[1] )

    try:
        for state, ent in sorted_entropies:
            table.add_row(
                    [state, round( ent[0], hp.PRECISION ), round( ent[1], hp.PRECISION )]
            )
        print( table )
    except TypeError:
        return


def var_questions_wrapper(type, message, values=None):
        if type == "confirm":
            questions_variables = [
                {
                    'type':    'confirm',
                    'message': message,
                    'name':    'var',
                    'default': True,
                }
            ]
        else:
            if isinstance(values, pd.DataFrame) :
                questions_variables = [
                    {
                        'type': type,
                        'name': 'var',
                        'message': message,
                        'choices': make_list_of_dict(values)
                    }
                ]
            else:
                questions_variables = [
                    {
                        'type': type,
                        'name': 'var',
                        'message': message,
                        'choices': values
                    }
                ]

        return question_prompt(["var"], questions_variables)["var"]

# ask the user for multiple (state, value) paits and return as evidence dictionary
def ask_multiple_evidences(df, v2cmap, to_drop=None):
    to_drop = [to_drop] if to_drop and not isinstance( to_drop, list ) else to_drop

    # iteratively ask for variables to add to evidence
    evidences = {}
    while True:
        choice = var_questions_wrapper( "confirm",
                                           "Add new variable to evidence:" )

        if not choice:
            break

        # remove already proposed variables from choices
        columns_to_drop = [i for i in evidences.keys()]
        if to_drop:
            columns_to_drop.extend( to_drop )
        df_values_dropped = df.drop( columns_to_drop, 1, inplace=False )

        state = var_questions_wrapper( "list",
                                          "Which variable do you want to add as evidence?",
                                          df_values_dropped )

        state_value = var_questions_wrapper( "list",
                                                "Value",
                                                list( df[state].unique() )
                                                )

        # build up evidence dict for query
        evidences[state] = v2cmap[state][state_value]

    return evidences