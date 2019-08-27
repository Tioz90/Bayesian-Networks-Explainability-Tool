import pandas as pd
import sklearn as sk

import helper as hp


# drop dataframe records
def drop_records(df):
    df = df.drop( columns="codice globale" )
    df = df.drop( df[df["lateralita"] == "Sconosciuta"].index )
    df = df.drop( df[df["pT"] == "unuseful"].index )
    df = df.drop( df[df["pM"] == "x"].index )
    df = df.drop( df[df["differenziazione"] == "Sconosciuto o non applicabile"].index )
    df = df.drop( columns="mut17q21" )
    df = df.drop( columns="FISH" )
    df = df.drop( columns="loss 17" )

    df = df.dropna( subset=["recettori estrogeni", "recettori progestinici", "ki67"] )
    df = df.dropna( subset=["c erbB 2"] )  # aspettare conferma
    df = df.dropna( subset=["pT", "pN"] )  # aspettare conferma

    return df


# bin data
def bin_records(df):
    df["eta arrotondata"] = pd.cut( df["eta arrotondata"], bins=[0, 40, 50, float( "inf" )],
                                    labels=["<40", "40-50", ">50"], include_lowest=True )
    df["ki67"] = pd.cut( df["ki67"], bins=[0, 14, 20, 30, float( "inf" )], labels=["<14", "14-20", "20-30", ">30"],
                         include_lowest=True )
    df["recettori estrogeni"] = pd.cut( df["recettori estrogeni"], bins=[0, 10, 50, float( "inf" )],
                                        labels=["negativo", "debolmente positivo", "fortemente positivo"],
                                        include_lowest=True )
    df["recettori progestinici"] = pd.cut( df["recettori progestinici"], bins=[0, 10, 50, float( "inf" )],
                                           labels=["negativo", "debolmente positivo", "fortemente positivo"],
                                           include_lowest=True )

    df.loc[df["pN"] != "0", "pN"] = "!0"
    # df["pN"] = df["pN"].astype("category")
    # df["pN"] = pd.cut(df["pN"], bins=[float("-inf"), 0, float("inf")], labels=["0", "!0"])
    # df["pT"] = df["pT"].astype("category")

    # for col in list(df.columns):
    # 	if df[col].dtype == "object" or df[col].dtype == "category":
    # 		df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes  # convert object to categorical codes

    # crea colonne con encoding dei valori e salva dizionario di mapping tra i due

    # for col in df:
    # 	print(f"{col}: {len(df[col].unique())}")

    return df


# invert mapping of a dictionary
def invert_dict(dict):
    inv_map = {v: k for k, v in dict.items()}

    return inv_map


# make dictionary mapping
def make_mappings(df, num_values):
    import copy

    df_copy = copy.copy( df )

    code_to_value_map = {}
    for col in list( df_copy.columns ):
        df_copy[col] = df_copy[col].astype( "category" )
        df[col] = df[col].astype( "category" )
        df_copy[col] = df_copy[col].cat.codes
        code_to_value_map[col] = dict( enumerate( df[col].cat.categories ) )

    # create inverted dictionary
    value_to_code_map = {}
    for key in code_to_value_map:
        value_to_code_map[key] = invert_dict( code_to_value_map[key] )

    return df, df_copy, code_to_value_map, value_to_code_map


# slice dataset into original values and encoded ones
def slice_codes(df, column_num):
    values = list( range( 0, column_num ) )
    codes = list( range( column_num, 2 * column_num ) )

    df_values = df.iloc[:, values]
    df_codes = df.iloc[:, codes]
    # mapping_dict = codes_to_states(df_values, df_codes)
    df_codes = sk.utils.shuffle( df_codes, random_state=0 )

    return df_values, df_codes


def print_variable_values(df, var_list):
    if not isinstance( var_list, list ):
        var_list = [var_list]

    for col in var_list:
        counts = df[col].value_counts().to_dict()
        total = len( df[col] )

        for var in df[col].unique():
            print( f"{var}: {hp.probability( counts[var], total ):{hp.WIDTH}.{hp.PRECISION}f}" )


def print_variable_hist(df, var_list):
    from ascii_graph import Pyasciigraph

    for col in var_list:
        print( "" )
        graph = Pyasciigraph()
        # get count of each value of each selected variable, create a list of tuples and print the histogram

        for line in graph.graph( f"{col} absolute value distribution",
                                 list( zip( df[col].value_counts().index, df[col].value_counts() ) ) ):
            print( line )
        print_variable_values( df, col )
    print( "" )


# compute the normalised entropy of given variables
def compute_entropy(df, var_list):
    from math import log

    entropies = {}
    ent = 0
    # base = e if base is None else base

    for col in var_list:
        entropies[col] = []
        counts = df[col].value_counts()
        rows = df.shape[0]
        counts = counts / rows
        base = len( df[col].unique() )

        for p_i in counts:
            ent -= p_i * log( p_i, 2 )

        entropies[col].append( ent )
        entropies[col].append( ent / log( len( df[col].unique() ), 2 ) )

    return entropies
