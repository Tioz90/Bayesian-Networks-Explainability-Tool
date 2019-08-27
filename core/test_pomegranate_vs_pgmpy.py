import datetime

from colorama import Fore

import helper as hp
import preprocessing as pp
import bayesian as bn

if __name__ == '__main__':
    print( Fore.GREEN + f"-----------------------------------------" )
    print( f"{datetime.datetime.now()}" )
    print( f"-----------------------------------------" )

    df = hp.read_dataset( '../DBMedico/DBBCTI_20042014_VMMZ_GL.xls', "excel", sheet="DB3" )

    print( f"Number of lines before cleaning: {len( df )}" )

    # drop records according to specifications
    df = pp.drop_records( df )

    print( f"Number of lines after cleaning: {len( df )}" )

    # bin data in columns according to specifications
    df = pp.bin_records( df )

    NUM_VALUES = len( df.columns )

    # make dictionaries mapping categorical codes to original values and viceversa
    df_values, df_codes, code_to_value_map, value_to_code_map = pp.make_mappings( df, NUM_VALUES )

    # slice original dataset into value and code datasets
    # (df_values, df_codes) = pp.slice_codes(df, NUM_VALUES)

    from bayesian import BN

    print( "Building Bayesian Network ..." )
    model = BN( df_codes, df_values, code_to_value_map, value_to_code_map )

    #############################################################
    G = model.convert_to_pgmpy()

    from pgmpy.inference import VariableElimination
    G_infer = VariableElimination( G )

# test conditional query
    var_target = hp.var_questions_wrapper( "list",
                                           "Which variables do you want to predict?",
                                           df_values )

    evidences = hp.ask_multiple_evidences( df_values, value_to_code_map, to_drop=var_target )

    # with pomegranate
    posterior = model.conditional_query( var_target, evidences )

    print( f"Given observed" + Fore.GREEN + " evidence:" + Fore.RESET )
    for state, state_value in evidences.items():
        state_value = code_to_value_map[state][state_value]
        print( Fore.GREEN + f"{state}" + Fore.RESET + " with value " + Fore.GREEN + f"{state_value}" + Fore.RESET )
    print( "then the" + Fore.CYAN + " output" + Fore.RESET + " is:" )
    bn.summarise_variable( posterior, code_to_value_map )

    # with pgmpy
    reply = G_infer.query( variables=[var_target], evidence=evidences ) # returns DiscreteFactor
    print( reply )

    # test joint
    reply = G_infer.query( variables=["mut17q21"], evidence=evidences )
    print( reply )
    reply = G_infer.query( variables=["loss 17"], evidence=evidences )
    print( reply )
    reply = G_infer.query( variables=["mut17q21", "loss 17"], evidence=evidences )
    print( reply )

    #  MAP query
    reply = G_infer.map_query( variables=[var_target], evidence=evidences ) #returns dict
    print( reply )
    #  MPE query
    # TODO
    reply = G_infer.map_query( variables=[var_target], evidence=evidences ) #returns dict
    print( reply )


# test independencies
    var_source = hp.var_questions_wrapper( "list",
                                           "Which variable do you want to check for independencies?",
                                           df_values )
    # remove already added variables from choices
    df_values_dropped = df_values.drop( var_source, 1, inplace=False )
    var_evidence = hp.var_questions_wrapper( "checkbox",
                                             "Which variables do you want to add as evidence?",
                                             df_values_dropped )

    # with pomegranate
    separations = model.d_separated( var_source, var_evidence )
    model.plot_model( independenciesplot=True, evidence=var_evidence, separations=separations, source=var_source,
                      showconnectedness=True )

    # with pgmpy
    reply = G.active_trail_nodes(var_source) # returns dict
    print( reply )


