import datetime

from colorama import Fore
from switch import Switch

import helper as hp
import machine_learning as ml
import preprocessing as pp
import bayesian as bn


if __name__ == '__main__':
    print( Fore.GREEN + f"--------------------------------------------------------" )
    print( "BAYESIAN NETWORK INTERFACING TOOL - Thomas Tiotto (2019)" )
    print( f"--------------------------------------------------------" )
    print(Fore.RESET + "")

    data_set_path = '../DBMedico/DBBCTI_20042014_VMMZ_GL.xls'

    data_set_path = input( f"Data set [{data_set_path}] : " ) or data_set_path

    df = hp.read_dataset( data_set_path, "excel", sheet="DB3" )

    print( f"Number of records in data set before cleaning: {len( df )}" )

    # drop records according to specifications
    df = pp.drop_records( df )

    print( f"Number of records in data set after cleaning: {len( df )}" )
    print("")

    # bin data in columns according to specifications
    df = pp.bin_records( df )

    NUM_VALUES = len( df.columns )

    # make dictionaries mapping categorical codes to original values and viceversa
    df_values, df_codes, code_to_value_map, value_to_code_map = pp.make_mappings( df, NUM_VALUES )

    # slice original dataset into value and code datasets
    # (df_values, df_codes) = pp.slice_codes(df, NUM_VALUES)

    from PyInquirer import Separator

    questions_actionlv1 = [
        {
            'type':    'list',
            'name':    'actionlv1',
            'message': 'What do you want to do?',
            'choices': [
                        Separator(),
                        "   Build Bayesian Network",
                        Separator(),
                        "   Inspect data set",
                        "   ML",
                        Separator(),
                        "Exit"
                        ]
        }
    ]
    questions_actionlv2ml = [
        {
            'type':    'list',
            'name':    'actionlv2ml',
            'message': 'What do you want to do?',
            'choices': [Separator(),
                        "Find best algorithm",
                        "Classify",
                        Separator(),
                        "Back"]
        }
    ]
    questions_actionlv2bn = [
        {
            'type':    'list',
            'name':    'actionlv2bn',
            'message': 'What do you want to do?',
            'choices': [Separator("Exploration"),
                        "   Plot model",
                        "   Marginals",
                        "   Entropies",
                        "   Independencies",
                        Separator("Updating"),
                        "   Conditional probability query",
                        "   MPE query",
                        "   Pseudo-MPE query",
                        Separator("Dialogues"),
                        "   Exhaustive dialogue",
                        "   Independencies dialogue",
                        "   Threshold dialogue",
                        "   Branch probabilities",
                        Separator("Other"),
                        "   Classify (WIP)",
                        "   Compare MPE algorithms",
                        "   Export to UAI",
                        Separator(),
                        "Back"]
        }
    ]
    questions_actionlv2inspect = [
        {
            'type':    'list',
            'name':    'actionlv2inspect',
            'message': 'What do you want to do?',
            'choices': ["Variable values distribution",
                        "Entropies",
                        Separator(),
                        "Back"]
        }
    ]

    model = None
    mpe_ready = False

    # menu loop
    while True:
        with Switch( hp.question_prompt( ["actionlv1"], questions_actionlv1 )["actionlv1"] ) as case:

            if case( "   Inspect data set" ):
                while True:
                    with Switch( hp.question_prompt( ["actionlv2inspect"], questions_actionlv2inspect )[
                                     "actionlv2inspect"] ) as case1:
                        if case1( "Back" ):
                            break

                        if case1( "Variable values distribution" ):
                            var = hp.var_questions_wrapper("checkbox",
                                                           "Which variables do you want to inspect?",
                                                           df_values)
                            pp.print_variable_hist( df_values, var )

                        if case1( "Entropies" ):
                            var = hp.var_questions_wrapper("checkbox",
                                                           "Which variables do you want to inspect?",
                                                           df_values)
                            hp.print_entropies_table( df_values, var )

            if case( "Exit" ):
                break

            if case( "   ML" ):
                while True:
                    with Switch( hp.question_prompt( ["actionlv2ml"], questions_actionlv2ml )["actionlv2ml"] ) as case2:
                        if case2( "Back" ):
                            break

                        if case2( "Find best algorithm" ):
                            var = hp.var_questions_wrapper("checkbox",
                                                           "Which variables do you want to mark for classification?",
                                                           df_values)
                            # test algorithms to find best
                            ml.test_classification_algorithms( df_codes, var )

                        if case2( "Classify" ):
                            print( "WIP" )
                        # TODO give choice of algorithm and use best one found

            if case( "   Build Bayesian Network" ):
                print( "Building Bayesian Network from",  data_set_path, "...")
                model = bn.BN( df_codes, df_values, code_to_value_map, value_to_code_map )
                mpe_graph = None

                while True:
                    with Switch( hp.question_prompt( ["actionlv2bn"], questions_actionlv2bn )["actionlv2bn"] ) as case3:

                        ##################### EXPLORATION #####################
                        if case3( "   Plot model" ) and model:
                            # plot with original column names
                            model.plot_model( independenciesplot=False, showconnectedness=True )

                        if case3( "   Marginals" ) and model:
                            print( "Choose variables of which to print the marginal (a priori of evidence) probability" )

                            var = hp.var_questions_wrapper("checkbox",
                                                           "Which variables do you want to inspect?",
                                                           df_values)
                            model.print_marginals( var, code_to_value_map )

                        if case3( "   Entropies" ) and model:
                            print( "Choose variables of which to print the entropy" )
                            var = hp.var_questions_wrapper("checkbox",
                                                           "Which variables do you want to inspect?",
                                                           df_values)
                            model.print_entropies( var )

                        if case3( "   Independencies" ):
                            print( "Choose a source variable and a set of evidences to see which other variables have influence on the source, given the evidence." )
                            var_source = hp.var_questions_wrapper("list",
                                                           "Which variable do you want to check for independencies?",
                                                           df_values)
                            # remove already added variables from choices
                            df_values_dropped = df_values.drop(var_source, 1, inplace=False)
                            var_evidence = hp.var_questions_wrapper("checkbox",
                                                           "Which variables do you want to add as evidence?",
                                                           df_values_dropped)

                            separations = model.d_separated( var_source, var_evidence )
                            model.plot_model( independenciesplot=True, evidence=var_evidence, separations=separations, source=var_source, showconnectedness=False )

                            print("Given" + Fore.MAGENTA + " source " + Fore.RESET + "variable " + Fore.MAGENTA + var_source + Fore.RESET + " and given " + Fore.GREEN + "evidence: " + Fore.RESET )
                            for ev in var_evidence:
                                print( Fore.GREEN + f"{ev}" + Fore.RESET + ", ", end="" )
                            print("")
                            print("the following variables have no effect on " + Fore.MAGENTA + var_source + Fore.RESET + ":")
                            for state, sep in separations.items():
                                if sep and state != var_source:
                                    print( Fore.CYAN + state + Fore.RESET + ", ", end="" )
                            print("\n")

                        ##################### UPDATING #####################
                        if case3( "   Conditional probability query" ):
                            print( "Choose a variable of which to predict the values given the chosen evidence." )
                            var_target = hp.var_questions_wrapper("list",
                                                           "Which variable do you want to predict?",
                                                           df_values)

                            evidences = hp.ask_multiple_evidences(df_values, value_to_code_map, to_drop=var_target)
                            posterior = model.conditional_query(var_target, evidences)

                            print( "Given" + Fore.MAGENTA + " target" + Fore.RESET + " variable " + Fore.MAGENTA + var_target + Fore.RESET + " and observed" + Fore.GREEN + " evidence:" + Fore.RESET )
                            for state, state_value in evidences.items():
                                state_value = code_to_value_map[state][state_value]
                                print( Fore.GREEN + state + Fore.RESET + " with value "  + Fore.GREEN + state_value + Fore.RESET )
                            print( "then the"  + Fore.CYAN + " predicted values"  + Fore.RESET + " for " + Fore.MAGENTA + var_target + Fore.RESET + " are:" )
                            for key, val in posterior["probability"].items():
                                print( Fore.CYAN + code_to_value_map[posterior["state"]][key] + ": " + hp.prob_dict[val] + Fore.RESET + " (" + str( round( val*100, 2 ) ) +"%)" )
                            print("")

                        if case3( "   MPE query" ):
                            print( "Choose a set of evidences to find the most probable assignment of values to remaining variables." )
                            evidences = hp.ask_multiple_evidences( df_values, value_to_code_map )
                            mpe_result = model.mpe_query( evidences )

                            print( f"Given observed" + Fore.GREEN + " evidence:" + Fore.RESET )
                            for state, state_value in evidences.items():
                                state_value = code_to_value_map[state][state_value]
                                print(
                                    Fore.GREEN + f"{state}" + Fore.RESET + " with value " + Fore.GREEN + f"{state_value}" + Fore.RESET )
                            print( "then the" + Fore.CYAN + " most probable configuration" + Fore.RESET + " of the other variables is:" )
                            for state, state_value in mpe_result.items():
                                state_value = code_to_value_map[state][state_value]
                                print(
                                    Fore.CYAN + f"{state}" + Fore.RESET + " with value " + Fore.CYAN + f"{state_value}" + Fore.RESET )

                        if case3( "   Pseudo-MPE query" ):
                            threshold = 0.0
                            threshold = input( f"Choose a threshold [{threshold}]: " )
                            if len(threshold) == 0:
                                threshold = 0.0
                            else:
                                threshold= float( threshold )
                            print(
                                f"Choose a set of evidences to find the ''most probable'' assignment of values to remaining variables.\n Variables less probable than {threshold} will automatically be discarded" )
                            evidences = hp.ask_multiple_evidences( df_values, value_to_code_map )
                            mpe_graph = bn.MPEGraph(evidences, code_to_value_map, model)
                            mpe_graph.generate_pseudo_mpe( threshold=threshold )
                            mpe_graph.plot()

                        ##################### DIALOGUES #####################
                        if case3( "   Exhaustive dialogue" ) and model:
                            print("Given initial evidences, the system proposes the best next (variable, value).",
                                  "\nRefusing a proposal creates alternative explanation branches.",
                                  "\nThe dialogue ends when all variables have been accepted at least once.")
                            initial_evidences = hp.ask_multiple_evidences( df_values, value_to_code_map )

                            mpe_graph = model.dialogue( initial_evidences, mode="exhaustive" )

                        if case3( "   Independencies dialogue" ) and model:
                            print( "Given initial evidences, the system proposes the best next (variable, value).",
                                   "\nVariables that are independent of the ones in evidence are not considered.",
                                   "\nRefusing a proposal creates alternative explanation branches.",
                                   "\nThe dialogue ends when there are no more dependent variables." )
                            initial_evidences = hp.ask_multiple_evidences( df_values, value_to_code_map )

                            mpe_graph = model.dialogue( initial_evidences, mode="separations" )

                        if case3( "   Threshold dialogue" ) and model:
                            low_bound = 0.4
                            ref_bound = 1

                            print( "Given initial evidences, the system proposes the best next (variable, value).",
                                   f"\nProposals that are less probable than {low_bound} or that the user has refused more than {ref_bound+1} times, are discarded.",
                                   "\nRefusing a proposal creates alternative explanation branches.",
                                   "\nThe dialogue ends when there are no more proposals to consider." )
                            initial_evidences = hp.ask_multiple_evidences( df_values, value_to_code_map )

                            mpe_graph = model.dialogue( initial_evidences, mode="threshold", threshold_mode="dynamic", low_bound=low_bound, ref_bound=ref_bound )

                        # TODO da controllare se hanno senso
                        if case3( "   Branch probabilities" ):
                            print( "Choose a branch id from the previously executed dialogue of which to display the total probability." )
                            if not mpe_graph:
                                print( Fore.RED + "Execute dialogue first" + Fore.RESET )
                                continue

                            while True:
                                try:
                                    branch_id = int( input( "Which branch?: " ) )
                                except ValueError:
                                    print( Fore.RED + "Specify a branch" + Fore.RESET )
                                    continue
                                branch_proba = mpe_graph.branch_probability( branch_id )
                                if branch_proba:
                                    print(
                                        Fore.GREEN + f"Probability of branch {branch_id}: {branch_proba}" + Fore.RESET )
                                    break
                                else:
                                    print( Fore.RED + "Inexistent branch" + Fore.RESET )

                        ##################### OTHER #####################
                        if case3( "   Classify (WIP)" ):
                            print( "Choose variables that need to be predicted." )
                            var = hp.var_questions_wrapper("list",
                                                          "Which variable do you want to test on?",
                                                          df_values)

                            # TODO
                            ml.test_bayesian_network( var, df_codes, df_values, code_to_value_map, value_to_code_map )
                            # bn_predictions = model.predict( var )
                            # hp.print_accuracy( df_codes, bn_predictions, var )

                        if case3( "   Compare MPE algorithms" ) and model:
                            dict_algorithms = { 0: "pseudo-MPE", 1: "DAOOPT", 2: "pgmpy" }
                            dict_algorithms_inv = { "pseudo-MPE": 0, "DAOOPT": 1, "pgmpy": 2 }

                            iterations = int( input( "Compare over how many iterations?: " ) )
                            print("")

                            iter_results = []
                            i = 0
                            while i < iterations:
                                if i != iterations:
                                    print( 'Iteration {} of {}'.format( i+1, iterations ), end='\r' )
                                else:
                                    print( 'Iteration {} of {}'.format( i+1, iterations ) )

                                mpe_solutions = []
                                evidence = model.initial_random_evidence()

                                # pseudo-MPE
                                graph = bn.MPEGraph( evidence, code_to_value_map, model )
                                pseudo_mpe = graph.generate_pseudo_mpe( threshold=0 )
                                mpe_solutions.append( pseudo_mpe )

                                # MPE DAOOPT
                                # path = "../export/"
                                #
                                # model.export_evidence_to_uai( path + "evidence.uai.evid", evidence, format="pgmpy" )
                                #
                                # from pgmpy.readwrite import UAIWriter
                                # writer = UAIWriter( model.model_pgmpy )
                                # writer.write_uai( path + "pgmpy.uai" )
                                #
                                # mpe_daoopt = bn.daoopt_solver( path + "pgmpy.uai" , path + "evidence.uai.evid", path + "sol_pgmpy.txt", model.node_names, evidence, "pgmpy", model.model_pgmpy )
                                # # some evidences bring DAOOPT to error so we skip them
                                # if mpe_daoopt == -1:
                                #     # print( Fore.RED, "DAOOPT had issues with this evidence, skipping and trying another", Fore.RESET )
                                #     mpe_solutions.pop()
                                #     # i -= 1
                                #     continue
                                # mpe_solutions.append( mpe_daoopt )

                                # MPE pgmpy
                                mpe_pgmpy_ve = model.mpe_query( evidence, algorithm="Variable Elimination" )
                                mpe_solutions.append( mpe_pgmpy_ve )

                                comparison_result = bn.compare_mpes( mpe_solutions )

                                iter_results.append( comparison_result )

                                i += 1

                            questions_algorithms = [
                                {
                                    'type':    "list",
                                    'name':    'alg',
                                    'message': "Which algorithm to you want to compare to others?",
                                    'choices': [ "pseudo-MPE", "DAOOPT", "pgmpy", Separator(), "Back" ]
                                }
                            ]
                            while True:
                                alg = hp.question_prompt( ["alg"], questions_algorithms )["alg"]
                                if alg !=  "Back":
                                    to_compare = dict_algorithms_inv[alg]

                                    from collections import Counter

                                    # sum values for each metric across iterations
                                    result = Counter()
                                    for elem in iter_results:
                                        for key, value in elem.items():
                                            result[key] += value

                                    # calculate average for each metric and display it
                                    print( Fore.CYAN + "Average distances of", alg, "to others:" )
                                    score_to_compare = bn.score_mpes( result, to_compare, len( iter_results ) )
                                    for elem in score_to_compare:
                                        print( "\t" + elem + ": " + str( score_to_compare[elem] ) )
                                    print( Fore.RESET )
                                else:
                                    break

                        if case3( "   Export to UAI" ):
                            path = "../export"

                            model.export_model_to_uai( path )
                            evidence = model.initial_random_evidence()
                            model.export_evidence_to_uai( path, evidence )

                            print( "Exported model and randomly generated evidence exported in UAI format to " + str( path ) )

                        if case3( "Back" ):
                            break