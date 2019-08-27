import networkx as nx
from colorama import Fore
import numpy as np

import helper as hp


def daoopt_solver(uai_path, evid_path, export_path, nodes, evidence, format, model_pgmpy):
    import subprocess, os

    if format == "pomegranate":
        # immutable index of variables
        index = {k: v for v, k in enumerate( nodes )}
        inverse_index = {k: v for k, v in enumerate( nodes )}

    if format == "pgmpy":
        from pgmpy.readwrite import UAIWriter

        writer = UAIWriter( model_pgmpy )
        domain = writer.get_domain()
        domain = sorted( domain.items(), key=lambda x: (x[1], x[0]) )

    f = open( export_path, "w" )

    # call DAOOPT and capture its output from file
    mydir = os.getcwd()
    cmd = [ mydir + "/daoopt",
            "-f", uai_path,
            "-e", evid_path
            ]
    FNULL = open( os.devnull, 'w' )
    result = subprocess.run( cmd, stdout=subprocess.PIPE, stderr=FNULL )
    f.write( result.stdout.decode( 'utf-8' ) )

    f = open( export_path, "r" )
    # f = open( os.getcwd() + "/pomegranate.uai.MPE", "r" )
    # For example, an input model with 3 binary variables may have a solution line: 3 0 1 0
    try:
        assignments = [ int(i) for i in [ i for i in f.read().split( '\n' ) if i ][ -1 ].split()[3:None] ]
        if len( assignments ) != len( nodes ):
            return -1
    except ValueError:
        return -1

    result = {}
    for i, v in enumerate( assignments ):
        if format == "pomegranate":
            result[ inverse_index[i] ] = v

        if format == "pgmpy":
            result[ domain[i][0] ] = v

    # remove keys in evidence from solution
    for k in evidence.keys():
        result.pop( k, None )

    # return sorted( result.items(), key=lambda x: (x[0], x[1]) )
    return result

# return average score of algorithm against all others
def score_mpes(results, to_score, iterations):
    scores = {}
    for dist, res in results.items() :
        scores[dist] = np.mean( [ x for j, x in enumerate( res[to_score, :] ) if j != to_score ] ) / iterations

    return scores


# calculate confusion matrices for each distance metric
def compare_mpes(solutions):
    results = {}
    values = []

    # sort state assignments for easy comparison
    for sol in solutions:
        values.append( [ x[1] for x in sorted( sol.items(), key=lambda k: k[1] ) ] )

    # Hamming distance
    def ham(i, j):
        from scipy.spatial.distance import hamming
        return hamming( values[ i ], values[ j ] )
    # Jaccard distance
    def jac(i, j):
        s1 = set( values[ i ] )
        s2 = set( values[ j ] )
        return 1.0 - len( s1.intersection( s2 ) ) / len( s1.union( s2 ) )

    # confusion matrix of results
    results["hamming"] = np.zeros( ( len( solutions ), len( solutions ) ) )
    results["jaccard"] = np.zeros( ( len( solutions ), len( solutions ) ) )
    for i in range( len( solutions ) ):
        for j in range( len( solutions ) ):
            results["hamming"][i, j] = ham(i, j)
            results["jaccard"][i, j] = jac(i, j)

    return results


# compute the entropy of a categorical variable
def entropy(distr, normalized=False):
    from math import log

    ent = 0

    for p_i in distr:
        if p_i != 0:
            ent -= p_i * log( p_i, 2 )

    if normalized:
        ent /= log( len( distr ), 2 )

    return ent


# return list of keys given dictionary of evidence
def translate_evidence(evidence):
    if isinstance( evidence, dict ):
        return list( evidence.keys() )


# print a histogram of probabilities of given variable
def summarise_variable(var, code_to_value_map):
    from ascii_graph import Pyasciigraph

    graph = Pyasciigraph()

    for line in graph.graph( Fore.CYAN + f"{var['state']}"  + Fore.RESET + " has value distribution (probability value)" + Fore.CYAN ,
                             [ (code_to_value_map[var['state']][k], v * 100) for k, v in var["probability"].items() ] ):

        print( line )
    # print_variable_values( df, col )
    print( "" + Fore.RESET )


class BN:
    def __init__(self, df_codes, df_values, code_to_value_map, value_to_code_map):
        import pomegranate as pg

        # learn network structure from dataset of codes with original column names
        # algorithm: one of 'chow-liu', 'greedy', 'exact',
        self.model = pg.BayesianNetwork.from_samples( df_codes.values,
                                                      algorithm="greedy",
                                                      state_names=df_codes.columns,
                                                      n_jobs=-1 )

        self.epsilon_smoothing()
        self.nodes = self.model.states
        self.node_names = [ s.name for s in self.nodes ]
        self.evidence = {}
        self.df = df_values
        self.df_cod = df_codes
        self.code2value = code_to_value_map
        self.value2code = value_to_code_map

        # convert to pgmpy for easy calculations
        self.model_pgmpy = self.convert_to_pgmpy()

    # add a small positive constant to zero-valued entries in CPTs while maintaining normalisation
    def epsilon_smoothing(self):
        epsilon = np.nextafter(0, 1)

        for k, state in enumerate(self.model.states):
            card = self.get_cardinality(state.name)

            if state.distribution.name == "DiscreteDistribution":
                num_zero = list(state.distribution.parameters[0].values()).count(0)
                num_non_zero = card - num_zero
                for key, val in state.distribution.parameters[0].items():
                    if val == 0:
                        state.distribution.parameters[0][key] += epsilon * num_non_zero
                    else:
                        state.distribution.parameters[0][key] -= epsilon * num_zero

            if state.distribution.name == "ConditionalProbabilityTable":
                # extract CPT to easily work on it
                cpt = np.array([ x[-1] for x in state.distribution.parameters[0] ]).reshape((card,-1), order="F")
                for j in range(cpt.shape[1]):
                    num_zero = list( cpt[:,j] ).count( 0 )
                    num_non_zero = card - num_zero
                    for i in range( cpt[:,j].shape[0] ):
                        if cpt[i,j] == 0:
                            cpt[i,j] += epsilon * num_non_zero
                        else:
                            cpt[i,j] -= epsilon * num_zero

                # put modified elements back into the model
                for l,i in enumerate( np.nditer( cpt, order='F' ) ):
                    self.model.states[k].distribution.parameters[0][l][-1] = i

        return

    def dialogue(self, initial_evidences, mode, threshold_mode="", low_bound=None, ref_bound=None ):
        # if the probability of a (state, value) pair is smaller than lower_bound then refuse automatically
        boundaries = { key: {} for key in self.node_names }

        if mode == "threshold":
            for state in self.node_names:
                boundaries[state]["refuse_bound"] = { key: 0 for key in self.df_cod[state].unique() }
                if threshold_mode == "dynamic":
                    # if better than random
                    boundaries[state]["lower_bound"] = 1.0 / self.get_cardinality(state)
            # upper_bound *= upp_bound

        # create explanation polytree rooted in initial evidence
        mpe_graph = MPEGraph( initial_evidences, self.code2value, self )

        # add initial evidence
        self.evidence = initial_evidences

        # propose states to user and update evidence based on replies
        while True:
            # show the explanation polytree at each step
            mpe_graph.plot()

            if mode == "separations":
                # calculate d-separations
                separated = self.evidence_d_separation()

            # returns ordered list of most probable states tagged with original column names
            mpe_states = self.next_most_probable_states()

            # remove unprobable or too many times refused states from mpe_states
            if mode == "threshold" and mpe_states:
                mpe_states_tmp = []
                for state in mpe_states:
                    if threshold_mode == "dynamic":
                        if state["probability"] >= boundaries[state["state"]]["lower_bound"] \
                                and boundaries[state["state"]]["refuse_bound"][state["value"]] <= ref_bound:
                            mpe_states_tmp.append(state)
                    else:
                        if state["probability"] >= low_bound \
                                and boundaries[state["state"]]["refuse_bound"][state["value"]] <= ref_bound:
                            mpe_states_tmp.append(state)
                mpe_states = mpe_states_tmp

            if mode == "separations":
                # remove d-separated nodes from mpe_states
                mpe_states = [ x for x in mpe_states if not separated[x["state"]] ]

                # display current model with d-separations
                self.plot_model( independenciesplot=True, evidence=self.evidence, separations=separated, showconnectedness=True)

            var_still_explain = list( set( self.node_names ) - set( [ state for state in self.evidence.keys() ] ) )
            print("Variables still to explain:" + Fore.GREEN, end=" ")
            for var in var_still_explain:
                print(var, end=", ")
            print("" + Fore.RESET)
            # hp.print_probability_table(mpe_states, self.code2value)

            # while we have more states to propose
            if mpe_states:
                next_state = mpe_states[0]
                print("*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
                print(f"The next most probable state is {hp.prob_dict[next_state['probability']]} " + Fore.CYAN +  f"{next_state['state']} " + Fore.RESET + "with value " + Fore.CYAN + f"{self.code2value[next_state['state']][next_state['value']]}" + Fore.RESET)
            else:
                print(Fore.GREEN + "End of predictions")
                break

            choice = hp.var_questions_wrapper("confirm",
                                              "Enter to accept or n to refuse:")
            # if the proposed state is not accepted by expert
            if not choice:
                # increment counter for refusal
                if mode == "threshold":
                    boundaries[next_state["state"]]["refuse_bound"][next_state["value"]] +=1

                for alternative_state in mpe_states[1:]:
                    choice = hp.var_questions_wrapper("confirm",
                                                      f"Is state {alternative_state['state']} with value {self.code2value[alternative_state['state']][alternative_state['value']]} more correct?")

                    # if the alternatively proposed state is accepted by expert
                    if choice:
                        # generate counterfactual branch of explanation tree
                        mpe_graph.generate_alternative_branch( mode=mode, boundaries=boundaries, threshold_mode=threshold_mode, low_bound=low_bound, ref_bound=ref_bound )
                        # add chosen state as node in explanation tree
                        # mpe_tree.add_branch_node(alternative_state)
                        mpe_graph.add_node( alternative_state )
                        # append next most probable state to evidence dictionary
                        self.evidence[alternative_state["state"]] = alternative_state["value"]
                        break
                    # if the expert doesn't accept the alternative state
                    elif not choice:
                        if mode == "threshold":
                            boundaries[alternative_state["state"]]["refuse_bound"][alternative_state["value"]] += 1
                        continue
                # if we have exhausted all alternative states on exiting the proposal loop
                if not choice:
                    print(Fore.YELLOW + "No more suggestions" + Fore.RESET)
                    exit(0)
            elif choice:
                # mpe_tree.add_branch_node(next_state)
                mpe_graph.add_node( next_state )
                # append next most probable state to evidence dictionary
                self.evidence[next_state["state"]] = next_state["value"]

        return mpe_graph

    # MAP query using pgmpy
    def map_query(self, targets, evidences, algorithm ):
        if algorithm == "Variable Elimination":
            from pgmpy.inference import VariableElimination
            model_infer = VariableElimination( self.model_pgmpy )
        if algorithm == "Belief Propagation":
            from pgmpy.inference import BeliefPropagation
            model_infer = BeliefPropagation( self.model_pgmpy )
        if algorithm == "MPLP":
            from pgmpy.inference import Mplp
            model_infer = Mplp( self.model_pgmpy.to_markov_model() )

        return model_infer.map_query( variables=list( targets ), evidence=evidences )

    # MPE query using pgmpy
    def mpe_query(self, evidences, algorithm="Variable Elimination"):
        targets = set( self.node_names ) - set( evidences )

        query = self.map_query( targets, evidences, algorithm )

        # return sorted( query.items(), key=lambda x: (x[0], x[1]) )
        return query

    # execute conditional probability query
    def conditional_query(self, target, evidences):
        # if no evidence was supplied we want the marginals
        if not evidences:
            evidences = {}

        posterior = self.model.predict_proba( evidences, n_jobs=-1 )

        if all( isinstance( state, (int, np.integer) ) for state in posterior ):
            return None

        # state_counter to keep track of state names as they are not saved in DiscreteDistribution objects
        for state_counter, state in enumerate( posterior ):
            if not isinstance( state, (int, np.integer) ) and self.df.columns[state_counter] == target:  # if the value is not int, it's a DiscreteDistribution
                # save state name together with value and probability, limitation of pomegranate
                target_posterior = {
                                    "state":       self.df.columns[state_counter],
                                    "probability": state.parameters[0]
                                    }

                return target_posterior

    def get_cardinality(self, name):
        for state in self.model.states:
            if state.name == name:
                return len( state.distribution.keys() )

        return 0

    def convert_to_pgmpy(self):
        from pgmpy.models import BayesianModel
        from pgmpy.factors.discrete import TabularCPD
        G = BayesianModel()

        # representation of the BN
        import json
        model_dict = json.loads( self.model.to_json() )

        # immutable index of variables
        index = {k: v for v, k in enumerate( self.node_names )}
        inverse_index = {k: v for k, v in enumerate( self.node_names )}

        # add nodes
        for node in model_dict["states"]:
            G.add_node(node["name"])

        # add edges
        for child, edge in enumerate(model_dict["structure"]):
            if edge:
                for father in edge:
                    G.add_edge( inverse_index[father], inverse_index[child] )

        # define individual CPDs
        for state in model_dict["states"]:
            if state["distribution"]["name"] == "DiscreteDistribution":
                cardinality = len( state["distribution"]["parameters"][0] )

                ordered_tuples = np.array( sorted( state["distribution"]["parameters"][0].items(), key=lambda x: x[0] ) )
                values = np.expand_dims( ordered_tuples[:, 1], axis=0 )

                G.add_cpds( TabularCPD( variable=state["name"], variable_card=cardinality, values=values ) )

            if state["distribution"]["name"] == "ConditionalProbabilityTable":
                table = np.array( state["distribution"]["table"] )
                table = table.astype( float )

                cardinality = int( max( table[:, -2] ) + 1 )

                evidence = [ inverse_index[p] for p in model_dict["structure"][index[state["name"]]] ]
                ev_cardinality = np.zeros( len( evidence ) )
                for i in range(0, len( evidence )):
                    ev_cardinality[i] = int( max( table[:, i] ) ) + 1

                # num_evidences = int( len( state["distribution"]["table"] ) / cardinality )
                # values = np.zeros( (cardinality, num_evidences ) )
                # for i in range( num_evidences ):
                #     values[:, i] = np.array([ x[-1] for x in state["distribution"]["table"][i*cardinality: (i+1)*cardinality] ])
                values = np.reshape( table[:,-1], (cardinality,-1), order='F' )

                G.add_cpds( TabularCPD( variable=state["name"], variable_card=cardinality, values=values, evidence=evidence, evidence_card=ev_cardinality ) )

        if not G.check_model():
            raise ValueError( "Something is wrong in the converted model" )

        return G

    # see https://github.com/radum2275/merlin for UAI format description
    def export_evidence_to_uai(self, path, evidence, format):
        f = open( path, "w+" )

        if format == "pomegranate":
            import json

            # representation of the BN
            model_dict = json.loads( self.model.to_json() )

            # immutable index of variables
            index = {k: v for v, k in enumerate( self.node_names )}
            inverse_index = {k: v for k, v in enumerate( self.node_names )}

        if format == "pgmpy":
            from pgmpy.readwrite import UAIWriter

            writer = UAIWriter( self.model_pgmpy )
            domain = writer.get_domain()
            domain = sorted( domain.items(), key=lambda x: (x[1], x[0]) )

        # number of variables in the sample
        f.write( str( len( evidence ) ) + "\n" )

        # state/value pairs
        for state in evidence:
            if format == "pomegranate":
                f.write( " " + str( index[state] ) + " " + str( evidence[state] ) + "\n" )

            if format == "pgmpy":
                f.write( " " + str( [x[0] for x in domain].index( state ) ) + " " + str( evidence[state] ) + "\n"  )

        return

    # export the pomegranate model to .uai format for use in external solver
    def export_model_to_uai(self, path, format):
        import json

        f = open( path, "w+" )

        if format != "DAOOPT":
            f.write("c\n")
            f.write( "c Bayesian Network exported from pomegranate - Thomas Tiotto (2019)" + "\n")
            f.write("c\n\n")

        # representation of the BN
        model_dict = json.loads(self.model.to_json())

        # immutable index of variables
        index = { k: v for v, k in enumerate(self.node_names) }
        inverse_index = { k: v for k, v in enumerate(self.node_names) }

        ###### PREAMBLE ######
        # type of network
        f.write( "BAYES\n" )
        # number of variables
        f.write( str( self.model.state_count() ) + "\n" )
        # cardinalities of variables
        for name in self.node_names:
            f.write( str( self.get_cardinality( name ) ) + " " )

        ###### CLIQUES ######
        if format != "DAOOPT":
            f.write( "\n\nc\n" )
            f.write( "c Cliques" + "\n" )
            f.write( "c\n\n" )
        else:
            f.write( "\n" )

        # number of cliques/CPTs
        f.write( str( len( model_dict["structure"] ) ) + "\n" )

        # cliques
        for i, parents in enumerate( model_dict["structure"] ):
            # parents = self.find_parents( state )
            # parents = model_dict["structure"][state]
            f.write( str( len( parents ) + 1 ) + " " )
            # f.write( str( index[state] ) + " "  )
            f.write( str( i ) + " "  )
            for parent in parents:
                # f.write( str( index[parent] ) + " " )
                f.write( str( parent ) + " " )
            f.write( "\n" )

        ###### FUNCTION TABLES ######
        if format != "DAOOPT":
            f.write( "\nc\n" )
            f.write( "c CPTs" + "\n" )
            f.write( "c\n\n" )
        else:
            f.write( "\n" )

        for state in model_dict["states"]:
            if state["distribution"]["name"] == "DiscreteDistribution":
                f.write( str( len( state["distribution"]["parameters"][0] ) ) + "\n" )
                sorted_vals = sorted( state["distribution"]["parameters"][0].items(), key=lambda x: x[0] )
                f.write( " " )
                for val in sorted_vals:
                    f.write( str( val[1] ) + " ")

            if state["distribution"]["name"] == "ConditionalProbabilityTable":
                f.write( str( len( state["distribution"]["table"] ) ) )
                num_states = int( max( map( lambda x: x[-2], state["distribution"]["table"] ) ) ) + 1
                for i, val in enumerate(state["distribution"]["table"]):
                    if i % num_states == 0:
                        f.write("\n")
                        f.write( " " )
                    f.write( val[-1] + " ")
            f.write("\n\n")

        return

    # generate random subset of variables from the dataset
    def initial_random_evidence(self, reproducible=None):
        from numpy.random import randint, choice, seed

        # initialise evidence dictionary
        evidence = {}

        if reproducible:
            seed( 0 )

        # select random subset of random size
        selection_size = randint( 1, len( self.node_names ) )
        selection_names = choice( self.node_names, size=selection_size, replace=False )

        # select random state for each selected variable
        for var in selection_names:
            # var_value = choice(self.df[var].unique())
            var_value = randint( 1, len( self.df[var].unique() ) )
            evidence[var] = var_value

        return evidence

    # workaround to plot the model structure as the built-in function is not working on my setup
    # evidence and separations are list or dict
    def plot_model(self, independenciesplot, evidence=None, separations=None, source=None, directed=False, showconnectedness=False):
        from graphviz import Digraph, Graph

        if separations:
            if source and evidence:
                separations = {k: v for k, v in separations.items() if v and k not in evidence and k not in source}
            elif evidence:
                separations = {k: v for k, v in separations.items() if v and k not in evidence}
            elif source:
                separations = {k: v for k, v in separations.items() if v and k not in source}

        if not directed:
            dot = Graph( '../graphs/bayesian_net', comment=self.model.name, format="png" )
        else:
            dot = Digraph( '../graphs/bayesian_net', comment=self.model.name, format="png" )

        for state in self.model.states:
            # if present, print source node as filled
            if source and state.name in source:
                dot.node( state.name, state.name, style="filled" )
            # if any, print evidence nodes in bold
            elif evidence and state.name in evidence:
                dot.node( state.name, state.name, style="bold" )
            # if any, print d-separated nodes in light grey
            elif separations and state.name in separations:
                dot.node( state.name, state.name, style="dotted", color="lightgrey", fontcolor="lightgrey" )
            else:
                dot.node( state.name, state.name )

        def arc_label(connectedness):
            return str( round( connectedness, 2 ) )

        def arc_width(connectedness):
            return str( abs( 10 * connectedness ) )

        for parent, child in self.model.edges:
            if separations and ( parent.name in separations or child.name in separations):
                dot.edge( parent.name, child.name, style="dotted", color="lightgrey" )
            else:
                if showconnectedness:
                    connectedness = self.mutual_information( parent.name, child.name, self.evidence)

                    if connectedness == -1:
                        dot.edge( parent.name, child.name )
                    else:
                        dot.edge( parent.name, child.name, label=arc_label(connectedness), penwidth=arc_width(connectedness) )
                else:
                    dot.edge( parent.name, child.name )

        if not independenciesplot:
            # basic plot
            label = r'\n\nBayesian Network structure'
        else:
            # d-separation plot
            if source and evidence:
                label = r"\n\nBayesian Network independencies\nFILLED: query source | BOLD: evidence"
            elif source:
                label = r"\n\nBayesian Network independencies\nFILLED: query source"
            elif evidence:
                label = r"\n\nBayesian Network independencies\nBOLD: evidence"
            else:
                label = r"\n\nBayesian Network independencies\n"
        if showconnectedness:
            label += r"\nMutual information shown on edges (not for variables in evidence)"
        dot.attr( label=label )

        dot.render( view=True )

    def mutual_information(self, X, Y, evidence):
        # if one of the variables is already in the evidence set then return because it makes no sense to calculate
        if X in evidence or Y in evidence:
            return -1

        # set up inference using variable elimination algorithm
        from pgmpy.inference import VariableElimination
        model_infer = VariableElimination( self.model_pgmpy )

        # calculate joint distribution
        joint =  model_infer.query( variables=[X, Y], evidence=evidence )

        # calculate marginals from joint
        Y_mar = joint.marginalize([X], inplace=False).values
        X_mar = joint.marginalize([Y], inplace=False).values

        # sometimes order of joint table is inverted, I want to guarantee Y on rows
        if joint.variables[0] != Y:
            XY_joint = np.transpose( joint.values )
        else:
            XY_joint = joint.values

        from math import log
        mutual_info = 0
        for i in range( len( Y_mar ) ):
            for j in range( len( X_mar ) ):
                try:
                    mutual_info += XY_joint[i,j] * log( XY_joint[i,j] / ( Y_mar[i] * X_mar[j] ) )
                except ValueError:
                    # in information theory 0*log(0)=0 so I can skip the value
                    mutual_info = mutual_info

        return mutual_info

    # return sorted list with least entropic next state at the head
    def next_most_probable_states(self, alt_evidence=None):
        import numpy as np

        mpe_states = []
        if alt_evidence:
            posterior = self.model.predict_proba( alt_evidence, n_jobs=-1 )
        else:
            posterior = self.model.predict_proba( self.evidence, n_jobs=-1 )

        if all( isinstance( state, (int, np.integer) ) for state in posterior ):
            return None

        # state_counter to keep track of state names as they are not saved in DiscreteDistribution objects
        for state_counter, state in enumerate( posterior ):
            if not isinstance( state, (int, np.integer) ):  # if the value is not int, it's a DiscreteDistribution
                # save state name together with value and probability, limitation of pomegranate
                mpe_state = {"state":       self.node_names[state_counter],
                             "value":       max( state.parameters[0], key=state.parameters[0].get ),
                             "probability": max( state.parameters[0].values() ),
                             "entropy":     entropy( state.parameters[0].values(), normalized=False ),
                             "efficiency":  entropy( state.parameters[0].values(), normalized=True )}
                mpe_states.append( mpe_state )

        return sorted( mpe_states, key=lambda k: k["efficiency"], reverse=False )

    # for each variable in the list run a prediction using BN model and return dataframe with predictions in columns
    def predict(self, var_list):
        import pandas as pd

        # None columns are marked for prediction
        X = self.df_cod.copy()
        X.loc[:, var_list] = None

        # run prediction
        y_pred = pd.DataFrame( self.model.predict( X.values ), columns=X.columns )

        return y_pred

    # compute the entropy of given variables
    def print_entropies(self, variables):
        from prettytable import PrettyTable

        marginals = self.model.marginal()

        for state_counter, distribution in enumerate( marginals ):
            if self.node_names[state_counter] in variables:
                table = PrettyTable()
                table.title = self.node_names[state_counter]
                table.field_names = ["Value", "Entropy", "Efficiency"]

                table.add_row( [self.node_names[state_counter],
                                round( entropy( marginals[state_counter].values() ), hp.PRECISION ),
                                round( entropy( marginals[state_counter].values(), normalized=True ), hp.PRECISION )
                                ] )
                print( table )

        return

    def print_marginals(self, variables, code_to_value_map):
        from prettytable import PrettyTable

        marginals = self.model.marginal()

        for state_counter, distribution in enumerate( marginals ):
            if self.node_names[state_counter] in variables:

                table = PrettyTable()
                table.title = self.node_names[state_counter]
                table.field_names = ["Value", "Probability"]

                for i, val in enumerate( distribution.values() ):
                    table.add_row(
                            [code_to_value_map[self.df.columns[state_counter]][marginals[state_counter].keys()[i]],
                             round( val, hp.PRECISION )] )
                print( table )

        return

    # Return children of a given node (identified by its name)
    def find_children(self, node):
        children = []

        for parent, child in self.model.edges:
            if node == parent.name:
                children.append( child.name )

        return children

    # Return parents of a given node (identified by its name)
    def find_parents(self, node):
        parents = []

        for i, state in enumerate( self.model.states ):
            if node in self.find_children( state.name ):
                parents.append( state.name )

        return parents

    # Phase 1 of Koller and Friedman SEPARATED algorithm, used to mark all ancestors of nodes
    # in evidence
    def evidence_ancestors(self, evidence):
        import copy

        visit_nodes = copy.copy( evidence )
        ancestors = set()

        while len( visit_nodes ) > 0:
            node = visit_nodes.pop()
            if node not in ancestors:
                for parent in self.find_parents( node ):
                    visit_nodes.append( parent )
            ancestors.add( node )

        return ancestors

    # Phase 2 of Koller and Friedman SEPARATED algorithm,
    def d_separation(self, source, target, evidence):
        ancestors = self.evidence_ancestors( evidence )

        visit_nodes = [(source, "up")]
        visited = set()

        while len( visit_nodes ) > 0:
            (node, direction) = visit_nodes.pop()

            if (node, direction) not in visited:
                if node not in evidence and node == target:
                    return False

                visited.add( (node, direction) )

                if direction == "up" and node not in evidence:
                    for par in self.find_parents( node ):
                        visit_nodes.append( (par, "up") )
                    for chi in self.find_children( node ):
                        visit_nodes.append( (chi, "down") )
                elif direction == "down":
                    if node not in evidence:
                        for chi in self.find_children( node ):
                            visit_nodes.append( (chi, "down") )

                    if node in ancestors:
                        for par in self.find_parents( node ):
                            visit_nodes.append( (par, "up") )

        return True

    # find all d-separated nodes from sources given the evidence
    def d_separated(self, source, evidence):
        separated_from_source = {}

        for target in self.node_names:
            separated_from_source[target] = self.d_separation( source, target, evidence )

        separated_from_source[source] = True

        return separated_from_source

    def evidence_d_separation(self, alt_evidence=None):
        from copy import copy

        if alt_evidence:
            evidence = alt_evidence
        else:
            evidence = self.evidence

        separated_from_source = {}

        for source in evidence:
            # Remove source from evidence or all other nodes result d-separated
            source_evidence = copy( evidence )
            source_evidence.pop( source )
            source_evidence = translate_evidence( source_evidence )

            for target in self.node_names:
                separated_from_source[target] = self.d_separation( source, target, source_evidence )
                if target in evidence:
                    separated_from_source[target] = False
            # separated_from_source[(source, source)] = True

        return separated_from_source


class MPEGraph:
    def __init__(self, init_evidence, values_map, model):
        self.G = nx.DiGraph()
        self.roots = []
        self.map = values_map
        self.model = model
        self.evidence = init_evidence
        self.branch_prefix_num = 1

        # add each state in initial evidence to graph (with probability = 1)
        for node in init_evidence:
            node_value = self.map[node][init_evidence[node]]
            probability = hp.prob_dict[1]
            self.G.add_node( "1 " + node + ": " + str( node_value ) + "\n(initial evidence)",
                             string="State: " + node + ", Value: " + node_value + "\n(initial evidence)",
                             value=node_value,
                             probability=probability,
                             branch_id={1})
            self.roots.append( "1 " + node + ": " + str( node_value ) +  "\n(initial evidence)" )
        self.last_node = self.roots

    # add node as successor of last inserted or of the ones supplied
    def add_node(self, state, parent=None):
        import itertools

        node_value = self.map[state["state"]][state["value"]]
        name_string = "\nState: " + state["state"] + ", Value: " + node_value + \
                      "\n(" + hp.prob_dict[state["probability"]] + ")"

        if not parent:
            node = "1 " + state["state"] + ": " + str( node_value ) +  "\n(" + hp.prob_dict[state["probability"]] + ")"
            branch_id = {1}
        else:
            node = str(self.branch_prefix_num) + " " + state["state"] + ": " + str( node_value ) +  "\n(" + hp.prob_dict[state["probability"]] + ")"
            branch_id = {self.branch_prefix_num}

        self.G.add_node( node,
                         string=name_string,
                         value=node_value,
                         probability=state["probability"],
                         branch_id=branch_id)

        # create edges between last added node(s) and new one
        if not parent:
            self.last_node = [self.last_node] if self.last_node and not isinstance( self.last_node, list ) else self.last_node
            edges = list( itertools.product( self.last_node, [node] ) )
            self.G.add_edges_from( edges )
            # only update main pointer if it's a main branch being created
            self.last_node = node
        else:
            if isinstance( parent, list ):
                edges = list( itertools.product( parent, [node] ) )
                self.G.add_edges_from( edges )
            else:
                self.G.add_edge(parent, node)

            return node

    # printo probability of a given explanation branch
    def branch_probability(self, branch_id):
        branch_probability = 1

        # find only son of evidences
        start_node = list( self.G.neighbors(self.roots[0]) )

        for node in nx.dfs_preorder_nodes(self.G, source=start_node[0]):
            if branch_id in self.G.node[node]["branch_id"]:
                # print(f"{branch_probability} x {self.G.node[node]['probability']}", end='')
                branch_probability *= self.G.node[node]["probability"]
            # print(f" = {branch_probability}")

        if branch_probability != 1:
            return branch_probability
        else:
            return None

    # when user accepts an alternative explanation generate a minimally-entropic alternative branch that
    # includes all remaining variables
    def generate_alternative_branch(self, mode, boundaries=None, threshold_mode="", low_bound=None, ref_bound=None ):
        alt_node = self.last_node
        alt_evidence = self.model.evidence.copy()

        # increment the branch counter so node names are unique
        self.branch_prefix_num += 1

        # also add all ancestors of node to the new branch, ignoring roots
        if alt_node != self.roots:
            for prev in self.G.predecessors(alt_node):
                self.G.node[prev]["branch_id"].add( self.branch_prefix_num )

        # generate the MPE branch rooted as the given state (states[0])
        # for i, state in enumerate(states):
        while True:
            if mode == "separations":
                # calculate d-separations
                separated = self.model.evidence_d_separation( alt_evidence )

            # returns ordered list of most probable states tagged with original column names
            mpe_states = self.model.next_most_probable_states( alt_evidence )

            # remove unprobable or too many times refused states from mpe_states
            if mode == "threshold" and mpe_states:
                mpe_states_tmp = []
                for state in mpe_states:
                    if threshold_mode == "dynamic":
                        if state["probability"] >= boundaries[state["state"]]["lower_bound"] \
                                and boundaries[state["state"]]["refuse_bound"][state["value"]] <= ref_bound:
                            mpe_states_tmp.append( state )
                    else:
                        if state["probability"] >= low_bound \
                                and boundaries[state["state"]]["refuse_bound"][state["value"]] <= ref_bound:
                            mpe_states_tmp.append( state )
                mpe_states = mpe_states_tmp

            if mode == "separations":
                # remove d-separated nodes from mpe_states
                mpe_states = [x for x in mpe_states if not separated[x["state"]]]

            # if we have no more feasible nodes then exit
            if not mpe_states:
                break

            # add alternative branch node
            alt_node = self.add_node( mpe_states[0], parent=alt_node )

            # update alternative evidence (local to this branch)
            alt_evidence[mpe_states[0]["state"]] = mpe_states[0]["value"]

        return

    # generate minimally-entropic assignment to X for P(X|E)
    def generate_pseudo_mpe(self, threshold):
        pseudo_mpe = {}

        # alt_evidence = self.model.initial_random_evidence()
        alt_evidence = self.evidence.copy()

        while True:
            # returns ordered list of most probable states tagged with original column names
            mpe_states = self.model.next_most_probable_states( alt_evidence )

            if not mpe_states:
                break

            # add alternative branch node
            self.add_node( mpe_states[0] )

            if threshold and mpe_states[0]["probability"] <= threshold:
                return pseudo_mpe

            # update alternative evidence (local to this branch)
            alt_evidence[mpe_states[0]["state"]] = mpe_states[0]["value"]

            # update solution
            pseudo_mpe[mpe_states[0]["state"]] = mpe_states[0]["value"]

        return pseudo_mpe

    # plot the graph
    def plot(self):

        from networkx.drawing.nx_agraph import to_agraph

        dot = to_agraph( self.G )
        dot.layout( "dot" )
        dot.draw( "../graphs/pseudo_mpe.png", format="png" )

        from graphviz import view
        view( "../graphs/pseudo_mpe.png" )

        return